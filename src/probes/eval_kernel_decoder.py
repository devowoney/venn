"""Readout of the kernel-aware Gaussian decoder (D-056): reconstruction from the true latent, and the explicit forecast
p(x_hat | H_t) = ∫ N(x; mu(z), diag sigma^2(z)) p(z_hat = z | H_t) dz under the three latent priors, against the flow
decoders trained on the same priors (VALIDATION launches only).

  reconstruction   x from the TRUE z(t): RMSE of mu vs the observed field, mean sigma, NLL, coverage of mu ± 1.645 sigma,
                   consistency |K mu - S| / |S|
  forecast         members z ~ prior (fixed Gaussian N(zhat, Sigma) / network / exaggerated, the same samplers as the
                   flow runs), x = mu(z) + sigma(z) eps: RMSE of the member mean, fair CRPS, spread / RMSE, corr(spread,
                   error) over launches, rank histogram -- next to the flow decoder's numbers for the same prior
  likelihood       KL split (design record §17): log p(x_hat = x_obs) ≈ log p_zhat(K x_obs) + log N(x_obs; mu(K x_obs),
                   sigma^2) + const. The decoder term is the same for every prior; the LATENT LOG SCORE
                   log p_zhat(z_true) is what separates them, computed exactly from each prior's quantile tables and
                   Gaussian copula: log q(z) = log N(eta; 0, R) - sum_i log dq_i/deta (eta_i = the normal score of z_i)
Run:  python -m src.probes.eval_kernel_decoder --kd .tmps/runs_kd/kd_seed0 --seed 0 --out .tmps/eval_kd/seed0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.models.family_net import DramaticSampler          # noqa: E402
from src.models.family_prior import ETA, sample_from_tables   # noqa: E402
from src.models.kernel_decoder import KernelDecoder        # noqa: E402
from src.probes.eval_flow_decoder import fair_crps         # noqa: E402
from src.train.train_kernel_decoder import kernel_data     # noqa: E402

FD, EFD = ".tmps/runs_fd", ".tmps/eval_fd"
PRIORS = ("gauss", "net", "dramatic")
COL = dict(gauss="#eb6834", net="#1baf7a", dramatic="#2a78d6")


def flow_run(seed: int, lead: int, prior: str) -> str:
    """Name of the flow-decoder run trained on this prior (seed 0 has the original names)."""
    if seed == 0:
        return f"fdD_l{lead}" if prior == "dramatic" else f"fdB_l{lead}_{prior}"
    return f"fdS{seed}_l{lead}_{prior}"


def latent_logpdf(off: torch.Tensor, tab: torch.Tensor, Lr: torch.Tensor) -> torch.Tensor:
    """log q(zhat + off) of a table + Gaussian-copula prior: off [n, K] offsets from zhat, tab [n, K, L] quantile tables
    on ETA, Lr [K, K] copula Cholesky -> [n]. Truths outside a table's range get the edge slope (finite, heavily
    penalized)."""
    n, K, L = tab.shape
    g = torch.as_tensor(ETA, dtype=tab.dtype, device=tab.device)
    j = torch.searchsorted(tab.contiguous(), off[..., None].contiguous())[..., 0].clamp(1, L - 1)   # [n, K]
    q0, q1 = tab.gather(-1, (j - 1)[..., None])[..., 0], tab.gather(-1, j[..., None])[..., 0]
    slope = ((q1 - q0) / float(g[1] - g[0])).clamp_min(1e-9)
    eta = g[j - 1] + (off - q0) / slope                       # normal score (linear beyond the table edges)
    y = torch.linalg.solve_triangular(Lr, eta.T, upper=False).T
    log_cop = -0.5 * (y ** 2).sum(1) - torch.log(torch.diagonal(Lr)).sum() - 0.5 * K * np.log(2 * np.pi)
    return log_cop - torch.log(slope).sum(1)


class PriorDraws:
    """Members and quantile tables of one prior for the validation launches (canonical z)."""

    def __init__(self, prior: str, run: str, hf_dir: str, seed: int, val: np.ndarray, dev: str):
        pr = np.load(os.path.join(run, "prior.npz"), allow_pickle=True)
        self.zhat = torch.tensor(pr["zhat"], dtype=torch.float32, device=dev)
        self.row = {int(t): i for i, t in enumerate(val)}
        K = self.zhat.shape[1]
        if prior == "dramatic":
            d = DramaticSampler(run, hf_dir, seed, val, dev)
            self.tab, self.Lr = d.tab, d.Lr
        elif prior == "net":
            q = np.load(os.path.join(run, "netprior.npz"))
            s, m, w = q["sigma"], q["m"][val], q["w"][val]
            self.tab = torch.tensor(s[None, :, None] * (m[..., None] + w[..., None] * ETA), dtype=torch.float32,
                                    device=dev)
            self.Lr = torch.tensor(q["chol_R"], dtype=torch.float32, device=dev)
        else:                                                  # fixed Gaussian N(zhat, Sigma) as table + copula
            Sig = pr["Sigma"]
            s = np.sqrt(np.diag(Sig))
            self.tab = torch.tensor(np.broadcast_to(s[None, :, None] * ETA, (len(val), K, len(ETA))).copy(),
                                    dtype=torch.float32, device=dev)
            self.Lr = torch.tensor(np.linalg.cholesky(Sig / np.outer(s, s)), dtype=torch.float32, device=dev)

    def rows(self, t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor([self.row[int(x)] for x in t.tolist()], device=self.tab.device)

    def sample(self, t: torch.Tensor, M: int) -> torch.Tensor:
        return sample_from_tables(self.zhat[t], self.tab[self.rows(t)], self.Lr, M)

    def logpdf(self, t: torch.Tensor, ztrue: torch.Tensor) -> torch.Tensor:
        return latent_logpdf(ztrue - self.zhat[t], self.tab[self.rows(t)], self.Lr)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--hf_dir", default="/home/sysadmin/jlee/venn/.claude/worktrees/history-forecastor/.tmps/runs_hf")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)
    ck = torch.load(os.path.join(args.kd, "model.pt"), map_location=dev, weights_only=False)
    kd = KernelDecoder(ck["masks"].to(dev), ck["mu"].to(dev), ck["sd"].to(dev), ch=tuple(ck["ch"]), emb=ck["emb"],
                       logvar_range=ck["logvar_range"]).to(dev).eval()
    kd.load_state_dict(ck["state"])
    d = kernel_data(args.hf_dir, args.seed, 8000, dev)
    z, X = d["z"], d["X"]
    T, M = len(X), args.members
    res = dict(seed=args.seed, members=M)

    # --- 1. reconstruction from the true latent ------------------------------------------------------------------------
    val0 = torch.arange(8000, T, device=dev)
    se = nll = cov = sig = cons = 0.0
    for i in range(0, len(val0), 64):
        t = val0[i:i + 64]
        mean, lv = kd(z[t])
        sd_ = torch.exp(0.5 * lv)
        se += float(((X[t] - mean) ** 2).sum())
        nll += float((0.5 * ((X[t] - mean) ** 2 / sd_ ** 2 + lv + np.log(2 * np.pi))).sum())
        cov += float(((X[t] - mean).abs() <= 1.645 * sd_).float().sum())
        sig += float(sd_.sum())
        S = z[t] * kd.sd_s + kd.mu_s
        cons = max(cons, float((mean.flatten(1) @ kd.K.T - S).abs().max() / S.abs().max()))
    n = len(val0) * X[0].numel()
    res["recon"] = dict(rmse=(se / n) ** 0.5, mean_sigma=sig / n, nll_per_cell=nll / n, coverage90=cov / n,
                        consistency=cons, obs_noise=d["obs_noise"])
    snap_t = int(val0[len(val0) // 2])
    snap_mean, snap_lv = kd(z[snap_t:snap_t + 1])
    print(f"[kd] seed {args.seed} reconstruction from true z: RMSE {res['recon']['rmse']:.4f} (obs noise "
          f"{d['obs_noise']}), mean sigma {res['recon']['mean_sigma']:.4f}, 90 % coverage "
          f"{100 * res['recon']['coverage90']:.1f} %, NLL/cell {res['recon']['nll_per_cell']:.3f}, |K mu - S|/|S| {cons:.1e}")

    # --- 2. forecasts under the three priors -----------------------------------------------------------------------------
    lines = []
    for a in args.leads:
        val = np.arange(8000 - a, T - a)
        launches = torch.as_tensor(val, device=dev)
        for prior in PRIORS:
            run = os.path.join(FD, flow_run(args.seed, a, prior))
            pd_ = PriorDraws(prior, run, args.hf_dir, args.seed, val, dev)
            pr = np.load(os.path.join(run, "prior.npz"))
            assert np.allclose(pr["mu"], ck["mu"].cpu().numpy(), rtol=1e-4) and np.allclose(pr["sd"], ck["sd"].cpu().numpy(), rtol=1e-4)
            sse = crps = var = 0.0
            ranks = torch.zeros(M + 1, device=dev)
            per = np.zeros((len(val), 2))
            lq = []
            for i0 in range(0, len(val), args.chunk):
                t = launches[i0:i0 + args.chunk]
                B = len(t)
                zs = pd_.sample(t, M)
                x = kd.sample(zs.reshape(B * M, -1)).view(B, M, *X.shape[1:])
                y = X[t + a]
                mean = x.mean(1)
                sse += float(((mean - y) ** 2).sum())
                crps += float(fair_crps(x.transpose(0, 1), y).sum())
                v = x.var(1)
                var += float(v.sum())
                per[i0:i0 + B, 0] = ((mean - y) ** 2).flatten(1).mean(1).sqrt().cpu().numpy()
                per[i0:i0 + B, 1] = v.flatten(1).mean(1).sqrt().cpu().numpy()
                ranks += torch.bincount((x < y[:, None]).sum(1).flatten(), minlength=M + 1).float()
                lq.append(pd_.logpdf(t, z[t + a]))
            ncell = len(val) * X[0].numel()
            rmse = (sse / ncell) ** 0.5
            spread = (var / ncell * (M + 1) / M) ** 0.5
            lq = torch.cat(lq)
            r = dict(rmse=rmse, crps=crps / ncell, spread_over_rmse=spread / rmse,
                     spread_error_corr=float(np.corrcoef(per[:, 0], per[:, 1])[0, 1]),
                     rank_hist=(ranks / ranks.sum()).cpu().tolist(),
                     latent_log_score=float(lq.mean()), latent_log_score_median=float(lq.median()))
            fm = os.path.join(EFD, flow_run(args.seed, a, prior), "metrics.json")
            if os.path.exists(fm):
                f = json.load(open(fm))
                r["flow"] = dict(rmse=f["rmse"]["flow"], crps=f["flow"]["crps"],
                                 spread_over_rmse=f["flow"]["spread_over_rmse"],
                                 spread_error_corr=f["flow"]["spread_error_corr"])
            res[f"lead{a}_{prior}"] = r
            fl = r.get("flow", {})
            lines.append(f"lead {a:2d} {prior:8s}: kernel RMSE {rmse:.4f} CRPS {r['crps']:.4f} s/R "
                         f"{r['spread_over_rmse']:.2f} corr {r['spread_error_corr']:+.2f} | flow RMSE "
                         f"{fl.get('rmse', np.nan):.4f} CRPS {fl.get('crps', np.nan):.4f} s/R "
                         f"{fl.get('spread_over_rmse', np.nan):.2f} | latent log score {r['latent_log_score']:.2f}")
            print(lines[-1], flush=True)
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "summary.txt"), "w").write(
        f"seed {args.seed} reconstruction: {json.dumps(res['recon'])}\n" + "\n".join(lines) + "\n")
    figures(res, X[snap_t].cpu(), snap_mean[0].cpu(), torch.exp(0.5 * snap_lv[0]).cpu(), args.leads, args.out, M)


def figures(res, truth, mean, sd, leads, out, M):
    # --- reconstruction snapshot ---
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for v, name in enumerate(("SSH", "SST")):
        lim = float(truth[v].abs().max())
        for c, (img, title, cm, lo, hi) in enumerate((
                (truth[v], "truth x(t)", "RdBu_r", -lim, lim), (mean[v], "decoded mean μ(z true)", "RdBu_r", -lim, lim),
                (sd[v], "decoded σ(z true)", "Blues", 0, None), ((truth[v] - mean[v]).abs(), "|truth − μ|", "Blues", 0, None))):
            im = ax[v, c].imshow(img.numpy(), cmap=cm, vmin=lo, vmax=hi, origin="lower")
            ax[v, c].set_xticks([])
            ax[v, c].set_yticks([])
            ax[v, c].set_title(f"{name}: {title}", fontsize=9)
            fig.colorbar(im, ax=ax[v, c], fraction=0.046)
    r = res["recon"]
    fig.suptitle(f"kernel decoder, seed {res['seed']}, reconstruction from the true latent (validation): RMSE "
                 f"{r['rmse']:.4f}, mean σ {r['mean_sigma']:.4f}, 90 % coverage {100 * r['coverage90']:.0f} %", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recon.png"), dpi=110)
    plt.close(fig)
    # --- forecast scores: kernel vs flow, per prior and lead ---
    fig, ax = plt.subplots(1, 2 * len(leads) + 1, figsize=(5 * (2 * len(leads) + 1), 4))
    x = np.arange(len(PRIORS))
    k = 0
    for a in leads:
        for key, lab in (("rmse", "RMSE of member mean"), ("crps", "CRPS")):
            kv = [res[f"lead{a}_{p}"][key] for p in PRIORS]
            fv = [res[f"lead{a}_{p}"].get("flow", {}).get(key, np.nan) for p in PRIORS]
            ax[k].bar(x - 0.2, kv, 0.4, color=[COL[p] for p in PRIORS], label="kernel decoder")
            ax[k].bar(x + 0.2, fv, 0.4, color=[COL[p] for p in PRIORS], alpha=0.45, hatch="//", label="flow decoder")
            ax[k].set_xticks(x, PRIORS)
            ax[k].set_title(f"lead {a}: {lab}", fontsize=10)
            lo = 0.9 * np.nanmin(kv + fv)
            ax[k].set_ylim(lo, 1.05 * np.nanmax(kv + fv))
            ax[k].spines[["top", "right"]].set_visible(False)
            k += 1
    ax[0].legend(fontsize=8, frameon=False)
    for a, ls in zip(leads, ("-", "--")):
        for p in PRIORS:
            h = np.array(res[f"lead{a}_{p}"]["rank_hist"]) * (M + 1)
            ax[k].plot(h, color=COL[p], ls=ls, label=f"lead {a} {p}")
    ax[k].axhline(1, color="#8c8b86", lw=1, ls=":")
    ax[k].set_title("kernel decoder rank histograms (1 = flat)", fontsize=10)
    ax[k].legend(fontsize=7, frameon=False)
    ax[k].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scores.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
