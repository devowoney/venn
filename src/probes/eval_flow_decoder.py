"""Readout of the flow-matching decoder (D-048, SOP 05 "Flow-matching decoder"), lead a, VALIDATION launches only.

Every member m of a launch t draws its own latent state z~_m ~ N(zhat(t), Sigma_a) (the twin's forecast + the
out-of-fold Gaussian error) and is decoded two ways:
  flow     : the flow-matching decoder, its source sample (pure noise, or x(t) + noise) carried to a field that fits z~_m
  linear   : the fair baseline, ridge decoder D(z~_m) + iid per-cell Gaussian residual (training variance)
References (one field per launch): D(zhat) (deterministic chain), D(z_true) (decoding ceiling), persistence x(t), and
the flow conditioned on the TRUE latent (its own decoding ceiling, `--true_members` members).

Scores against the OBSERVED field x(t+a), field units (each variable ~unit std), pooled over cells and launches:
  RMSE of the member mean (headline); fair CRPS per cell; spread / RMSE (finite-M corrected; ~1 = calibrated);
  corr over launches of (launch spread, launch error); rank histogram of the truth among the members;
  energy spectra: fields (members vs truth) and error (truth - mean) vs spread (member - mean).
Run:  python -m src.probes.eval_flow_decoder --run .tmps/runs_fd/fd_seed0 --out .tmps/eval_fd/fd_seed0
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
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402

from src.models.flow_decoder import FlowDecoder            # noqa: E402
from src.models.frontal_decoder import FrontalDecoder      # noqa: E402
from src.train.fit_frontal_decoder import load_eye         # noqa: E402

COL = dict(flow="#2a78d6", linear="#eb6834", truth="#1b1b1a", ceiling="#8c8b86", persistence="#1baf7a")
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#eb6834", "#a8401b"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def fair_crps(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair CRPS per cell: members x [M, ...], truth y [...]. Uses the sorted-members identity for sum |x_i - x_j|."""
    M = x.shape[0]
    xs = x.sort(0).values
    w = (2 * torch.arange(M, device=x.device, dtype=x.dtype) - M + 1).view(-1, *[1] * (x.dim() - 1))
    pair = (w * xs).sum(0)                                     # sum_{i<j} |x_i - x_j|
    return (x - y).abs().mean(0) - pair / (M * (M - 1))


def radial_spectrum(f: torch.Tensor) -> torch.Tensor:
    """Isotropic power spectrum of fields [..., H, W] -> [..., kmax] (mean power per integer wavenumber shell)."""
    H, W = f.shape[-2:]
    P = torch.fft.fft2(f).abs() ** 2 / (H * W)
    ky, kx = torch.meshgrid(torch.fft.fftfreq(H, 1 / H), torch.fft.fftfreq(W, 1 / W), indexing="ij")
    k = torch.sqrt(kx ** 2 + ky ** 2).round().long().to(f.device)
    kmax = min(H, W) // 2
    out = torch.zeros(*f.shape[:-2], kmax, device=f.device)
    for i in range(1, kmax + 1):
        m = k == i
        out[..., i - 1] = P[..., m].mean(-1)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--true_members", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16, help="launches per batch")
    ap.add_argument("--ode_steps", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--replot", action="store_true", help="only redraw the figures of a finished readout")
    ap.add_argument("--dramatic", action="store_true", help="members from the exaggerated family prior (D-054)")
    ap.add_argument("--hf_dir", default="/home/sysadmin/jlee/venn/.claude/worktrees/history-forecastor/.tmps/runs_hf")
    args = ap.parse_args()
    if args.replot:
        return replot(args.out)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)

    pr = dict(np.load(os.path.join(args.run, "prior.npz"), allow_pickle=True))
    ck = torch.load(os.path.join(args.run, "model.pt"), map_location=dev, weights_only=False)
    flow = FlowDecoder(ck["K"], V=ck["V"], ch=tuple(ck["ch"]), emb=ck["emb"], sigma0=ck["sigma0"],
                       source=ck.get("source", "launch"),        # D-048 runs predate the switch
                       zmap=ck.get("zmap", 0)).to(dev).eval()
    flow.load_state_dict(ck["state"])
    dk = torch.load(os.path.join(args.run, "decoder.pt"), map_location=dev, weights_only=False)
    lin = FrontalDecoder(dk["K"], dk["lags"], dk["field_shape"], dk["ridge"]).to(dev)
    lin.load_state_dict(dk["state"])
    res_sd = dk["res_var"].to(dev).sqrt()
    S_all, Y, shape, obs_noise, _ = load_eye(ck["encoder_run"], dev)
    X = Y.float().reshape(len(Y), *shape)
    a, t_tr, T = int(pr["lead"]), int(pr["t_tr"]), len(Y)
    mu, sd = (torch.tensor(pr[k], dtype=torch.float32, device=dev) for k in ("mu", "sd"))
    z = torch.tensor(pr["z"], dtype=torch.float32, device=dev)
    zhat = torch.tensor(pr["zhat"], dtype=torch.float32, device=dev)
    L = torch.tensor(pr["chol"], dtype=torch.float32, device=dev)
    npath = os.path.join(args.run, "netprior.npz")
    netp = None
    if os.path.exists(npath):                                  # D-053: the run was trained on the network's p(z)
        q = np.load(npath)
        netp = {k: torch.tensor(q[k], dtype=torch.float32, device=dev) for k in ("m", "w", "chol_R", "sigma")}
    K, M = z.shape[1], args.members
    launches = torch.arange(t_tr - a, T - a, device=dev)          # every target in the validation set
    n = len(launches)
    dram = None
    if args.dramatic:                                          # D-054: exaggerated family shapes on the network centre
        from src.models.family_net import DramaticSampler
        dram = DramaticSampler(args.run, args.hf_dir, int(ck["cfg"]["obs_seed"]), launches.cpu().numpy(), dev)
    print(f"[eval] {args.run} source={flow.source} cond={ck['cfg']['cond']} lead {a}: {n} validation launches, M={M}")

    def dec_lin(zz):                                           # canonical z [..., K] -> field [..., V, H, W]
        return lin(zz * sd + mu)

    methods = ("flow", "linear")
    sse = {m: 0.0 for m in methods + ("det", "ceiling", "persistence", "flow_true")}
    crps = {m: 0.0 for m in methods}
    var = {m: 0.0 for m in methods}
    per_launch = {m: np.zeros((n, 2)) for m in methods}        # (RMSE of mean, spread) per launch
    ranks = {m: torch.zeros(M + 1, device=dev) for m in methods}
    spec = {k: 0.0 for k in ("truth", "flow_member", "linear_member", "flow_err", "flow_spread",
                             "linear_err", "linear_spread")}
    keep = {}
    snap_t = int(launches[n // 2])
    ncell = X[0].numel()
    for i0 in range(0, n, args.chunk):
        t = launches[i0:i0 + args.chunk]
        B = len(t)
        y = X[t + a]                                           # [B,V,H,W]
        if dram is not None:
            zs = dram.sample(t, M)
        elif netp is None:
            zs = zhat[t][:, None] + torch.randn(B, M, K, device=dev) @ L.T     # [B,M,K] latent members
        else:
            eta = torch.randn(B, M, K, device=dev) @ netp["chol_R"].T
            zs = zhat[t][:, None] + netp["sigma"] * (netp["m"][t][:, None] + netp["w"][t][:, None] * eta)
        xl = X[t][:, None].expand(B, M, *shape).reshape(B * M, *shape)
        mem = {"flow": flow.sample(xl, zs.reshape(B * M, K), steps=args.ode_steps).view(B, M, *shape),
               "linear": dec_lin(zs) + res_sd.view(*shape) * torch.randn(B, M, *shape, device=dev)}
        for m in methods:
            x = mem[m]
            mean = x.mean(1)
            sse[m] += float(((mean - y) ** 2).sum())
            crps[m] += float(fair_crps(x.transpose(0, 1), y).sum())
            v = x.var(1)                                       # unbiased member variance per cell
            var[m] += float(v.sum())
            per_launch[m][i0:i0 + B, 0] = ((mean - y) ** 2).flatten(1).mean(1).sqrt().cpu().numpy()
            per_launch[m][i0:i0 + B, 1] = v.flatten(1).mean(1).sqrt().cpu().numpy()
            ranks[m] += torch.bincount((x < y[:, None]).sum(1).flatten(), minlength=M + 1).float()
            spec[f"{m}_member"] += radial_spectrum(x[:, 0]).sum((0, 1))   # launches, variables
            spec[f"{m}_err"] += radial_spectrum(y - mean).sum((0, 1))
            spec[f"{m}_spread"] += radial_spectrum(x - mean[:, None]).mean(1).sum((0, 1))
        spec["truth"] += radial_spectrum(y).sum((0, 1))
        sse["det"] += float(((dec_lin(zhat[t]) - y) ** 2).sum())
        sse["ceiling"] += float(((dec_lin(z[t + a]) - y) ** 2).sum())
        sse["persistence"] += float(((X[t] - y) ** 2).sum())
        mt = args.true_members
        xt = flow.sample(X[t][:, None].expand(B, mt, *shape).reshape(B * mt, *shape),
                         z[t + a][:, None].expand(B, mt, K).reshape(B * mt, K), steps=args.ode_steps)
        sse["flow_true"] += float(((xt.view(B, mt, *shape).mean(1) - y) ** 2).sum())
        if snap_t in t.tolist():
            j = t.tolist().index(snap_t)
            keep = dict(truth=y[j].cpu(), launch=X[snap_t].cpu(), det=dec_lin(zhat[snap_t]).cpu(),
                        **{m: mem[m][j].cpu() for m in methods})
        if i0 % (args.chunk * 25) == 0:
            print(f"[eval] {i0 + B}/{n}", flush=True)

    tot = n * ncell
    res = dict(lead=a, launches=n, members=M, source=flow.source, prior="dramatic" if dram is not None else ("net" if netp is not None else "gauss"), cond=ck["cfg"]["cond"], obs_noise=obs_noise,
               rmse={k: (v / tot) ** 0.5 for k, v in sse.items()})
    for m in methods:
        rmse = res["rmse"][m]
        spread = (var[m] / tot * (M + 1) / M) ** 0.5           # finite-M corrected
        r = np.corrcoef(per_launch[m][:, 0], per_launch[m][:, 1])[0, 1]
        res[m] = dict(crps=crps[m] / tot, spread=spread, spread_over_rmse=spread / rmse, spread_error_corr=float(r),
                      rank_hist=(ranks[m] / ranks[m].sum()).cpu().tolist())
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    np.savez(os.path.join(args.out, "per_launch.npz"), **{m: per_launch[m] for m in methods},
             **{f"spec_{k}": (v / n).cpu().numpy() for k, v in spec.items()})
    lines = [f"lead {a}, {n} validation launches, M={M}, source={res['source']}, cond={res['cond']}, field units (obs noise {obs_noise})",
             "RMSE vs observed field: " + "  ".join(f"{k} {v:.4f}" for k, v in res["rmse"].items())]
    for m in methods:
        r = res[m]
        lines.append(f"{m:7s}: CRPS {r['crps']:.4f}  spread {r['spread']:.4f}  spread/RMSE {r['spread_over_rmse']:.3f}"
                     f"  corr(spread, error) over launches {r['spread_error_corr']:+.3f}")
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    np.savez(os.path.join(args.out, "snapshot.npz"), **{k: v.numpy() for k, v in keep.items()})
    figures(res, per_launch, {k: (v / n).cpu().numpy() for k, v in spec.items()}, keep, args.out, M)


def replot(out: str):
    """Redraw the figures from a finished readout (metrics.json, per_launch.npz, snapshot.npz) without resampling."""
    res = json.load(open(os.path.join(out, "metrics.json")))
    pl = np.load(os.path.join(out, "per_launch.npz"))
    keep = {k: torch.tensor(v) for k, v in np.load(os.path.join(out, "snapshot.npz")).items()}
    figures(res, {m: pl[m] for m in ("flow", "linear")},
            {k[5:]: pl[k] for k in pl.files if k.startswith("spec_")}, keep, out, res["members"])


def figures(res, per_launch, spec, keep, out, M):
    # --- 1. scores: RMSE bars, rank histograms, spread vs error per launch, spectra --------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    names = [("persistence", "persistence x(t)"), ("det", "linear D(ẑ)"), ("linear", "linear-Gauss mean"),
             ("flow", "flow mean"), ("ceiling", "linear D(z true)"), ("flow_true", "flow | z true")]
    cols = [COL["persistence"], COL["ceiling"], COL["linear"], COL["flow"], COL["ceiling"], COL["flow"]]
    vals = [res["rmse"][k] for k, _ in names]
    ax[0].barh(range(len(names)), vals, color=cols, height=0.6)
    for i, v in enumerate(vals):
        ax[0].text(v, i, f" {v:.4f}", va="center", fontsize=8, color="#3b3b39")
    ax[0].axvline(res["obs_noise"], color="#8c8b86", lw=1, ls="--")
    ax[0].text(res["obs_noise"], len(names) - 0.4, " obs noise", fontsize=7, color="#6b6a66")
    ax[0].set_yticks(range(len(names)), [l for _, l in names], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel(f"field RMSE vs observed x(t+{res['lead']})")
    ax[0].set_title(f"lead {res['lead']}: accuracy", fontsize=10)
    xs = np.arange(M + 1)
    for m, off in (("flow", -0.2), ("linear", 0.2)):
        ax[1].bar(xs + off, np.array(res[m]["rank_hist"]) * (M + 1), width=0.4, color=COL[m],
                  label=f"{m}  spread/RMSE {res[m]['spread_over_rmse']:.2f}")
    ax[1].axhline(1, color="#8c8b86", lw=1, ls="--")
    ax[1].set_xlabel("rank of the truth among the members")
    ax[1].set_ylabel("frequency x (M+1)  (1 = flat)")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("calibration (all cells)", fontsize=10)
    for m in ("flow", "linear"):
        pl = per_launch[m]
        ax[2].scatter(pl[:, 1], pl[:, 0], s=5, alpha=0.4, color=COL[m],
                      label=f"{m}  corr {res[m]['spread_error_corr']:+.2f}")
    lim = max(per_launch["flow"].max(), per_launch["linear"].max())
    ax[2].plot([0, lim], [0, lim], color="#8c8b86", lw=1, ls="--")
    ax[2].set_xlabel("launch spread (RMS member std)")
    ax[2].set_ylabel("launch RMSE of the mean")
    ax[2].legend(fontsize=8, frameon=False, markerscale=3)
    ax[2].set_title("does the spread follow the error?", fontsize=10)
    k = np.arange(1, len(spec["truth"]) + 1)                       # wavenumber shells 1 .. 32
    ax[3].loglog(k, spec["truth"], color=COL["truth"], lw=2, label="truth field")
    for m in ("flow", "linear"):
        ax[3].loglog(k, spec[f"{m}_member"], color=COL[m], lw=2, label=f"{m} member field")
        ax[3].loglog(k, spec[f"{m}_err"], color=COL[m], lw=1.2, ls="--", label=f"{m} error (truth − mean)")
        ax[3].loglog(k, spec[f"{m}_spread"], color=COL[m], lw=1.2, ls=":", label=f"{m} spread (member − mean)")
    ax[3].set_xlabel("wavenumber (cycles / domain)")
    ax[3].set_ylabel("power")
    ax[3].legend(fontsize=7, frameon=False)
    ax[3].set_title("energy spectra (both variables summed)", fontsize=10)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "scores.png"), dpi=130)
    plt.close(fig)

    # --- 2. one validation launch: truth / two members / mean / spread / error, per decoder and variable ------------
    if not keep:
        return
    cols = [f"truth x(t+{res['lead']})", "member 1", "member 2", "member mean", "member spread", "|truth − mean|"]
    fig, ax = plt.subplots(4, 6, figsize=(17, 11))
    for v, vname in enumerate(("SSH", "SST")):
        lim = float(keep["truth"][v].abs().max())
        # one colour scale for spread and error across BOTH decoders, so their sizes compare by eye
        emax = max(float(max(keep[m][:, v].std(0).max(), (keep["truth"][v] - keep[m][:, v].mean(0)).abs().max()))
                   for m in ("flow", "linear"))
        for r, m in enumerate(("flow", "linear")):
            row = 2 * v + r
            x = keep[m][:, v]
            mean, spr = x.mean(0), x.std(0)
            err = (keep["truth"][v] - mean).abs()
            panels = [(keep["truth"][v], DIV, -lim, lim), (x[0], DIV, -lim, lim), (x[1], DIV, -lim, lim),
                      (mean, DIV, -lim, lim), (spr, SEQ, 0, emax), (err, SEQ, 0, emax)]
            for c, (img, cm, lo, hi) in enumerate(panels):
                im = ax[row, c].imshow(img.numpy(), cmap=cm, vmin=lo, vmax=hi, origin="lower")
                ax[row, c].set_xticks([])
                ax[row, c].set_yticks([])
                if row == 0:
                    ax[row, c].set_title(cols[c], fontsize=10)
                if c in (3, 5):
                    fig.colorbar(im, ax=ax[row, c], fraction=0.046)
            ax[row, 0].set_ylabel(f"{vname} — {m}", fontsize=10)
    fig.suptitle(f"lead {res['lead']}, one validation launch, {M} members "
                 f"(source={res.get('source', 'launch')}, cond={res['cond']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "snapshot.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
