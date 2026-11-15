"""Train and read the family-shape distribution network (D-052) at lead 1 -- the latent distribution ALONE, no decoder.

Data: prior.npz of a flow-decoder run (true z, out-of-fold zhat on [0, t_tr), the twin's zhat on validation, Sigma_1)
and the observer families (`fam` in module 2's norm.npz). Launches: fit [t0, 7750-a), model selection [7750-a, 8000-a)
(CRPS, best checkpoint kept), validation [8000-a, T-a) only scored.
Two networks, same inputs: `family` (Gaussian / ramp + tail / uniform by family) and `gauss` (Gaussian for every family:
keeps the per-launch adaptation, drops the family shape). Reference: the fixed Gaussian N(zhat, sigma_i^2) of D-048.
Scores per family on the true z(t+1), canonical z units: CRPS, PIT, corr(predicted spread, |error|) over launches.
Run:  python -m src.probes.train_family_net --run .tmps/runs_fd/fdzmap_seed0 --out .tmps/eval_fn/seed0
"""
from __future__ import annotations

import argparse
import copy
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

from src.models.family_net import GRID, SHAPES, FamilyNet, cdf_on_grid, crps_on_grid, launch_features   # noqa: E402

FAMS = ("stationary", "cyclic", "chaotic", "noise")
COL = dict(family="#2a78d6", gauss_net="#1baf7a", fixed="#eb6834", truth="#1b1b1a", grey="#8c8b86")


def to_t(x, dev):
    return torch.tensor(np.nan_to_num(x), dtype=torch.float32, device=dev)


def quantile(cdf: torch.Tensor, p: float) -> torch.Tensor:
    """p-quantile of CDFs on GRID [..., G] -> [...]."""
    g = GRID.to(cdf.device)
    j = torch.searchsorted(cdf.contiguous(), torch.full(cdf.shape[:-1] + (1,), p, device=cdf.device)).clamp(1, len(g) - 1)
    c0, c1 = cdf.gather(-1, j - 1), cdf.gather(-1, j)
    w = ((p - c0) / (c1 - c0).clamp_min(1e-12)).clamp(0, 1)
    return (g[j - 1] + w * (g[j] - g[j - 1]))[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hf_dir", default="/home/sysadmin/jlee/venn/.claude/worktrees/history-forecastor/.tmps/runs_hf")
    ap.add_argument("--obs_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_only", action="store_true", help="reload the saved networks, only score and plot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    dev = args.device

    pr = dict(np.load(os.path.join(args.run, "prior.npz"), allow_pickle=True))
    nm = np.load(os.path.join(args.hf_dir, f"twin_seed{args.obs_seed}", "norm.npz"), allow_pickle=True)
    assert (nm["channels"] == pr["channels"]).all()
    fam = nm["fam"]
    z, zhat, a, t_tr = pr["z"], pr["zhat"], int(pr["lead"]), int(pr["t_tr"])
    T, K = z.shape
    sigma = np.sqrt(np.diag(pr["Sigma"]))
    common, feats, s, t0, periods = launch_features(z, zhat, sigma, fam, a, 7750)
    fit_t = np.arange(t0, 7750 - a)
    sel_t = np.arange(7750 - a, t_tr - a)
    val_t = np.arange(t_tr - a, T - a)
    y = np.full((T, K), np.nan)
    y[:T - a] = (z[a:] - zhat[:T - a]) / sigma                      # scaled error of each launch's forecast
    C, S, Y = to_t(common, dev), to_t(s, dev), to_t(np.clip(y, -7.9, 7.9), dev)
    Fe = {f: to_t(v, dev) for f, v in feats.items()}
    print(f"[fn] lead {a}: fit {len(fit_t)} launches (t >= {t0}), select {len(sel_t)}, validation {len(val_t)}")

    def batch(t):
        t = torch.as_tensor(t, device=dev)
        return C[t], {f: v[t] for f, v in Fe.items()}, S[t], Y[t]

    def score(model, t) -> torch.Tensor:
        with torch.no_grad():
            out = []
            for i in range(0, len(t), 500):
                c, fe, ss, yy = batch(t[i:i + 500])
                out.append(crps_on_grid(model.cdf(c, fe, ss), yy))
            return torch.cat(out)                                    # [n, K] scaled units

    feat_dims = {f: v.shape[-1] for f, v in feats.items()}
    models, hist = {}, {}
    for name, shapes in (("family", None), ("gauss_net", {f: "gauss" for f in SHAPES})):
        model = FamilyNet(fam, feat_dims, shapes).to(dev)
        if args.eval_only:
            ck = torch.load(os.path.join(args.out, f"{name}.pt"), map_location=dev, weights_only=False)
            model.load_state_dict(ck["state"])
            models[name] = model.eval()
            hist[name] = json.load(open(os.path.join(args.out, "metrics.json")))["history"][name]
            continue
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        best, best_state, h = float("inf"), None, []
        for step in range(1, args.steps + 1):
            c, fe, ss, yy = batch(np.random.choice(fit_t, args.batch))
            loss = crps_on_grid(model.cdf(c, fe, ss), yy).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 200 == 0:
                v = float(score(model, sel_t).mean())
                h.append((step, float(loss), v))
                if v < best:
                    best, best_state = v, copy.deepcopy(model.state_dict())
                print(f"[fn] {name:9s} step {step:5d}  train {float(loss):.4f}  select {v:.4f}", flush=True)
        model.load_state_dict(best_state)
        models[name], hist[name] = model.eval(), h
        torch.save(dict(state=best_state, shapes=model.shapes, feat_dims=feat_dims), os.path.join(args.out, f"{name}.pt"))

    # --- validation ------------------------------------------------------------------------------------------------
    sig = torch.tensor(sigma, dtype=torch.float32, device=dev)
    yv = Y[torch.as_tensor(val_t, device=dev)]
    g = GRID.to(dev)
    fixed_cdf = cdf_on_grid(torch.exp(-0.5 * g ** 2)).expand(len(val_t), K, len(g))
    cdfs = {"fixed": fixed_cdf}
    for name, m in models.items():
        with torch.no_grad():
            c, fe, ss, _ = batch(val_t)
            cdfs[name] = m.cdf(c, fe, ss)
    res, lines = dict(lead=a, launches=len(val_t), periods=periods, history=hist, per_family={}), []
    lines.append(f"lead {a}, {len(val_t)} validation launches, CRPS in canonical z units (lower = better)")
    lines.append(f"{'family':10s}  {'fixed Gauss':>11s}  {'net Gauss':>9s}  {'net family':>10s}   "
                 f"corr(spread, |error of median|) net Gauss / net family")
    pit, spread, err = {}, {}, {}
    for name, cd in cdfs.items():
        res[f"crps_{name}"] = (crps_on_grid(cd, yv) * sig).mean(0).cpu().numpy()
        j = torch.searchsorted(g.contiguous(), yv.contiguous()).clamp(0, len(g) - 1)
        pit[name] = cd.gather(-1, j[..., None])[..., 0].cpu().numpy()
        spread[name] = (quantile(cd, 0.84) - quantile(cd, 0.16)).cpu().numpy() / 2
        # error of each distribution's own centre (its median), which the network may move away from zhat
        err[name] = (yv - quantile(cd, 0.5)).abs().cpu().numpy()
    for f in FAMS + ("all",):
        msk = np.ones(K, bool) if f == "all" else (np.asarray(fam) == f)
        r = {k: float(res[f"crps_{k}"][msk].mean()) for k in cdfs}
        cor = {k: float(np.corrcoef(spread[k][:, msk].ravel(), err[k][:, msk].ravel())[0, 1]) for k in models}
        rm = {k: float(np.sqrt((err[k][:, msk] ** 2).mean())) for k in cdfs}
        res["per_family"][f] = dict(n=int(msk.sum()), crps=r, spread_error_corr=cor, rmse_of_median_sigma=rm)
        lines.append(f"{f:10s}  {r['fixed']:11.4f}  {r['gauss_net']:9.4f}  {r['family']:10.4f}   "
                     f"{cor['gauss_net']:+.2f} / {cor['family']:+.2f}   "
                     f"(family vs fixed {100 * (r['family'] / r['fixed'] - 1):+.1f} %, "
                     f"vs net Gauss {100 * (r['family'] / r['gauss_net'] - 1):+.1f} %)   "
                     f"RMSE of centre (σ): zhat {rm['fixed']:.2f}, net {rm['gauss_net']:.2f} / {rm['family']:.2f}")
    for k in cdfs:
        res[f"crps_{k}"] = res[f"crps_{k}"].tolist()
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(cdfs, pit, spread, err, fam, z, zhat, sigma, val_t, a, periods, pr["channels"], args.out)


def figures(cdfs, pit, spread, err, fam, z, zhat, sigma, val_t, a, periods, chan, out):
    fam = np.asarray(fam)
    names = (("fixed", "fixed Gaussian"), ("gauss_net", "net, Gaussian"), ("family", "net, family shape"))
    # --- 1. PIT per family -------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    bins = np.linspace(0, 1, 11)
    for k, f in enumerate(FAMS):
        m = fam == f
        for (key, lab), off in zip(names, (-0.03, 0, 0.03)):
            h = np.histogram(pit[key][:, m], bins=bins)[0] / pit[key][:, m].size * 10
            ax[k].bar(bins[:-1] + 0.05 + off, h, width=0.029, color=COL[key], label=lab)
        ax[k].axhline(1, color=COL["grey"], ls="--", lw=1)
        ax[k].set_title(f"{f} ({m.sum()} ch): PIT of true z(t+{a})", fontsize=10)
        ax[k].legend(fontsize=7, frameon=False)
        ax[k].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pit.png"), dpi=120)
    plt.close(fig)

    # --- 2. one channel per family in time: truth and the 5-95 % band of each distribution ---------------------------
    pick = {}
    for f in ("stationary", "cyclic", "chaotic"):
        idx = np.flatnonzero(fam == f)
        pick[f] = idx[0] if f != "cyclic" else max(idx, key=lambda i: periods[int(i)])
    fig, ax = plt.subplots(3, 1, figsize=(15, 9))
    w = slice(0, 400)
    tt = val_t[w] + a
    for r, (f, i) in enumerate(pick.items()):
        for key, lab in names[::-1]:
            lo = quantile(cdfs[key][w, i], 0.05).cpu().numpy() * sigma[i] + zhat[val_t[w], i]
            hi = quantile(cdfs[key][w, i], 0.95).cpu().numpy() * sigma[i] + zhat[val_t[w], i]
            if key == "fixed":
                ax[r].plot(tt, lo, color=COL[key], lw=0.8, ls="--", label=f"{lab} 5-95 %")
                ax[r].plot(tt, hi, color=COL[key], lw=0.8, ls="--")
            else:
                ax[r].fill_between(tt, lo, hi, color=COL[key], alpha=0.25, lw=0, label=f"{lab} 5-95 %")
        ax[r].plot(tt, z[tt, i], color=COL["truth"], lw=1, label="true z")
        ax[r].plot(tt, zhat[val_t[w], i], color=COL["family"], lw=0.8, label="forecast ẑ")
        ax[r].set_title(f"{f} channel {int(chan[i])}, lead {a}, validation", fontsize=10)
        ax[r].legend(fontsize=7, frameon=False, ncol=5)
        ax[r].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bands.png"), dpi=110)
    plt.close(fig)

    # --- 3. does the predicted spread follow the error? (binned, per family) -----------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    for k, f in enumerate(FAMS):
        m = fam == f
        for key, lab in names[1:]:
            sp, er = spread[key][:, m].ravel(), err[key][:, m].ravel()
            q = np.quantile(sp, np.linspace(0, 1, 11))
            b = np.clip(np.digitize(sp, q[1:-1]), 0, 9)
            xs = [sp[b == j].mean() for j in range(10)]
            ys = [np.sqrt((er[b == j] ** 2).mean()) for j in range(10)]
            ax[k].plot(xs, ys, "o-", color=COL[key], label=lab)
        lim = ax[k].get_xlim()
        ax[k].plot(lim, lim, color=COL["grey"], ls="--", lw=1)
        ax[k].set_xlabel("predicted spread (half 16-84 % width, σ units)")
        ax[k].set_ylabel("RMS error of the median in the bin")
        ax[k].set_title(f"{f}: spread vs error (deciles)", fontsize=10)
        ax[k].legend(fontsize=7, frameon=False)
        ax[k].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "spread_error.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
