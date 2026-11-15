"""Step A of D-051: the family-shaped latent prior ALONE, scored on the true z(t+a) (no decoder yet).

Fits `FamilyPrior` on the out-of-fold departures of the training launches (the same honest errors as Sigma_1) and reads
it on the VALIDATION launches (the twin's forecast), against the plain Gaussian N(zhat, diag Sigma) per channel:
  PIT histograms per family (flat = calibrated shape), fair CRPS per family (marginal, canonical z units),
  the cyclic error shape along the direction of travel split by the reach (near crest vs mid-cycle), the chaotic error
  shape vs the uniform window, one cyclic channel's 5-95 % band in time; the copula's normal-score correlation.
Run:  python -m src.probes.eval_family_prior --run .tmps/runs_fd/fdzmap_seed0 --out .tmps/eval_fp/seed0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from scipy.special import ndtr                             # noqa: E402

from src.models.family_prior import ETA, FamilyPrior, ramp_logpdf, sample_from_tables   # noqa: E402
from src.probes.eval_flow_decoder import fair_crps         # noqa: E402

FAMS = ("stationary", "cyclic", "chaotic", "noise")
COL = dict(family="#2a78d6", gauss="#eb6834", truth="#1b1b1a", grey="#8c8b86")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a flow-decoder run dir holding prior.npz")
    ap.add_argument("--hf_dir", default="/home/sysadmin/jlee/venn/.claude/worktrees/history-forecastor/.tmps/runs_hf")
    ap.add_argument("--obs_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    pr = dict(np.load(os.path.join(args.run, "prior.npz"), allow_pickle=True))
    nm = np.load(os.path.join(args.hf_dir, f"twin_seed{args.obs_seed}", "norm.npz"), allow_pickle=True)
    assert (nm["channels"] == pr["channels"]).all()
    fam = nm["fam"]
    z, zhat, a, t_tr = pr["z"], pr["zhat"], int(pr["lead"]), int(pr["t_tr"])
    T, K = z.shape
    fit_t = np.arange(64, t_tr - a)
    fp = FamilyPrior().fit(z, zhat, pr["Sigma"], fam, a, fit_t)
    pickle.dump(fp, open(os.path.join(args.out, "family_prior.pkl"), "wb"))
    val = np.arange(t_tr - a, T - a)
    print(f"[fp] lead {a}: fit on {np.sum(fit_t >= fp.t0)} out-of-fold launches (t >= {fp.t0}), "
          f"{len(val)} validation launches; families " + ", ".join(f"{f} {np.sum(fam == f)}" for f in FAMS))

    # --- marginal scores on validation ----------------------------------------------------------------------------
    ztrue, zh = z[val + a], zhat[val]
    pit_f = fp.pit(ztrue, zh, val)
    pit_g = ndtr((ztrue - zh) / fp.sigma)
    tab = torch.tensor(fp.tables(val))
    gtab = torch.tensor((fp.sigma[:, None] * ETA[None])[None].repeat(len(val), 0), dtype=torch.float32)
    I = torch.eye(K)
    M = args.members
    eta = torch.randn(len(val), M, K)
    zt, zht = torch.tensor(ztrue, dtype=torch.float32), torch.tensor(zh, dtype=torch.float32)
    crps_f = fair_crps(sample_from_tables(zht, tab, I, M, eta).transpose(0, 1), zt).mean(0).numpy()   # per channel
    crps_g = fair_crps(sample_from_tables(zht, gtab, I, M, eta).transpose(0, 1), zt).mean(0).numpy()
    res = dict(lead=a, launches=len(val), members=M, periods={int(i): fp.period[i] for i in fp.cyc},
               cyclic_params={int(i): dict(sb_over_sigma=fp.par[i][0] / fp.sigma[i],
                                           st_over_sigma=fp.par[i][1] / fp.sigma[i], kappa=fp.par[i][2],
                                           mean_reach_over_sigma=float(np.nanmean(fp.reach(i, val)) / fp.sigma[i]))
                              for i in fp.cyc},
               per_family={})
    lines = [f"lead {a}, {len(val)} validation launches, canonical z units, marginal CRPS (M={M})"]
    for f in FAMS:
        m = fam == f
        if not m.any():
            continue
        r = dict(n=int(m.sum()), crps_family=float(crps_f[m].mean()), crps_gauss=float(crps_g[m].mean()),
                 pit_family=np.histogram(pit_f[:, m], bins=10, range=(0, 1))[0].tolist(),
                 pit_gauss=np.histogram(pit_g[:, m], bins=10, range=(0, 1))[0].tolist())
        res["per_family"][f] = r
        lines.append(f"{f:10s} ({r['n']:2d} ch): CRPS family {r['crps_family']:.4f}  gauss {r['crps_gauss']:.4f}  "
                     f"({100 * (r['crps_family'] / r['crps_gauss'] - 1):+.1f} %)")
    lines.append(f"all        ({K} ch): CRPS family {crps_f.mean():.4f}  gauss {crps_g.mean():.4f}  "
                 f"({100 * (crps_f.mean() / crps_g.mean() - 1):+.1f} %)")
    lines.append("cyclic channels: ch period sb/sig st/sig kappa mean(r)/sig  CRPS family/gauss")
    for i in fp.cyc:
        p = res["cyclic_params"][int(i)]
        lines.append(f"  {int(pr['channels'][i]):3d} {fp.period[i]:4d}  {p['sb_over_sigma']:.2f}  {p['st_over_sigma']:.2f}"
                     f"  {p['kappa']:.3f}  {p['mean_reach_over_sigma']:.2f}   {crps_f[i]:.4f}/{crps_g[i]:.4f}")
    off = np.abs(fp.R - np.eye(K))
    lines.append(f"copula R: mean |off-diagonal| {off.sum() / (K * K - K):.3f}, max {off.max():.3f}")
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(fp, fam, z, zhat, a, val, pit_f, pit_g, pr["channels"], args.out)


def figures(fp, fam, z, zhat, a, val, pit_f, pit_g, chan, out):
    # --- 1. PIT per family ------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    bins = np.linspace(0, 1, 11)
    for k, f in enumerate(FAMS):
        m = fam == f
        for pit, c, lab, off in ((pit_g, COL["gauss"], "Gaussian", 0.022), (pit_f, COL["family"], "family", -0.022)):
            h = np.histogram(pit[:, m], bins=bins)[0] / pit[:, m].size * 10
            ax[k].bar(bins[:-1] + 0.05 + off, h, width=0.044, color=c, label=lab)
        ax[k].axhline(1, color=COL["grey"], ls="--", lw=1)
        ax[k].set_title(f"{f} ({m.sum()} ch): PIT of the true z(t+{a})", fontsize=10)
        ax[k].set_xlabel("PIT")
        ax[k].legend(fontsize=8, frameon=False)
        ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("density (1 = calibrated)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pit.png"), dpi=120)
    plt.close(fig)

    # --- 2. error shapes: cyclic along the travel direction, split by the distance to the crest; chaotic vs uniform -
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    U, R, SB, ST, DI = [], [], [], [], []
    for i in fp.cyc:
        s = fp.state[i]["s"][val]
        U.append(s * (z[val + a, i] - zhat[val, i]) / fp.sigma[i])
        R.append(fp.reach(i, val) / fp.sigma[i])
        DI.append(fp.state[i]["dist"][val] / fp.state[i]["A"][val])        # 0 = at the crest, 1 = mid-cycle
        SB.append(np.full(len(val), fp.par[i][0] / fp.sigma[i]))
        ST.append(np.full(len(val), fp.par[i][1] / fp.sigma[i]))
    U, R, SB, ST, DI = (np.concatenate(v) for v in (U, R, SB, ST, DI))
    q1, q2 = 0.2, 0.7
    g = np.linspace(-5, 5, 401)
    for k, (sel, name) in enumerate(((DI <= q1, f"near crest/trough (dist ≤ {q1} A)"),
                                     ((DI > q1) & (DI <= q2), "between"),
                                     (DI > q2, f"mid-cycle (dist > {q2} A)"))):
        ax[k].hist(U[sel], bins=np.linspace(-5, 5, 61), density=True, color="#cde2fb", label="truth (validation)")
        # model density averaged over the launches in the bin (each launch has its own reach)
        idx = np.random.default_rng(0).choice(np.flatnonzero(sel), size=min(3000, sel.sum()), replace=False)
        dens = np.mean([np.exp(ramp_logpdf(g, np.full_like(g, R[j]), SB[j], ST[j])) for j in idx], 0)
        ax[k].plot(g, dens, color=COL["family"], lw=2, label="ramp + tail (fitted)")
        ax[k].plot(g, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), color=COL["gauss"], lw=1.5, ls="--", label="Gaussian")
        ax[k].axvline(0, color=COL["grey"], lw=0.8)
        ax[k].set_title(f"cyclic, {name}, n={sel.sum()}", fontsize=9)
        ax[k].set_xlabel("s (z - ẑ) / σ   (+ = direction of travel)")
        ax[k].legend(fontsize=7, frameon=False)
    ch = fam == "chaotic"
    D = ((z[val + a] - zhat[val]) / fp.sigma)[:, ch].ravel()
    ax[3].hist(D, bins=np.linspace(-5, 5, 61), density=True, color="#cde2fb", label="truth (validation)")
    h = 1 / (2 * np.sqrt(3))
    ax[3].plot([-5, -np.sqrt(3), -np.sqrt(3), np.sqrt(3), np.sqrt(3), 5], [0, 0, h, h, 0, 0],
               color=COL["family"], lw=2, label="uniform ẑ ± √3σ")
    ax[3].plot(g, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), color=COL["gauss"], lw=1.5, ls="--", label="Gaussian")
    ax[3].set_title(f"chaotic ({ch.sum()} ch)", fontsize=9)
    ax[3].set_xlabel("(z - ẑ) / σ")
    ax[3].legend(fontsize=7, frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "shapes.png"), dpi=120)
    plt.close(fig)

    # --- 3. one cyclic channel in time: truth, forecast, 5-95 % band of both priors ----------------------------------
    i = max(fp.cyc, key=lambda c: fp.period[c])                    # a slow cycle shows the ramp best
    w = val[:3 * fp.period[i]]
    tab = fp.tables(w)[:, i]
    lv = ndtr(ETA)
    lo, hi = (np.array([np.interp(p, lv, tab[j]) for j in range(len(w))]) for p in (0.05, 0.95))
    fig, ax = plt.subplots(figsize=(15, 3.6))
    tt = w + a
    ax.fill_between(tt, zhat[w, i] + lo, zhat[w, i] + hi, color=COL["family"], alpha=0.3, lw=0, label="family 5-95 %")
    ax.plot(tt, zhat[w, i] - 1.645 * fp.sigma[i], color=COL["gauss"], lw=0.8, ls="--", label="Gaussian 5-95 %")
    ax.plot(tt, zhat[w, i] + 1.645 * fp.sigma[i], color=COL["gauss"], lw=0.8, ls="--")
    ax.plot(tt, z[tt, i], color=COL["truth"], lw=1, label="true z")
    ax.plot(tt, zhat[w, i], color=COL["family"], lw=1, label="forecast ẑ")
    ax.set_title(f"cyclic channel {int(chan[i])} (period {fp.period[i]}), lead {a}, validation", fontsize=10)
    ax.set_xlabel("time step")
    ax.legend(fontsize=8, frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cyclic_band.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
