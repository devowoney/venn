"""Does the observer COVER the field? Layer 1's own success measure (D-038, 2026-11-02).

Layer 1 is a sensor: its job is to see everything significant, possibly mixed, not to be pure. So the question is not
"does every truth mode have its own eye" but "can the field, and every signal in it, be read back from the K eyes".
Everything below is a linear readout fitted on the training slice [0, t_fit) and scored on validation [t_fit, T):

  field map     per-cell reconstruction R^2 of the field (SSH and SST) from the K channels -> WHERE coverage fails
  per mode      decode R^2 of each truth amplitude from all K channels; in which variable the mode is visible
                (`only`: ssh / sst / both); best mask coverage of its patch in SSH and in SST
  per patch     field R^2 inside each truth mode's patch (cells where |phi| > 30 % of its max)

Reading note: a mode visible ONLY in SST exists in the field as an AR1 low-passed, time-lagged copy, so its raw
amplitude cannot be decoded at the same instant: for a cycle of period P the ceiling is cos^2(atan(2 pi tau / P))
(tau = `data.sst_tau`). A low decode there is not a loss; the patch R^2 is the real test.

Run:  conda run -n oceanai python -m src.probes.coverage --runs .tmps/runs/obs_t8000_l1_seed0 ...
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import GenConfig, generate_field  # noqa: E402
from src.probes.eye_compare import FAM_COL  # noqa: E402

PATCH_FRAC = 0.3          # a mode's patch = cells where |phi| > 30 % of its max


def coverage(run: str) -> dict:
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
    S, M = art["S"].astype(np.float64), art["masks"].astype(np.float64)
    t_fit, T = int(art["t_fit"]), S.shape[0]
    tr, va = slice(0, t_fit), slice(t_fit, T)
    field, truth = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)), seed=int(cfg.seed))
    V, H, W = field.shape[1:]
    S1 = np.hstack([np.ones((T, 1)), S])

    # field map: per-cell R^2, both variables
    X = field.reshape(T, -1).astype(np.float64)
    B = np.linalg.lstsq(S1[tr], X[tr], rcond=None)[0]
    res = ((S1[va] @ B - X[va]) ** 2).mean(0)
    xv = X[va].var(0)
    r2map = (1 - res / (xv + 1e-12)).reshape(V, H, W)
    r2_var = [1 - res.reshape(V, -1)[v].sum() / xv.reshape(V, -1)[v].sum() for v in range(V)]

    tau = float(cfg.data.sst_tau)
    modes, ic, ih = [], 0, 0
    for m in truth["modes"]:
        if m["family"] == "stationary":
            continue
        name = f"cyc P{cfg.data.cyclic_periods[ic % len(cfg.data.cyclic_periods)]:.0f}" if m["family"] == "cyclic" \
            else f"chao {ih}"
        period = float(cfg.data.cyclic_periods[ic % len(cfg.data.cyclic_periods)]) if m["family"] == "cyclic" else None
        ic += m["family"] == "cyclic"
        ih += m["family"] == "chaotic"
        a = m["amp"].astype(np.float64)
        b = np.linalg.lstsq(S1[tr], a[tr], rcond=None)[0]
        dec = 1 - ((S1[va] @ b - a[va]) ** 2).mean() / a[va].var()
        patch = np.abs(m["phi"]) > PATCH_FRAC * np.abs(m["phi"]).max()
        vis = [v for v, nm in enumerate(("ssh", "sst")) if m["only"] in (nm, "both")]
        patch_r2 = float(np.median(np.concatenate([r2map[v][patch] for v in vis])))
        cap = float(np.cos(np.arctan(2 * np.pi * tau / period)) ** 2) if (period and m["only"] == "sst") else None
        modes.append(dict(name=name, family=m["family"], only=m["only"], decode=float(dec), patch_r2=patch_r2,
                          cover_ssh=float(M[:, 0][:, patch].mean(1).max()), cover_sst=float(M[:, 1][:, patch].mean(1).max()),
                          n_eyes=int(((M[:, 0][:, patch].mean(1) > 0.5) | (M[:, 1][:, patch].mean(1) > 0.5)).sum()),
                          sst_cap=cap, patch=patch))
    return dict(r2map=r2map, r2_var=r2_var, modes=modes, truth=truth)


def figure(c: dict, title: str, path: str):
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], hspace=0.35, wspace=0.12)
    for v, vn in enumerate(("SSH", "SST")):
        ax = fig.add_subplot(gs[0, v])
        im = ax.imshow(c["r2map"][v], cmap="Blues", vmin=0, vmax=1, origin="lower")
        for m in c["modes"]:                                 # outline every truth patch on the map
            if m["only"] in (vn.lower(), "both"):
                ax.contour(m["patch"].astype(float), levels=[0.5], colors=[FAM_COL[m["family"]]], linewidths=1.2)
        ax.set_title(f"{vn}: field reconstructed per cell from the eyes (validation R²), whole {vn} "
                     f"{c['r2_var'][v]:.3f}\noutlines = truth patches visible in {vn} (violet cyclic, pink chaotic)",
                     fontsize=9, loc="left")
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)

    ax = fig.add_subplot(gs[1, :])
    ms = c["modes"]
    x = np.arange(len(ms))
    ax.bar(x - 0.27, [m["patch_r2"] for m in ms], 0.26, color="#2a78d6", label="field R² inside the patch")
    ax.bar(x, [m["decode"] for m in ms], 0.26, color="#8c8b86", label="mode decoded from all eyes (R²)")
    ax.bar(x + 0.27, [max(m["cover_ssh"], m["cover_sst"]) for m in ms], 0.26, color="#d6d5cf",
           label="best mask coverage of the patch")
    for i, m in enumerate(ms):
        if m["sst_cap"] is not None:                          # the instant-decode ceiling of an SST-only cycle
            ax.plot([i - 0.13, i + 0.13], [m["sst_cap"]] * 2, color="#0b0b0b", lw=1.5)
        ax.text(i, 1.04, f"{m['only']}\n{m['n_eyes']} eye(s)", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, [m["name"] for m in ms], fontsize=9)
    for t, m in zip(ax.get_xticklabels(), ms):
        t.set_color(FAM_COL[m["family"]])
    ax.set_ylim(0, 1.18)
    ax.set_title("per truth mode (text: where it is visible, eyes covering > 50 % of its patch; black tick = "
                 "decode ceiling of an SST-only cycle)", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
    ax.grid(alpha=0.2, axis="y")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=".tmps/coverage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        c = coverage(run)
        print(f"\n=== {name}: whole field R2 SSH {c['r2_var'][0]:.3f}  SST {c['r2_var'][1]:.3f} ===")
        print(f"{'mode':>9} {'visible':>7} | {'patch R2':>8} {'decode':>6} {'cover SSH/SST':>13} {'eyes>50%':>8}")
        for m in c["modes"]:
            cap = f"  (SST-only cap {m['sst_cap']:.2f})" if m["sst_cap"] is not None else ""
            print(f"{m['name']:>9} {m['only']:>7} | {m['patch_r2']:8.2f} {m['decode']:6.2f} "
                  f"{m['cover_ssh']:6.2f}/{m['cover_sst']:.2f} {m['n_eyes']:>8}{cap}")
        figure(c, f"Coverage of {name}: can the field and every signal be read back from the eyes?",
               os.path.join(a.out, f"coverage_{name}.png"))


if __name__ == "__main__":
    main()
