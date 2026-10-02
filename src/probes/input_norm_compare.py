"""Does the eye capture features better when it sees a STANDARDIZED field instead of the raw one?

Question (user, 2026-10-01): "the eye lobe sees the raw field; let's see what a standardized field as input
changes in the capturing performance." Variants are `train.input_norm` = none | cell_z | cell_scale (SOP 02).

For every eye (run dir), everything scored on the unseen validation slice [t_fit, T) unless stated:

  population   family labels of the K channels (the evaluate.py labeller on the full series) vs the 1/3/6 target
  recon R2/C   balanced linear decode of the GENERATOR field (same target for every variant), over the noise
               ceiling -- how much of the state the eye keeps (memory/reconstruction_score.md)
  mode capture per hidden mode: best single-channel |corr| (is there ONE channel that IS the mode?) and the
               linear decode R2 from all K channels (does the eye as a whole hold it?); stationary = the constant
               mode, scored by the labeller's flat-channel count instead (it has no variance to correlate)
  noise mass   share of the total mask mass on cells whose signal is below the noise (C_c < 0.5) -- the price
               of equal per-cell weighting: standardizing lifts pure-noise cells to unit variance

Writes input_norm_compare.json + figures into --out (default .tmps/input_norm/).

Run:  conda run -n oceanai python -m src.probes.input_norm_compare
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import GenConfig, generate_field, normalize_input  # noqa: E402
from src.probes.family import FAMILIES, amp_ratio, label_family, series_stats  # noqa: E402
from src.probes.observer_stability import oos_recon_r2  # noqa: E402

VARIANTS = ("none", "cell_z", "cell_scale")
VCOL = {"none": "#8c8b86", "cell_z": "#2a78d6", "cell_scale": "#eb6834"}
VLAB = {"none": "raw (current)", "cell_z": "cell z-score", "cell_scale": "cell ÷ std"}
GROUPS = {   # testbed -> run-dir pattern per variant (raw baselines were trained earlier, same protocol)
    "M=10": {"none": ".tmps/runs/obs_t8000_seed{s}", "cell_z": ".tmps/runs/obs_t8000_cell_z_seed{s}",
             "cell_scale": ".tmps/runs/obs_t8000_cell_scale_seed{s}", "seeds": range(5)},
    "M=40": {"none": ".tmps/runs/m40_K16_seed{s}", "cell_z": ".tmps/runs/m40_K16_cell_z_seed{s}",
             "cell_scale": ".tmps/runs/m40_K16_cell_scale_seed{s}", "seeds": range(3)},
}


def score_eye(run: str, device: str) -> tuple[dict, np.ndarray, np.ndarray]:
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    norm = cfg.train.get("input_norm", "none")
    field, truth = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)), seed=int(cfg.seed))
    T = field.shape[0]
    t_fit = int(art["t_fit"])
    masks = art["masks"].astype(np.float64)
    K = len(masks)
    S = np.einsum("tn,kn->tk", normalize_input(field, norm, t_fit).reshape(T, -1).astype(np.float64),
                  masks.reshape(K, -1))
    assert np.allclose(S, art["S"], rtol=1e-3, atol=1e-3 * np.abs(S).max()), f"{run}: re-encoded S != saved S"

    # population, labelled like evaluate.py
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-12)
    labels = [label_family(series_stats(Sz[:, i]), amp_ratio=amp_ratio(S[:, i])) for i in range(K)]
    pop = {f: labels.count(f) for f in FAMILIES}

    # reconstruction of the generator field, out of sample, over the noise ceiling
    Y = torch.from_numpy(field.reshape(T, -1)).to(device).double()
    _, r2 = oos_recon_r2(Y, torch.from_numpy(S).to(device), t_fit, t_fit)
    var_c = Y[:t_fit].var(0)
    ceil = float((1.0 - float(cfg.data.obs_noise) ** 2 / var_c).mean())

    # per hidden mode capture on validation
    val = slice(t_fit, T)
    X = np.concatenate([np.ones((T, 1)), (S - S[:t_fit].mean(0)) / (S[:t_fit].std(0) + 1e-12)], 1)
    modes = []
    for m in truth["modes"]:
        a = m["amp"].astype(np.float64)
        if a.std() < 1e-10:
            modes.append(dict(family=m["family"], best_corr=float("nan"), decode_r2=float("nan")))
            continue
        av = (a[val] - a[val].mean()) / a[val].std()
        best = float(np.abs(Sz[val].T @ av / (T - t_fit) / (Sz[val].std(0) + 1e-12)).max())
        beta = np.linalg.lstsq(X[:t_fit], a[:t_fit], rcond=None)[0]
        res = a[val] - X[val] @ beta
        modes.append(dict(family=m["family"], best_corr=best, decode_r2=float(1 - res.var() / a[val].var())))

    # mask mass on noise-dominated cells (signal variance < noise variance  <=>  C_c < 0.5)
    noise_cell = (1.0 - float(cfg.data.obs_noise) ** 2 / var_c).cpu().numpy() < 0.5
    noise_mass = float(masks.reshape(K, -1)[:, noise_cell].sum() / masks.sum())

    fam = lambda f, k: [x[k] for x in modes if x["family"] == f and not np.isnan(x[k])]  # noqa: E731
    out = dict(run=run, norm=norm, seed=int(cfg.seed), population=pop, n_flat=pop["stationary"],
               recon_r2=r2, ceiling=ceil, recon_over_ceiling=r2 / ceil, noise_cell_share=float(noise_cell.mean()),
               noise_mass=noise_mass, modes=modes,
               cyc_best=float(np.mean(fam("cyclic", "best_corr"))), cha_best=float(np.mean(fam("chaotic", "best_corr"))),
               cyc_dec=float(np.mean(fam("cyclic", "decode_r2"))), cha_dec=float(np.mean(fam("chaotic", "decode_r2"))))
    return out, masks, noise_cell.reshape(masks.shape[1:])


def fig_summary(R: dict, out: str):
    metrics = [("recon_over_ceiling", "reconstruction R² / ceiling"), ("cyc_best", "cyclic: best single |corr|"),
               ("cha_best", "chaotic: best single |corr|"), ("cyc_dec", "cyclic: decode R² (all K)"),
               ("cha_dec", "chaotic: decode R² (all K)"), ("noise_mass", "mask mass on noise cells")]
    fig, axes = plt.subplots(len(R), len(metrics) + 1, figsize=(3.0 * (len(metrics) + 1), 3.3 * len(R)), squeeze=False)
    for g, (grp, rows) in enumerate(R.items()):
        for j, (key, title) in enumerate(metrics):
            ax = axes[g, j]
            for v, var in enumerate(VARIANTS):
                vals = [r[key] for r in rows if r["norm"] == var]
                if not vals:
                    continue
                ax.bar(v, np.mean(vals), 0.65, color=VCOL[var])
                ax.scatter(np.full(len(vals), v) + np.linspace(-0.15, 0.15, len(vals)), vals, color="k", s=9, zorder=3)
                ax.text(v, np.mean(vals) + 0.03, f"{np.mean(vals):.2f}", ha="center", fontsize=8)
            ax.set_xticks(range(len(VARIANTS)), [VLAB[v] for v in VARIANTS], fontsize=7, rotation=15)
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{grp}: {title}", fontsize=9, loc="left")
        ax = axes[g, -1]                                     # family population, stacked per variant
        cols = {"stationary": "#1baf7a", "cyclic": "#4a3aa7", "chaotic": "#e87ba4"}
        for v, var in enumerate(VARIANTS):
            rows_v = [r for r in rows if r["norm"] == var]
            if not rows_v:
                continue
            bottom = 0.0
            for f in FAMILIES:
                h = np.mean([r["population"][f] for r in rows_v])
                ax.bar(v, h, 0.65, bottom=bottom, color=cols[f], edgecolor="white", linewidth=2,
                       label=f if v == 0 else None)
                ax.text(v, bottom + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=7, color="white")
                bottom += h
        ax.set_xticks(range(len(VARIANTS)), [VLAB[v] for v in VARIANTS], fontsize=7, rotation=15)
        ax.set_title(f"{grp}: family population (target 1/3/6 ×K/10)", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Eye input: raw vs standardized field (validation [8000,10000); dots = seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "input_norm_summary.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_masks(masks_by_var: dict, noise_cell: np.ndarray, title: str, path: str):
    """Coverage map per variant: how much mask sums on each cell (SSH | SST), noise-dominated cells outlined."""
    fig, axes = plt.subplots(len(masks_by_var), 2, figsize=(7, 3.3 * len(masks_by_var)), squeeze=False)
    for r, (var, m) in enumerate(masks_by_var.items()):
        cover = m.sum(0)                                    # [V,H,W]
        for v in range(2):
            ax = axes[r, v]
            ax.imshow(cover[v], cmap="Oranges", origin="lower", vmin=0, vmax=np.percentile(cover, 99))
            ax.contour(noise_cell[v].astype(float), levels=[0.5], colors="#2a78d6", linewidths=0.8)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{VLAB[var]} — {'SSH' if v == 0 else 'SST'} coverage", fontsize=9)
    fig.suptitle(title + "\n(blue outline: cells where noise > signal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmps/input_norm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    R = {}
    for grp, spec in GROUPS.items():
        R[grp] = []
        for s in spec["seeds"]:
            mk, nc = {}, None
            for var in VARIANTS:
                run = spec[var].format(s=s)
                if not os.path.exists(os.path.join(run, "artifacts.npz")):
                    print(f"  missing {run}")
                    continue
                res, mk[var], nc = score_eye(run, device)
                R[grp].append(res)
                print(f"{grp} seed {s} {var:>10}: pop {res['population']} R2/C {res['recon_over_ceiling']:.3f} "
                      f"cyc best/dec {res['cyc_best']:.2f}/{res['cyc_dec']:.2f} cha best/dec "
                      f"{res['cha_best']:.2f}/{res['cha_dec']:.2f} noise mass {res['noise_mass']:.2f} "
                      f"(noise cells {res['noise_cell_share']:.2f})", flush=True)
            if s == 0 and mk:
                fig_masks(mk, nc, f"{grp} seed 0: where the eyes look", os.path.join(a.out, f"masks_{grp.replace('=', '')}_seed0.png"))
    fig_summary(R, a.out)
    with open(os.path.join(a.out, "input_norm_compare.json"), "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
