"""Figures for a trained VENN run. Look at the result, don't take a metric's word for it.

Writes PNGs into <run>/figs/ (ephemeral, D-014):
  1. masks.png        -- the K optimized selection kernels, SSH and SST layers
  2. masks_on_energy.png -- mask outlines over the field's temporal-std map (where the dynamics are)
  3. features.png     -- the K extracted scalar series s_i(t), with family label + matched mode
  4. feature_var.png  -- rolling-window variance of each channel vs time

Viz belongs to the tools layer (D-003); this only READS artifacts and never touches training.

Run:  conda run -n oceanai python -m src.probes.plots [--run .tmps/runs/<ts>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from omegaconf import OmegaConf          # noqa: E402

from src.data.synthetic import GenConfig, generate_field   # noqa: E402

FAM_COLOR = {"stationary": "#3b7dd8", "cyclic": "#2ca05a", "chaotic": "#d1495b"}
VAR_NAME = ("SSH", "SST")


def _grid(K: int) -> tuple[int, int]:
    ncol = 8 if K >= 8 else K
    return int(np.ceil(K / ncol)), ncol


def fig_masks(masks: np.ndarray, rows: list[dict], out: str) -> None:
    """The optimized kernels themselves, one block of panels per variable layer."""
    K, V = masks.shape[0], masks.shape[1]
    nr, nc = _grid(K)
    fig, axes = plt.subplots(nr * V, nc, figsize=(1.55 * nc, 1.75 * nr * V), squeeze=False)
    for v in range(V):
        for k in range(K):
            ax = axes[v * nr + k // nc][k % nc]
            ax.imshow(masks[k, v], cmap="magma", vmin=0, vmax=1, origin="lower")
            r = rows[k]
            ax.set_title(f"ch{k}  {VAR_NAME[v] if v < len(VAR_NAME) else v}\n"
                         f"{r['count']:.0f} cells", fontsize=7,
                         color=FAM_COLOR.get(r["label"], "k"))
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(K, nr * nc):                       # blank any unused panels
            axes[v * nr + j // nc][j % nc].axis("off")
    fig.suptitle("Optimized selection kernels  (bright = selected; title colour = family label)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_masks_on_energy(masks: np.ndarray, field: np.ndarray, rows: list[dict], out: str) -> None:
    """Where the kernels chose to look, against how dynamic each cell actually is."""
    V = masks.shape[1]
    energy = field.std(axis=0)                            # [V,H,W] temporal std per cell
    fig, axes = plt.subplots(1, V, figsize=(6.2 * V, 5.6), squeeze=False)
    for v in range(V):
        ax = axes[0][v]
        im = ax.imshow(energy[v], cmap="Greys", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.046, label="temporal std of the field")
        for k in range(masks.shape[0]):
            r = rows[k]
            ax.contour(masks[k, v], levels=[0.5], colors=[FAM_COLOR.get(r["label"], "k")],
                       linewidths=1.4)
            cy, cx = r["cy"], r["cx"]
            ax.text(cx, cy, str(k), color=FAM_COLOR.get(r["label"], "k"), fontsize=8,
                    ha="center", va="center", fontweight="bold")
        ax.set_title(f"{VAR_NAME[v] if v < len(VAR_NAME) else v}: kernel outlines over field energy")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([], [], color=c, label=f) for f, c in FAM_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_features(S: np.ndarray, rows: list[dict], out: str) -> None:
    """The K extracted scalar channels through time -- the CLAUDE.md success readout."""
    T, K = S.shape
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    fig, ax = plt.subplots(figsize=(13, 0.62 * K + 1.6))
    for k in range(K):
        r = rows[k]
        ax.plot(np.arange(T), Z[:, k] * 0.42 - k, lw=0.7,
                color=FAM_COLOR.get(r["label"], "k"))
        ax.text(-T * 0.015, -k, f"ch{k}", ha="right", va="center", fontsize=8)
        ax.text(T * 1.004, -k,
                f"{r['label']}  gap1={r['gap1']:.4f}  ->m{r['best_mode']}"
                f"({r['best_mode_family'][:4]}) {abs(r['best_corr']):.2f}",
                ha="left", va="center", fontsize=7, color=FAM_COLOR.get(r["label"], "k"))
    ax.set_xlim(0, T); ax.set_ylim(-K + 0.3, 1.0)
    ax.set_yticks([]); ax.set_xlabel("time step")
    ax.set_title("Extracted features s_i(t)  (standardized; colour = family label)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_feature_var(S: np.ndarray, rows: list[dict], out: str, win: int = 100) -> None:
    """Rolling-window variance of every channel: is the extracted signal stationary in time?"""
    T, K = S.shape
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    ker = np.ones(win) / win
    m = np.stack([np.convolve(Z[:, k], ker, mode="valid") for k in range(K)], 1)
    m2 = np.stack([np.convolve(Z[:, k] ** 2, ker, mode="valid") for k in range(K)], 1)
    rv = np.clip(m2 - m ** 2, 0, None)                    # rolling variance [T-win+1, K]
    t = np.arange(len(rv)) + win // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    for k in range(K):
        r = rows[k]
        axes[0].plot(t, rv[:, k], lw=1.0, color=FAM_COLOR.get(r["label"], "k"), alpha=0.85)
        axes[0].text(t[-1] * 1.005, rv[-1, k], f"{k}", fontsize=7,
                     color=FAM_COLOR.get(r["label"], "k"), va="center")
    axes[0].set_xlabel("time step"); axes[0].set_ylabel(f"variance in a {win}-step window")
    axes[0].set_title("Feature variance in time (each channel standardized over the full record)")
    axes[0].axhline(1.0, color="k", ls=":", lw=0.9)
    handles = [plt.Line2D([], [], color=c, label=f) for f, c in FAM_COLOR.items()]
    axes[0].legend(handles=handles, frameon=False, fontsize=8)

    order = np.argsort([r["gap1"] for r in rows])
    axes[1].barh(range(K), [rows[i]["gap1"] for i in order],
                 color=[FAM_COLOR.get(rows[i]["label"], "k") for i in order])
    axes[1].set_yticks(range(K)); axes[1].set_yticklabels([f"ch{i}" for i in order], fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("lag-1 gap variance  var(ds)/var(s)")
    axes[1].set_title("per-channel slowness (slow -> fast)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest .tmps/runs/*)")
    ap.add_argument("--win", type=int, default=100, help="rolling-variance window")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run = os.path.abspath(args.run or sorted(glob.glob(os.path.join(repo, ".tmps/runs/*")))[-1])
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    with open(os.path.join(run, "eval.json")) as f:
        rows = json.load(f)["channels"]                   # needs `evaluate` to have run first
    cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    field, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)),
                              seed=int(cfg.seed))

    figs = os.path.join(run, "figs")
    os.makedirs(figs, exist_ok=True)
    fig_masks(art["masks"], rows, os.path.join(figs, "masks.png"))
    fig_masks_on_energy(art["masks"], field, rows, os.path.join(figs, "masks_on_energy.png"))
    fig_features(art["S"], rows, os.path.join(figs, "features.png"))
    fig_feature_var(art["S"], rows, os.path.join(figs, "feature_var.png"), args.win)
    print(f"[plots] wrote 4 figures to {os.path.relpath(figs, repo)}/")


if __name__ == "__main__":
    main()
