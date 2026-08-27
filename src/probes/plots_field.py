"""Figures + animations of the synthetic SSH/SST testbed, and where the kernels activate on it.

Writes into <run>/figs/ (or .tmps/figs_field/ when no run is given):
  1. field_snapshots.png -- SSH & SST at several times, plus the mean and temporal-std maps
  2. field_modes.png     -- the 10 HIDDEN ground-truth modes: phi_k(x,y) + a_k(t) + family
  3. kernel_activation.png -- aggregate map: where the K kernels look, weighted by how much
                            signal each one actually carries; plus per-kernel activation strength
  4. field.gif           -- the field evolving in time with the kernel footprints drawn on it,
                            each outline PULSING with that channel's instantaneous activation,
                            next to a live bar chart of all K channel values

The GIF is the "where is the kernel activated most" view: bright/thick outline = that kernel is
strongly activated at that instant; the bar panel ranks all K at once.

Viz only -- reads artifacts, never trains (D-003).

Run:  conda run -n oceanai python -m src.probes.plots_field                  # newest run
      conda run -n oceanai python -m src.probes.plots_field --no-run         # field only
      conda run -n oceanai python -m src.probes.plots_field --frames 200 --stride 4
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402
from omegaconf import OmegaConf                       # noqa: E402

from src.data.synthetic import GenConfig, generate_field      # noqa: E402

VAR_NAME = ("SSH", "SST")
FAM_COLOR = {"stationary": "#3b7dd8", "cyclic": "#2ca05a", "chaotic": "#d1495b"}


def _boundary(mask2d: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Outline pixels of a mask: selected cells that touch a non-selected cell. [H,W] bool."""
    m = mask2d > thr
    er = m.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        er &= np.roll(m, sh, axis=ax)
    return m & ~er


# ------------------------------------------------------------------------------ static figures
def fig_snapshots(field: np.ndarray, out: str, n: int = 5) -> None:
    """SSH/SST at n times + the mean and temporal-std maps, on a shared symmetric colour scale."""
    T, V = field.shape[0], field.shape[1]
    ts = np.linspace(0, T - 1, n).astype(int)
    lim = float(np.percentile(np.abs(field), 99))
    fig, axes = plt.subplots(V, n + 2, figsize=(2.05 * (n + 2), 2.3 * V), squeeze=False)
    for v in range(V):
        for j, t in enumerate(ts):
            ax = axes[v][j]
            im = ax.imshow(field[t, v], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
            ax.set_title(f"{VAR_NAME[v]}  t={t}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = axes[v][n]
        ax.imshow(field[:, v].mean(0), cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
        ax.set_title(f"{VAR_NAME[v]}  time mean", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[v][n + 1]
        im2 = ax.imshow(field[:, v].std(0), cmap="magma", origin="lower")
        ax.set_title(f"{VAR_NAME[v]}  temporal std", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Synthetic SSH/SST testbed — snapshots, time mean, and where the variance lives",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_modes(truth: dict, out: str, tmax: int = 600) -> None:
    """The hidden answer key: each injected mode's spatial pattern and its amplitude in time."""
    modes = truth["modes"]
    n = len(modes)
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 4.4), squeeze=False,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    for k, m in enumerate(modes):
        c = FAM_COLOR[m["family"]]
        ax = axes[0][k]
        ax.imshow(m["phi"], cmap="magma", origin="lower", vmin=0, vmax=1)
        ax.set_title(f"m{k}\n{m['family']}\n({m['scale']})", fontsize=7, color=c)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[1][k]
        ax.plot(m["amp"][:tmax], lw=0.7, color=c)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[1][0].set_ylabel(f"a_k(t), first {tmax}", fontsize=7)
    fig.suptitle("HIDDEN ground-truth modes:  spatial pattern phi_k  (top)  /  amplitude a_k(t)  "
                 "(bottom).  Never shown to the model.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _unique_contribution(field: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Drop-one importance per channel: how much balanced R^2 is LOST without it. [K]

    Scale-fair, unlike std(s_i) -- which just re-plots the mask-size ladder, since a 1143-cell
    footprint sums 140x more cells than an 8-cell one. This asks instead what each kernel
    contributes that no other kernel already provides.
    """
    T, K = S.shape
    Y = field.reshape(T, -1)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)                # per-cell standardized (balanced)
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    sst = (Y ** 2).sum()

    def r2(cols: list[int]) -> float:
        X = np.c_[np.ones(T), Z[:, cols]]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return float(1.0 - ((Y - X @ beta) ** 2).sum() / sst)

    full = r2(list(range(K)))
    return np.array([full - r2([j for j in range(K) if j != i]) for i in range(K)])


def fig_kernel_activation(field: np.ndarray, masks: np.ndarray, S: np.ndarray, out: str) -> None:
    """Where the kernels look, weighted by each one's UNIQUE contribution to the encoding."""
    V = masks.shape[1]
    uniq = _unique_contribution(field, S)                   # [K] drop-one delta balanced R^2
    w = np.clip(uniq, 0, None)
    w = w / (w.sum() + 1e-12)
    cover = np.einsum("k,kvhw->vhw", w, masks)              # [V,H,W] weighted sensor coverage
    plain = masks.sum(axis=0)                               # [V,H,W] plain overlap count
    fig, axes = plt.subplots(1, V + 1, figsize=(5.0 * (V + 1) * 0.8, 4.4))
    for v in range(V):
        ax = axes[v]
        im = ax.imshow(cover[v], cmap="viridis", origin="lower")
        ax.contour(field[:, v].std(0), levels=4, colors="w", linewidths=0.6, alpha=0.7)
        plt.colorbar(im, ax=ax, fraction=0.046, label="kernel coverage x unique contribution")
        ax.set_title(f"{VAR_NAME[v]}: where the kernels are activated\n"
                     f"(white contours = field temporal std)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[V]
    order = np.argsort(-uniq)
    cnt = masks.reshape(masks.shape[0], -1).sum(1)
    ax.barh(range(len(order)), uniq[order], color=["#3b7dd8" if u > 0 else "#d1495b"
                                                   for u in uniq[order]])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"ch{i} ({cnt[i]:.0f}c)" for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("UNIQUE contribution: balanced R$^2$ lost if this kernel is dropped")
    ax.set_title("which kernels actually matter\n(c = footprint in cells)", fontsize=9)
    frac = (plain.max(), float((masks.max(axis=0) > 0.5).mean()))
    n_dead = int((uniq < 1e-4).sum())
    fig.suptitle(f"Kernel activation — domain covered by >=1 kernel: {frac[1]*100:.1f}%   |   "
                 f"max kernels stacked on a cell: {frac[0]:.1f}   |   "
                 f"kernels contributing ~nothing unique: {n_dead}/{masks.shape[0]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------- animation
def gif_field(field: np.ndarray, out: str, masks: np.ndarray | None = None,
              S: np.ndarray | None = None, frames: int = 150, stride: int = 4,
              fps: int = 12) -> None:
    """Animate the field; overlay kernel outlines that pulse with each channel's activation."""
    V = field.shape[1]
    H, W = field.shape[2], field.shape[3]
    lim = float(np.percentile(np.abs(field), 99))
    ts = (np.arange(frames) * stride) % field.shape[0]
    Z = None
    if S is not None:
        Z = (S - S.mean(0)) / (S.std(0) + 1e-12)            # [T,K] standardized activations
        K = Z.shape[1]
        bnds = np.stack([np.stack([_boundary(masks[k, v]) for v in range(V)])
                         for k in range(K)])                # [K,V,H,W] bool outlines
        cmap = plt.get_cmap("turbo")

    ncol = V + (1 if Z is not None else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * V + (3.1 if Z is not None else 0), 4.9),
                             gridspec_kw={"width_ratios": [1] * V + ([0.62] if Z is not None else [])})
    axes = np.atleast_1d(axes)
    ims, ovs = [], []
    for v in range(V):
        ax = axes[v]
        ims.append(ax.imshow(field[ts[0], v], cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower"))
        ovs.append(ax.imshow(np.zeros((H, W, 4)), origin="lower", interpolation="nearest"))
        ax.set_title(f"{VAR_NAME[v] if v < len(VAR_NAME) else v}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    if Z is not None:
        axb = axes[V]
        bars = axb.barh(range(K), np.zeros(K), color="#888888")
        axb.set_xlim(-3.2, 3.2); axb.set_ylim(-0.6, K - 0.4); axb.invert_yaxis()
        axb.axvline(0, color="k", lw=0.7)
        axb.set_yticks(range(K)); axb.set_yticklabels([f"ch{i}" for i in range(K)], fontsize=7)
        axb.set_xlabel("channel activation  s_i(t)  (standardized)", fontsize=8)
        axb.set_title("which kernel is activated most", fontsize=10)
    ttl = fig.suptitle("", fontsize=12)

    def draw(fi: int):
        t = int(ts[fi])
        for v in range(V):
            ims[v].set_data(field[t, v])
        if Z is not None:
            a = np.clip(np.abs(Z[t]) / 2.5, 0.12, 1.0)              # outline opacity  [K]
            col = cmap(np.clip(np.abs(Z[t]) / 3.0, 0, 1))           # outline colour   [K,4]
            for v in range(V):
                rgba = np.zeros((H, W, 4))
                for k in np.argsort(np.abs(Z[t])):                  # strongest drawn last (on top)
                    b = bnds[k, v]
                    if b.any():
                        rgba[b] = (*col[k][:3], a[k])
                ovs[v].set_data(rgba)
            for k, bar in enumerate(bars):
                bar.set_width(Z[t, k])
                bar.set_color(cmap(min(1.0, abs(Z[t, k]) / 3.0)))
            top = int(np.argmax(np.abs(Z[t])))
            ttl.set_text(f"t = {t:4d}     most-activated kernel: ch{top} "
                         f"(s = {Z[t, top]:+.2f})     outline colour/opacity = |activation|")
        else:
            ttl.set_text(f"t = {t:4d}")
        return ims + ovs

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    anim = FuncAnimation(fig, draw, frames=len(ts), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)


# --------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest .tmps/runs/*)")
    ap.add_argument("--no-run", action="store_true", help="field only, no kernel overlay")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=4, help="time steps between GIF frames")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run = None
    if not args.no_run:
        cands = sorted(glob.glob(os.path.join(repo, ".tmps/runs/*")))
        run = os.path.abspath(args.run) if args.run else (cands[-1] if cands else None)

    if run:
        cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
        gen_cfg = GenConfig(**OmegaConf.to_container(cfg.data, resolve=True))
        seed = int(cfg.seed)
        art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
        masks, S = art["masks"], art["S"]
        figs = os.path.join(run, "figs")
    else:
        gen_cfg, seed, masks, S = GenConfig(), 0, None, None
        figs = os.path.join(repo, ".tmps/figs_field")
    os.makedirs(figs, exist_ok=True)

    field, truth = generate_field(gen_cfg, seed=seed)
    print(f"[field] field{field.shape} seed={seed}"
          + (f"  kernels from {os.path.relpath(run, repo)}" if run else "  (no kernel overlay)"))

    fig_snapshots(field, os.path.join(figs, "field_snapshots.png"))
    fig_modes(truth, os.path.join(figs, "field_modes.png"))
    if masks is not None:
        fig_kernel_activation(field, masks, S, os.path.join(figs, "kernel_activation.png"))
    gif_field(field, os.path.join(figs, "field.gif"), masks, S,
              frames=args.frames, stride=args.stride, fps=args.fps)
    print(f"[field] wrote figures + field.gif to {os.path.relpath(figs, repo)}/")


if __name__ == "__main__":
    main()
