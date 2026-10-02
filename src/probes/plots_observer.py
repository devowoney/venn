"""Animated inspection of the eye-lobe (observer): what each channel looks at, and what it then sees.

Two GIFs from one run dir that carries `masks_snap` (`train.snap_every` > 0):

  observer_training.gif  -- TRAINING time. One frame per mask snapshot. Row i = channel i:
                            its mask on SSH | on SST (the "activated pixels"), and the feature s_i(t) it
                            produces on a validation window. The title of each row carries the assigned
                            rung, the family the readout gives the series right now, amp_ratio, and the
                            IoU of the activated pixels against the final mask. Watch whether the eye
                            settles and whether each row turns into the family its rung asks for.
  observer_state.gif     -- PHYSICAL time, final observer. The SSH/SST state x(t) moving over the
                            validation window, every channel's final footprint drawn as a role-coloured
                            contour, and the 16 feature traces with a cursor at t.

Features are plotted RAW with zero kept on the y-axis, never standardized: standardizing would make a
flat (stationary) channel -- a large constant level with a little noise -- look exactly like chaos.

Run:  conda run -n oceanai python -m src.probes.plots_observer --run .tmps/runs/obs_t8000_seed0
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import GenConfig, generate_field, normalize_input  # noqa: E402
from src.probes.family import amp_ratio, label_family, series_stats  # noqa: E402

# categorical slots 1-3 of the validated reference palette (dataviz skill), fixed order = rung order
ROLE_COL = dict(slow="#2a78d6", cyclic="#eb6834", fast="#1baf7a")
ROLE_FAM = dict(slow="stationary", cyclic="cyclic", fast="chaotic")
INK, MUTED = "#0b0b0b", "#52514e"


def _ylim(y: np.ndarray) -> tuple[float, float]:
    """Y-range that always contains 0, so a constant level reads as flat and an anomaly as a wiggle."""
    lo, hi = min(0.0, float(y.min())), max(0.0, float(y.max()))
    pad = 0.08 * (hi - lo + 1e-12)
    return lo - pad, hi + pad


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel IoU of two binarized banks [K,...] -> [K]."""
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a & b).sum(1) / np.maximum((a | b).sum(1), 1)


def training_gif(field, snaps, steps, roles, t_fit, win, out, fps):
    """One frame per snapshot: per-channel mask (SSH|SST) + its feature on the window `win`."""
    K = snaps.shape[1]
    on = snaps > 0.5
    F = field.reshape(len(field), -1)                               # [T,N]
    # every snapshot's features, and the readout's verdict on the TRAINING slice (what it was fitted on)
    S = np.einsum("tn,jkn->jtk", F, snaps.reshape(len(snaps), K, -1))   # [n_snap,T,K]
    fam = [[label_family(series_stats(S[j, :t_fit, i]), amp_ratio=amp_ratio(S[j, :t_fit, i]))
            for i in range(K)] for j in range(len(snaps))]
    t = np.arange(*win)

    fig, axs = plt.subplots(K, 3, figsize=(12, 1.15 * K + 0.8),
                            gridspec_kw=dict(width_ratios=[1, 1, 4.2], hspace=0.6, wspace=0.06))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    ims, lines, titles = [], [], []
    for i in range(K):
        for v in range(2):
            ax = axs[i, v]
            ims.append(ax.imshow(snaps[0, i, v], cmap="Greys", vmin=0, vmax=1, interpolation="nearest"))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(("SSH mask", "SST mask")[v], fontsize=9, color=INK)
        ax = axs[i, 2]
        (ln,) = ax.plot(t, S[0, slice(*win), i], color=ROLE_COL[roles[i]], lw=1.2)
        lines.append(ln)
        ax.tick_params(labelsize=6, colors=MUTED)
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
        titles.append(ax.set_title("", fontsize=7.5, loc="left", color=INK))
    axs[-1, 2].set_xlabel("time step (validation window)", fontsize=8, color=MUTED)
    sup = fig.suptitle("", fontsize=11, color=INK)

    def draw(j):
        iou = _iou(on[j], on[-1])
        for i in range(K):
            for v in range(2):
                ims[2 * i + v].set_data(snaps[j, i, v])
            y = S[j, slice(*win), i]
            lines[i].set_ydata(y)
            axs[i, 2].set_ylim(*_ylim(y))
            ok = "ok" if fam[j][i] == ROLE_FAM[roles[i]] else "MISMATCH"
            titles[i].set_text(f"ch{i:<2d} rung={roles[i]:<6s} readout={fam[j][i]:<10s} ({ok})   "
                               f"amp_r={amp_ratio(S[j, :t_fit, i]):.3f}   active px={on[j, i].sum():4d}   "
                               f"IoU->final={iou[i]:.2f}")
        n_ok = sum(fam[j][i] == ROLE_FAM[roles[i]] for i in range(K))
        sup.set_text(f"Eye-lobe over TRAINING   step {steps[j]:5d}/{steps[-1]}   "
                     f"mean IoU->final {iou.mean():.2f}   channels obeying their rung {n_ok}/{K}")
        return ims + lines + titles

    anim = FuncAnimation(fig, draw, frames=len(snaps), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)


def state_gif(field, mask, roles, S, win, stride, out, fps):
    """Physical time: the state x(t), final footprints as contours, feature traces with a cursor."""
    K = mask.shape[0]
    t_all = np.arange(*win)
    frames = t_all[::stride]
    lim = np.percentile(np.abs(field[slice(*win)]), 99.5)          # one fixed symmetric colour scale

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(4, 7, width_ratios=[2.2, 0.09, 0.28, 1, 1, 1, 1], hspace=0.6, wspace=0.35,
                          left=0.03, right=0.98, top=0.9, bottom=0.1)
    ims = []
    for v in range(2):
        ax = fig.add_subplot(gs[v * 2:(v + 1) * 2, 0])             # SSH on top, SST below
        ims.append(ax.imshow(field[frames[0], v], cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"))
        # footprints of the COMPACT rungs only, in dark ink so they never compete with the field's colours.
        # The flat rungs read scattered pixels (see the training GIF) and would bury the map as contours.
        for i in range(K):
            if roles[i] == "slow" or not (mask[i, v] > 0.5).any():
                continue
            ax.contour(mask[i, v], levels=[0.5], colors=[INK], linewidths=0.8,
                       linestyles="solid" if roles[i] == "cyclic" else "dashed")
            yy, xx = np.nonzero(mask[i, v] > 0.5)
            ax.text(xx.mean(), yy.mean(), str(i), fontsize=7, color=INK, ha="center", va="center",
                    fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(("SSH-like state x(t)", "SST-like state x(t)")[v], fontsize=9, color=INK)
        cb = fig.colorbar(ims[v], cax=fig.add_subplot(gs[v * 2:(v + 1) * 2, 1]))
        cb.ax.tick_params(labelsize=7)
    cursors = []
    for i in range(K):
        ax = fig.add_subplot(gs[i // 4, 3 + i % 4])
        y = S[slice(*win), i]
        ax.plot(t_all, y, color=ROLE_COL[roles[i]], lw=0.9)
        ax.set_ylim(*_ylim(y))
        ax.axhline(0, color=MUTED, lw=0.5, alpha=0.5)
        cursors.append(ax.axvline(frames[0], color=INK, lw=0.8))
        ax.set_title(f"ch{i} {roles[i]}", fontsize=7.5, color=INK, loc="left")
        ax.tick_params(labelsize=5.5, colors=MUTED)
    handles = [plt.Line2D([], [], color=c, lw=2, label=f"{r} rung") for r, c in ROLE_COL.items()]
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=3, frameon=False)
    fig.text(0.02, 0.02, "map contours = final footprints: solid = cyclic rung, dashed = fast rung, number = channel "
             "(flat rungs read scattered pixels, not drawn)", fontsize=7.5, color=MUTED)
    sup = fig.suptitle("", fontsize=11, color=INK)

    def draw(k):
        tt = frames[k]
        for v in range(2):
            ims[v].set_data(field[tt, v])
        for c in cursors:
            c.set_xdata([tt, tt])
        sup.set_text(f"State and observer features over PHYSICAL time   t = {tt}")
        return ims + cursors

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with masks_snap (train.snap_every > 0)")
    ap.add_argument("--win", type=int, nargs=2, default=[8000, 8600], help="time window shown [a, b)")
    ap.add_argument("--stride", type=int, default=4, help="state GIF: time steps per frame")
    ap.add_argument("--out", default=".tmps/observer_viz")
    ap.add_argument("--only", choices=["training", "state"], default=None, help="render just one GIF")
    args = ap.parse_args()

    art = np.load(os.path.join(args.run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(args.run, ".hydra/config.yaml"))
    field, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)), seed=cfg.seed)
    # features are re-encoded from the snapshots, so feed the eye its own input (SOP 02 input_norm)
    field_in = normalize_input(field, cfg.train.get("input_norm", "none"), int(art["t_fit"]))
    roles = [str(r) for r in art["roles"]]
    snaps = art["masks_snap"].astype(np.float32)                    # [n,K,V,H,W]
    name = os.path.basename(os.path.normpath(args.run))
    os.makedirs(args.out, exist_ok=True)

    if args.only in (None, "training"):
        p = os.path.join(args.out, f"{name}_training.gif")
        training_gif(field_in, snaps, art["snap_steps"], roles, int(art["t_fit"]), args.win, p, fps=3)
        print(f"[viz] wrote {p}")
    if args.only in (None, "state"):
        p = os.path.join(args.out, f"{name}_state.gif")
        state_gif(field, snaps[-1], roles, art["S"], args.win, args.stride, p, fps=12)
        print(f"[viz] wrote {p}")


if __name__ == "__main__":
    main()
