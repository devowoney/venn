"""Look at the trained eyes directly: each kernel (mask on SSH and SST) next to the signal it extracts (2026-11-02).

One static figure per run:
  top row     the generator's truth patterns (SSH): the static pattern, the cycles, the chaotic patches -- where
              there is something to see
  16 blocks   per eye: mask on SSH | mask on SST (weights 0..1), and below them the extracted signal s_k(t) on a
              validation window, RAW (not standardized), so a stationary eye shows its large level with the small wiggle
              on top. Title = the eye's family (its own soft vote q), its level share, and the truth mode its
              fluctuation follows best (|corr|, raw SSH view or low-passed SST view).
The per-channel numbers come from `capture.capture` (same validation split and labeller).

Run:  conda run -n oceanai python -m src.probes.kernels --runs .tmps/runs/obs_t8000_pred_level_seed0
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.probes.capture import capture  # noqa: E402
from src.probes.eye_compare import FAM_COL  # noqa: E402

FAM_ORDER = ("stationary", "cyclic", "chaotic", "noise")
INK, MUTED = "#0b0b0b", "#52514e"


def mode_names(c: dict) -> list:
    """Readable names of the non-constant truth modes, in `capture`'s column order."""
    out, ic, ih = [], 0, 0
    for j in c["keep"]:
        if c["fam"][j] == "cyclic":
            out.append(f"cyc P{c['periods'][ic % len(c['periods'])]:.0f}")
            ic += 1
        else:
            out.append(f"chao {ih}")
            ih += 1
    return out


def figure(run: str, path: str):
    c = capture(run)
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    masks, phi, fam = art["masks"].astype(np.float64), art["truth_phi"].astype(np.float64), list(art["families"])
    K = masks.shape[0]
    names = mode_names(c)
    order = sorted(range(K), key=lambda k: (FAM_ORDER.index(c["q_fam"][k]) if c["q_fam"][k] in FAM_ORDER else 9, k))

    fig = plt.figure(figsize=(20, 27))
    outer = fig.add_gridspec(5, 1, height_ratios=[0.55, 1, 1, 1, 1], hspace=0.32)

    # --- truth patterns (where there is something to see) ---------------------------------------------
    top = outer[0].subgridspec(1, len(fam), wspace=0.08)
    tnames = ["static"] + names
    for j in range(len(fam)):
        ax = fig.add_subplot(top[0, j])
        v = np.abs(phi[j]).max() + 1e-12
        ax.imshow(phi[j], cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax.set_title(f"truth: {tnames[j]}", fontsize=8, color=FAM_COL.get(fam[j], INK))
        ax.set_xticks([]), ax.set_yticks([])

    # --- one block per eye, 4 per row -----------------------------------------------------------------
    t0 = int(c["va"].start)
    w = slice(t0, t0 + 400)
    for r in range(4):
        row = outer[r + 1].subgridspec(1, 4, wspace=0.18)
        for col in range(4):
            i = r * 4 + col
            if i >= K:
                continue
            k = order[i]
            blk = row[0, col].subgridspec(2, 2, height_ratios=[1, 0.8], hspace=0.35, wspace=0.05)
            famk = c["q_fam"][k]
            colk = FAM_COL.get(famk, MUTED)
            for v, vn in enumerate(("SSH", "SST")):
                ax = fig.add_subplot(blk[0, v])
                ax.imshow(masks[k, v], cmap="Blues", vmin=0, vmax=1, origin="lower")
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(vn, fontsize=7, color=MUTED)
                for s in ax.spines.values():
                    s.set_edgecolor(colk), s.set_linewidth(2)
            # what the fluctuation follows best: raw (SSH) or low-passed (SST) view of a truth mode
            jr, js = int(c["corr"][k].argmax()), int(c["corr_sst"][k].argmax())
            if c["corr"][k, jr] >= c["corr_sst"][k, js]:
                best = f"{names[jr]} {c['corr'][k, jr]:.2f}"
            else:
                best = f"{names[js]} (SST) {c['corr_sst'][k, js]:.2f}"
            ax = fig.add_subplot(blk[1, :])
            s = c["S"][w, k]
            ax.plot(np.arange(len(s)), s, color=colk, lw=1.2)
            ax.axhline(s.mean(), color=MUTED, lw=0.7, ls=":")
            ax.tick_params(labelsize=6)
            ax.set_xlabel("validation step", fontsize=6)
            lvl, sd = s.mean(), s.std()
            ax.set_title(f"level {lvl:.0f}, wiggle {sd:.0f} ({sd / (abs(lvl) + 1e-12):.0%} of level)",
                         fontsize=7, color=MUTED)
            fig.text(0, 0, "")                                   # keep layout engines quiet
            # block title above the masks
            ax_t = fig.add_subplot(blk[0, :], frameon=False)
            ax_t.set_xticks([]), ax_t.set_yticks([])
            src = f"q {c['q_max'][k]:.2f}" if np.isfinite(c["q_max"][k]) else "labeller"   # family-free eye: no vote
            ax_t.set_title(f"ch{k}  {famk} ({src})  level share {c['level_share'][k]:.2f}\n"
                           f"follows: {best}", fontsize=8, color=colk, pad=14)
    fig.suptitle(f"Trained eyes of {os.path.basename(run.rstrip('/'))}: kernel (mask) and extracted signal "
                 f"(raw scale, dotted = level). Masks: 0 = ignored, 1 = fully selected.", fontsize=13, y=0.995)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return c, order


def figure_layer2(run: str, path: str):
    """Layer 2 (D-039): one row per SPLIT layer-1 eye -- parent mask, its children's masks (SSH | SST side by side),
    and under each the signal it extracts (raw scale). Child titles: its own family vote, mass share of the parent,
    level share, and the truth mode its fluctuation follows best. Faded children (< fade_frac of the mass) are greyed.
    """
    c = capture(run)
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    kind, pidx, faded = art["kind"], art["parent_idx"], art["faded"]
    split = [int(k) for k in art["split_idx"]]
    plab = list(art["parent_label"])
    pm, pS, masks = art["parent_masks"].astype(np.float64), art["parent_S"].astype(np.float64), art["masks"]
    names = mode_names(c)
    t0 = int(c["va"].start)
    w = slice(t0, t0 + 400)
    nC = int((kind == "child").sum()) // max(len(split), 1)
    pmass = pm.sum(axis=(1, 2, 3))

    def two(m):                                                    # SSH | SST side by side, thin gap
        return np.concatenate([m[0], np.full((m.shape[1], 2), np.nan), m[1]], axis=1)

    fig = plt.figure(figsize=(4.0 * (nC + 1), 4.1 * len(split)))
    rows = fig.add_gridspec(len(split), 1, hspace=0.55)
    for r, k in enumerate(split):
        blk = rows[r].subgridspec(2, nC + 1, height_ratios=[1, 0.8], hspace=0.45, wspace=0.12)
        ax = fig.add_subplot(blk[0, 0])
        ax.imshow(two(pm[k]), cmap="Blues", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"PARENT ch{k} ({plab[k]})\nSSH | SST", fontsize=9, color=FAM_COL.get(plab[k], INK))
        ax = fig.add_subplot(blk[1, 0])
        ax.plot(pS[w, k], color=MUTED, lw=1.2)
        ax.tick_params(labelsize=6)
        ax.set_title(f"parent signal: level {pS[w, k].mean():.0f}, wiggle {pS[w, k].std():.0f}", fontsize=7, color=MUTED)
        kids = [i for i in range(len(kind)) if kind[i] == "child" and int(pidx[i]) == k]
        for j, i in enumerate(kids):
            fam = c["q_fam"][i]
            col = "#c9c8c2" if faded[i] else FAM_COL.get(fam, MUTED)
            share = float(masks[i].astype(np.float64).sum() / max(pmass[k], 1e-9))
            ax = fig.add_subplot(blk[0, j + 1])
            ax.imshow(two(masks[i].astype(np.float64)), cmap="Blues", vmin=0, vmax=1, origin="lower")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(col), s.set_linewidth(2)
            jr, js = int(c["corr"][i].argmax()), int(c["corr_sst"][i].argmax())
            best = (f"{names[jr]} {c['corr'][i, jr]:.2f}" if c["corr"][i, jr] >= c["corr_sst"][i, js]
                    else f"{names[js]} (SST) {c['corr_sst'][i, js]:.2f}")
            head = "faded" if faded[i] else f"{fam} (q {c['q_max'][i]:.2f})"
            ax.set_title(f"child {j + 1}: {head}\nmass {share:.0%}, level share {c['level_share'][i]:.2f}\n"
                         f"follows: {best}", fontsize=8, color=col)
            ax = fig.add_subplot(blk[1, j + 1])
            s = c["S"][w, i]
            ax.plot(s, color=col, lw=1.1)
            ax.axhline(s.mean(), color=MUTED, lw=0.6, ls=":")
            ax.tick_params(labelsize=6)
            ax.set_title(f"level {s.mean():.0f}, wiggle {s.std():.0f}", fontsize=7, color=MUTED)
    fig.suptitle(f"Layer 2 of {os.path.basename(run.rstrip('/'))}: each split eye and its children "
                 f"(raw signals; dotted = level; grey = faded child)", fontsize=13, y=1.0)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[".tmps/runs/obs_t8000_pred_level_seed0",
                                                   ".tmps/runs/obs_t8000_pred_level_seed2"])
    ap.add_argument("--out", default=".tmps/kernels_w2_level")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        art_files = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True).files
        (figure_layer2 if "kind" in art_files else figure)(run, os.path.join(a.out, f"kernels_{name}.png"))
        print(f"wrote {os.path.join(a.out, f'kernels_{name}.png')}")


if __name__ == "__main__":
    main()
