"""Atlas of a history forecaster on a LAYERED observer (D-041): every kernel, the signal it sees, and its forecast.

One folder of pages per forecaster run:
  00_overview.png     truth patterns of the generator (SSH) + every layer-2 kernel (max of its SSH and SST masks),
                      grouped by parent eye and
                      colored by family -- where each eye looks
  NN_parent_k.png     one page per SPLIT parent eye: the parent (layer-1 kernel + its signal, for context only) and, one
                      row per child, the child's kernel on SSH / SST and its signal over the validation set with the
                      forecasts of 4 launches: ensemble mean + 5-95 % member band (the uncertainty) + a few members
  99_cyclic_pass.png  the cyclic eyes layer 2 does not split (they are their own channel), same rows
Signals are in the observer's RAW units, so a stationary child shows its level with the small wiggle on top. Row titles:
family (observer's vote), share of the parent's mass, RMSE of the ensemble mean as % of signal size at h1/h16/h64,
spread/error at h16 (1 = honest), and the truth mode its fluctuation follows best (|corr| on validation).
Faded children (not in the history) are drawn with their kernel only.

Run:  python -m src.probes.atlas_history_forecaster --run .tmps/runs_hf/l2rw_all_seed0 --out .tmps/eval_hf/atlas/seed0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.probes.eval_history_forecaster import COL, load_run   # noqa: E402
from src.train.train_history_forecaster import ROLE_TO_FAMILY  # noqa: E402

MEMBERS = 32
N_LAUNCH, GAP = 4, 100            # 4 launches, 100 steps apart: each 64-step forecast is followed by a gap of truth


@torch.no_grad()
def forecasts(run: str, device: str):
    """Ensemble of every launch of the validation set, raw units, for the run's history channels.

    Returns ens [N, M, A, K] raw, launches L, the history S [T, K] raw, the run's norm and the observer artifacts."""
    model, nm, art, t_tr = load_run(run, device)
    ch = nm["channels"]
    S = art["S"][:, ch].astype(np.float64)
    T, A = S.shape[0], model.A
    L = np.arange(t_tr - 1, T - A)
    z = torch.tensor((S - nm["mu"]) / nm["sd"], dtype=torch.float32, device=device)
    torch.manual_seed(0)
    e = model(z[None, :T - 1], members=MEMBERS)[0]                    # [T-1, M, A, K], model z units
    ens = e[torch.as_tensor(L, device=device)].cpu().numpy() * nm["sd"] + nm["mu"]
    return ens, L, S, nm, art, t_tr


def channel_scores(ens, L, S, t_tr):
    """Per channel: RMSE of the ensemble mean as % of signal size [A,K] and spread / error [A,K]."""
    A = ens.shape[2]
    idx = L[:, None] + np.arange(1, A + 1)[None, :]
    y = S[idx]                                                         # [N, A, K]
    mean = ens.mean(1)
    err = np.sqrt(((mean - y) ** 2).mean(0))
    M = ens.shape[1]
    spread = np.sqrt((M + 1) / M * ens.var(1, ddof=1).mean(0))
    size = np.sqrt(S[:t_tr].mean(0) ** 2 + S[:t_tr].std(0) ** 2)
    return 100 * err / size, spread / np.maximum(err, 1e-12)


def truth_names(art):
    """Readable name of each generator mode (constant one = 'static'), numbered within its family."""
    out, n = [], {}
    for f in art["families"]:
        f = str(f)
        n[f] = n.get(f, 0) + 1
        out.append("static" if f not in ("cyclic", "chaotic") else f"{f[:4]} {n[f]}")
    return out


def best_mode(x, art, va):
    """The truth mode whose amplitude the signal's fluctuation follows best on validation: (name, |corr|)."""
    amp = art["truth_amp"][:, va].astype(np.float64)
    names = truth_names(art)
    xv = x[va] - x[va].mean()
    best, bc = "-", 0.0
    for j in range(amp.shape[0]):
        a = amp[j] - amp[j].mean()
        if a.std() < 1e-9 or xv.std() < 1e-12:
            continue
        c = abs(float((xv * a).mean() / (xv.std() * a.std())))
        if c > bc:
            best, bc = names[j], c
    return best, bc


def kernel_axes(fig, cell, mask, color, label):
    """Two small maps, mask on SSH and on SST (weights 0..1), framed in the family color."""
    sub = cell.subgridspec(1, 2, wspace=0.05)
    for v, vn in enumerate(("SSH", "SST")):
        ax = fig.add_subplot(sub[0, v])
        ax.imshow(mask[v], cmap="Blues", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(color), sp.set_linewidth(2)
        ax.set_title(f"{label} {vn}" if v == 0 else vn, fontsize=7, color=color, loc="left")


def signal_axes(ax, x, t_tr, fam, ens=None, L=None, j=None):
    """Truth over the validation window; if the channel is forecast, the 4 launches' fans on top."""
    t_a = t_tr + 200
    ts = np.arange(t_a - 40, t_a + N_LAUNCH * GAP + 10)
    ax.plot(ts, x[ts], color="k", lw=0.9, label="truth")
    col = COL.get(fam, "0.4")
    if ens is not None:
        A = ens.shape[2]
        for n_, t0 in enumerate(range(t_a, t_a + N_LAUNCH * GAP, GAP)):
            i0 = int(np.flatnonzero(L == t0)[0])
            mem = ens[i0, :, :, j]                                     # [M, A]
            tt = np.arange(t0 + 1, t0 + A + 1)
            lo, hi = np.percentile(mem, [5, 95], axis=0)
            ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0, label="5-95 % of members" if n_ == 0 else None)
            for m in range(4):
                ax.plot(tt, mem[m], color=col, lw=0.4, alpha=0.6, label="members" if (n_ == 0 and m == 0) else None)
            ax.plot(tt, mem.mean(0), color=col, lw=1.8, label="ensemble mean" if n_ == 0 else None)
            ax.plot([t0], [x[t0]], "o", color="k", ms=3.5, label="launch" if n_ == 0 else None)
    ax.tick_params(labelsize=7)
    ax.margins(x=0)


def row(fig, cell, mask, x, t_tr, fam, title, ens=None, L=None, j=None, legend=False):
    sub = cell.subgridspec(1, 2, width_ratios=[1, 3.2], wspace=0.08)
    kernel_axes(fig, sub[0, 0], mask, COL.get(fam, "0.4"), "")
    ax = fig.add_subplot(sub[0, 1])
    signal_axes(ax, x, t_tr, fam, ens, L, j)
    ax.set_title(title, fontsize=8.5, loc="left", color=COL.get(fam, "0.3"))
    if legend:
        ax.legend(fontsize=6.5, loc="upper right", ncol=5, framealpha=0.85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="history forecaster run dir trained on a LAYERED observer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ens, L, S, nm, art, t_tr = forecasts(args.run, args.device)
    rmse, ratio = channel_scores(ens, L, S, t_tr)
    T = S.shape[0]
    va = slice(t_tr, T)
    masks, S_all = art["masks"].astype(np.float64), art["S"].astype(np.float64)
    pidx, faded, kind = art["parent_idx"], art["faded"], art["kind"]
    roles = [ROLE_TO_FAMILY[str(r)] for r in art["roles"]]
    col_of = {int(c): j for j, c in enumerate(nm["channels"])}          # observer output -> history channel
    share = np.zeros(len(kind))
    for c in range(len(kind)):                                         # child's share of its parent's mask mass
        share[c] = masks[c].sum() / max(art["parent_masks"][pidx[c]].sum(), 1e-12)

    def child_title(c):
        nmode, cm = best_mode(S_all[:, c], art, va)
        head = f"out {c} ({kind[c]}) — {roles[c]}, {100 * share[c]:.0f} % of parent — follows {nmode} |r| {cm:.2f}"
        if c not in col_of:
            return head + " — FADED: not in the history"
        j = col_of[c]
        return head + (f"\nRMSE % of size h1/h16/h64 {rmse[0, j]:.1f} / {rmse[15, j]:.1f} / {rmse[63, j]:.1f}"
                       f" — spread/error h16 {ratio[15, j]:.2f}")

    # --- overview: truth patterns + every kernel (SSH) ---------------------------------------------------------
    phi = art["truth_phi"].astype(np.float64)
    tn = truth_names(art)
    parents = sorted(set(int(p) for p in pidx))
    ncol = max(len(phi), 9)
    fig = plt.figure(figsize=(ncol * 1.6, 1.9 * (len(parents) + 1) + 1))
    gs = fig.add_gridspec(len(parents) + 1, ncol, hspace=0.45, wspace=0.08, top=0.96, bottom=0.01)
    for i in range(len(phi)):
        ax = fig.add_subplot(gs[0, i])
        v = np.abs(phi[i]).max() + 1e-12
        ax.imshow(phi[i], cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"truth {tn[i]}", fontsize=8, color=COL.get(str(art["families"][i]), "k"))
    for r, p in enumerate(parents):
        ax = fig.add_subplot(gs[r + 1, 0])
        ax.imshow(art["parent_masks"][p].max(0), cmap="Greys", vmin=0, vmax=1, origin="lower")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"parent {p} ({art['parent_label'][p]})", fontsize=8)
        kids = [c for c in range(len(kind)) if pidx[c] == p]
        for i, c in enumerate(kids):
            ax = fig.add_subplot(gs[r + 1, i + 1])
            ax.imshow(masks[c].max(0), cmap="Blues", vmin=0, vmax=1, origin="lower")
            ax.set_xticks([]), ax.set_yticks([])
            colc = COL.get(roles[c], "0.4")
            for sp in ax.spines.values():
                sp.set_edgecolor(colc), sp.set_linewidth(2.5)
            ax.set_title(f"out {c}: {roles[c]}" + (" (faded)" if faded[c] else ""), fontsize=8, color=colc)
    fig.suptitle(f"{os.path.basename(args.run)}: generator truth (top) and every layer-2 kernel (max of its SSH and SST\n"
                 f"weights per cell), one row per parent eye (grey = parent, frame = family of the output)", fontsize=12)
    fig.savefig(os.path.join(args.out, "00_overview.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    # --- one page per split parent, one page for the cyclic pass-through eyes ---------------------------------
    pages = [(f"{n + 1:02d}_parent_{p}", [c for c in range(len(kind)) if pidx[c] == p and kind[c] == "child"], p)
             for n, p in enumerate([int(p) for p in art["split_idx"]])]
    pages.append(("99_cyclic_pass", [c for c in range(len(kind)) if kind[c] == "pass"], None))
    for name, outs, p in pages:
        nrow = len(outs) + (p is not None)
        fig = plt.figure(figsize=(17, 2.6 * nrow + 0.6))
        gs = fig.add_gridspec(nrow, 1, hspace=0.75)
        r0 = 0
        if p is not None:
            pm = art["parent_masks"][p].astype(np.float64)
            row(fig, gs[0], pm, art["parent_S"][:, p].astype(np.float64), t_tr, "parent",
                f"PARENT eye {p} (layer 1, labelled {art['parent_label'][p]}) — context only: its children below are "
                f"what the history forecaster reads and forecasts")
            r0 = 1
        for i, c in enumerate(outs):
            j = col_of.get(c)
            row(fig, gs[r0 + i], masks[c], S_all[:, c], t_tr, roles[c], child_title(c),
                ens if j is not None else None, L, j, legend=(i == 0))
        fig.suptitle(f"{os.path.basename(args.run)} — {name.split('_', 1)[1].replace('_', ' ')}: kernels (SSH | SST) "
                     f"and signals on validation, forecasts from 4 launches (raw observer units)", fontsize=11)
        fig.savefig(os.path.join(args.out, f"{name}.png"), dpi=90, bbox_inches="tight")
        plt.close(fig)
    print(f"[atlas] wrote {len(pages) + 1} pages to {args.out}")


if __name__ == "__main__":
    main()
