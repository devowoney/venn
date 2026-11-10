"""One-seed check before building HSV members (D-043, design Q5a): is the ROLLOUT a usable base for the spread?

Two questions, on the deterministic twin, validation launches:
  1. accuracy — rollout from the true z_t (lead-1 fed back 64 times) vs the direct forecast (all leads at once):
     RMSE by lead per family, % of signal size (persistence for reference). Does the rollout drift?
  2. growth — how a small start perturbation evolves (tangent-linear, exact forward-mode JVP), rollout vs direct.
     Fear: the direct heads, trained toward the conditional mean, damp a chaotic start error at long lead.

Run:  python -m src.probes.hsv_rollout_check --run .tmps/runs_hf/twin_seed0 --out .tmps/eval_hf/hsv/rollout_check
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.func import jvp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.models.hsv_spread import CachedRollout            # noqa: E402
from src.probes.eval_history_forecaster import COL, FAMS, load_run   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_dir", type=int, default=8, help="random start directions per launch for the growth test")
    ap.add_argument("--n_launch_tl", type=int, default=256, help="launches used for the growth test")
    ap.add_argument("--chunks", default="1,8,16", help="rollout chunk sizes to compare (1 = pure rollout)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    model, nm, art, t_tr = load_run(args.run, dev)
    S = art["S"][:, nm["channels"]].astype(np.float64)
    T, K = S.shape
    A = model.A
    z = torch.tensor((S - nm["mu"]) / nm["sd"], dtype=torch.float32, device=dev)
    fams = [str(f) for f in nm["fam"]]
    size = np.sqrt(S[:t_tr].mean(0) ** 2 + S[:t_tr].std(0) ** 2)
    to_pct = 100 * nm["sd"] / size                                  # z units -> % of signal size, per channel
    cr = CachedRollout(model, z)
    chunks = [int(c) for c in args.chunks.split(",")]
    RK = [f"rollout c{c}" for c in chunks]

    # --- 1. accuracy: every validation launch whose 64 targets are all in validation -----------------------------
    L = torch.arange(t_tr - 1, T - A, device=dev)
    idx = L[:, None] + torch.arange(1, A + 1, device=dev)[None]
    y = z[idx]                                                      # [N,A,K]
    fc = {}
    with torch.no_grad():
        for ck, name in zip(chunks, RK):
            parts = [cr.rollout(c, z[c], A, chunk=ck) for c in torch.split(L, 512)]
            fc[name] = torch.cat([p[0] for p in parts])
        fc["direct"] = torch.cat([p[1] for p in parts])
    fc["persistence"] = z[L][:, None].expand_as(y)
    rmse = {k: (((v - y) ** 2).mean(0).sqrt().cpu().numpy() * to_pct) for k, v in fc.items()}   # [A,K] % of size
    finite = all(bool(torch.isfinite(fc[k]).all()) for k in RK)

    # --- 2. growth: tangent-linear response to random unit-RMS start perturbations ---------------------------------
    g = torch.Generator(device=dev).manual_seed(0)
    Lt = L[torch.linspace(0, len(L) - 1, args.n_launch_tl, device=dev).long()]
    grow = {k: [] for k in RK + ["direct"]}
    for _ in range(args.n_dir):
        v = torch.randn(len(Lt), K, device=dev, generator=g)
        v = v / v.pow(2).mean(1, keepdim=True).sqrt()                # unit RMS over channels (z units)
        for c, vc in zip(torch.split(Lt, 64), torch.split(v, 64)):
            for ck, name in zip(chunks, RK):
                _, tr = jvp(lambda x: cr.rollout(c, x, A, chunk=ck)[0], (z[c],), (vc,))
                grow[name].append(tr)
            _, td = jvp(lambda x: cr.rollout(c, x, 1)[1], (z[c],), (vc,))
            grow["direct"].append(td)
            _, tr = jvp(lambda x: cr.rollout(c, x, A)[0], (z[c],), (vc,))          # pure rollout, for the FD check
    gr = {k: torch.cat(v).pow(2).mean(0).sqrt().cpu().numpy() for k, v in grow.items()}   # [A,K] RMS response
    # finite-difference sanity check of the JVP (small nonlinear perturbation, one direction)
    with torch.no_grad():
        eps = 1e-3
        fd = (cr.rollout(c, z[c] + eps * vc, 16)[0] - cr.rollout(c, z[c], 16)[0]) / eps   # last chunk, last direction
    fd_err = float((fd - tr[:, :16]).norm() / tr[:, :16].norm())

    # --- report ----------------------------------------------------------------------------------------------------
    show = [1, 4, 8, 16, 32, 64]
    ch = {f: [i for i, ff in enumerate(fams) if ff == f] for f in FAMS}
    lines = [f"HSV rollout check (D-043 Q5a) — {args.run}", f"validation launches {len(L)}, channels {K}, "
             f"rollout finite: {finite}, JVP vs finite difference (16 steps): rel err {fd_err:.1e}", "",
             "RMSE, % of signal size        " + "".join(f"  h{a:<5}" for a in show)]
    for f in FAMS:
        if not ch[f]:
            continue
        for k in ["direct"] + RK + ["persistence"]:
            lines.append(f"  {f + ' ' + k:<30}" + "".join(f"  {rmse[k][a - 1, ch[f]].mean():6.1f}" for a in show))
    lines += ["", "growth of a unit-RMS start perturbation (RMS response, z units; 1 = neither grows nor decays)",
              "                               " + "".join(f"  h{a:<5}" for a in show)]
    for f in FAMS:
        if not ch[f]:
            continue
        for k in ["direct"] + RK:
            lines.append(f"  {f + ' ' + k:<30}" + "".join(f"  {gr[k][a - 1, ch[f]].mean():6.2f}" for a in show))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt + "\n")

    leads = np.arange(1, A + 1)
    STY = ["--", "-.", (0, (1, 1)), (0, (5, 1))]
    fig, axs = plt.subplots(2, 4, figsize=(22, 9))
    for j, f in enumerate(FAMS):
        ax = axs[0, j]
        if ch[f]:
            for k, ls in [("direct", "-")] + list(zip(RK, STY)) + [("persistence", ":")]:
                ax.plot(leads, rmse[k][:, ch[f]].mean(1), linestyle=ls, color=COL[f] if k != "persistence" else "0.5",
                        lw=2, label=k)
        ax.set_title(f"{f} (n={len(ch[f])}): RMSE, % of signal size")
        ax.set_xlabel("lead"), ax.legend(fontsize=8)
        ax = axs[1, j]
        if ch[f]:
            for k, ls in [("direct", "-")] + list(zip(RK, STY)):
                ax.plot(leads, gr[k][:, ch[f]].mean(1), linestyle=ls, color=COL[f], lw=2, label=k)
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{f}: response to a unit start perturbation")
        ax.set_xlabel("lead"), ax.legend(fontsize=8)
    fig.suptitle(f"Rollout vs direct on the deterministic twin — {os.path.basename(args.run)}, validation", fontsize=13)
    fig.savefig(os.path.join(args.out, "rollout_check.png"), dpi=100, bbox_inches="tight")
    print(f"[check] wrote {args.out}")


if __name__ == "__main__":
    main()
