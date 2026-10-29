"""Compare eye variants side by side: fixed rung ladder vs SOFT family assignment (SOP 02 `role_mode`).

Question (user, 2026-10-29): instead of assigning each kernel's family before training, infer it from the
extracted signal, q_i = [q_stat, q_cyc, q_chao] with sum 1. Does the eye still capture the dynamics, and do the
families self-organize?

Scores per eye come from `input_norm_compare.score_eye` (population by the post-hoc labeller, out-of-sample
reconstruction R2/C, per-mode capture, mask mass on noise cells). Added here for soft eyes:
  agreement  share of channels whose argmax q names the same family as the post-hoc labeller
             (by construction they read the same quantities, so this checks the training reached a decided state)
  q figure   final q per channel (stacked bars) and q over training time from `q_snap`

Writes eye_compare.json + figures into --out (default .tmps/eye_compare/).

Run:  conda run -n oceanai python -m src.probes.eye_compare
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

from src.probes.input_norm_compare import score_eye  # noqa: E402

FAM_COL = {"stationary": "#1baf7a", "cyclic": "#4a3aa7", "chaotic": "#e87ba4"}
ROLE2FAM = {"slow": "stationary", "cyclic": "cyclic", "fast": "chaotic"}
PRESETS = {   # preset -> testbed -> variant -> run pattern (+ seeds); variants drawn in this order
    "soft_roles": {
        "M=10": {"fixed": ".tmps/runs/obs_t8000_seed{s}", "soft": ".tmps/runs/obs_t8000_soft_seed{s}", "seeds": range(5)},
        "M=40": {"fixed": ".tmps/runs/m40_K16_seed{s}", "soft": ".tmps/runs/m40_K16_soft_seed{s}", "seeds": range(3)},
    },
    # commitment levers on top of soft (2026-10-29): anneal = tau 1 -> 0.2, ent = lambda_ent 0.5
    "soft_commit": {
        "M=10": {"fixed": ".tmps/runs/obs_t8000_seed{s}", "soft": ".tmps/runs/obs_t8000_soft_seed{s}",
                 "anneal": ".tmps/runs/obs_t8000_soft_anneal_seed{s}", "ent": ".tmps/runs/obs_t8000_soft_ent_seed{s}",
                 "seeds": range(5)},
        "M=40": {"fixed": ".tmps/runs/m40_K16_seed{s}", "soft": ".tmps/runs/m40_K16_soft_seed{s}",
                 "anneal": ".tmps/runs/m40_K16_soft_anneal_seed{s}", "ent": ".tmps/runs/m40_K16_soft_ent_seed{s}",
                 "seeds": range(3)},
    },
    # population pull on top of the annealed soft default (2026-10-29): lambda_pop 0.5 (= anneal run) / 2 / 5
    "soft_pop": {
        "M=10": {"fixed": ".tmps/runs/obs_t8000_seed{s}", "pop0.5": ".tmps/runs/obs_t8000_soft_anneal_seed{s}",
                 "pop2": ".tmps/runs/obs_t8000_soft_pop2_seed{s}", "pop5": ".tmps/runs/obs_t8000_soft_pop5_seed{s}",
                 "seeds": range(5)},
    },
    # flatness measure in both modes (2026-10-29): ratio hinge (historical) vs log(amp_ratio) hinge (+ log q_stat)
    "soft_logflat": {
        "M=10": {"fixed": ".tmps/runs/obs_t8000_seed{s}", "fixed_log": ".tmps/runs/obs_t8000_logflat_seed{s}",
                 "soft": ".tmps/runs/obs_t8000_soft_pop5_seed{s}", "soft_log": ".tmps/runs/obs_t8000_soft_pop5_logflat_seed{s}",
                 "seeds": range(5)},
    },
    # whole-channel population target (2026-10-29), all on the new log-flatness default
    "soft_popint": {
        "M=10": {"fixed_log": ".tmps/runs/obs_t8000_logflat_seed{s}", "soft_log": ".tmps/runs/obs_t8000_soft_pop5_logflat_seed{s}",
                 "soft_int": ".tmps/runs/obs_t8000_soft_popint_seed{s}", "seeds": range(5)},
    },
}
VCOL = ["#8c8b86", "#2a78d6", "#eb6834", "#1baf7a"]
METRICS = [("recon_over_ceiling", "reconstruction R² / ceiling"), ("cyc_best", "cyclic: best single |corr|"),
           ("cha_best", "chaotic: best single |corr|"), ("cyc_dec", "cyclic: decode R² (all K)"),
           ("cha_dec", "chaotic: decode R² (all K)")]


def fig_summary(R: dict, variants: list, out: str):
    fig, axes = plt.subplots(len(R), len(METRICS) + 1, figsize=(3.0 * (len(METRICS) + 1), 3.3 * len(R)),
                             squeeze=False)
    for g, (grp, rows) in enumerate(R.items()):
        for j, (key, title) in enumerate(METRICS):
            ax = axes[g, j]
            for v, var in enumerate(variants):
                vals = [r[key] for r in rows if r["variant"] == var]
                if not vals:
                    continue
                ax.bar(v, np.mean(vals), 0.65, color=VCOL[v])
                ax.scatter(np.full(len(vals), v) + np.linspace(-0.15, 0.15, len(vals)), vals, color="k", s=9, zorder=3)
                ax.text(v, np.mean(vals) + 0.03, f"{np.mean(vals):.2f}", ha="center", fontsize=8)
            ax.set_xticks(range(len(variants)), variants, fontsize=8)
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{grp}: {title}", fontsize=9, loc="left")
        ax = axes[g, -1]                                     # family population by the labeller, stacked
        for v, var in enumerate(variants):
            rows_v = [r for r in rows if r["variant"] == var]
            if not rows_v:
                continue
            bottom = 0.0
            for f, col in FAM_COL.items():
                h = np.mean([r["population"][f] for r in rows_v])
                ax.bar(v, h, 0.65, bottom=bottom, color=col, edgecolor="white", linewidth=2, label=f if v == 0 else None)
                if h > 0.4:
                    ax.text(v, bottom + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=7, color="white")
                bottom += h
        ax.set_xticks(range(len(variants)), variants, fontsize=8)
        ax.set_title(f"{grp}: population (labeller; target 1/3/6 ×K/10)", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Eye variants, validation [8000,10000) (dots = seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eye_compare_summary.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_q(run: str, labels: list, title: str, path: str):
    """Final q per channel (stacked) + q over training per channel (one small panel each)."""
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    q, qs, steps = art["q"], art["q_snap"], art["snap_steps"]
    K = len(q)
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.3, 1, 1], hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(K)
    for j, (f, col) in enumerate(FAM_COL.items()):
        ax.bar(range(K), q[:, j], 0.75, bottom=bottom, color=col, edgecolor="white", linewidth=1.5, label=f"q_{f[:4]}")
        bottom += q[:, j]
    for i, lab in enumerate(labels):                          # the post-hoc labeller's verdict, for comparison
        ax.text(i, 1.03, lab[:4], ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(range(K), [str(i) for i in range(K)], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("final q (training slice)")
    ax.set_title("final soft family per channel; text = post-hoc labeller (full series)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.25))
    for i in range(K):
        a = fig.add_subplot(gs[1 + i // 8, i % 8])
        for j, col in enumerate(FAM_COL.values()):
            a.plot(steps, qs[:, i, j], color=col, lw=1.5)
        a.set_ylim(-0.03, 1.03)
        a.set_title(f"ch {i}", fontsize=8)
        a.tick_params(labelsize=6)
        if i % 8:
            a.set_yticklabels([])
    fig.suptitle(title + " — q over training steps (green stat, violet cyc, pink chao)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="soft_roles", choices=list(PRESETS))
    ap.add_argument("--out", default=".tmps/eye_compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = PRESETS[a.preset]
    variants = [k for k in next(iter(groups.values())) if k != "seeds"]
    R = {}
    for grp, spec in groups.items():
        R[grp] = []
        for s in spec["seeds"]:
            for var in variants:
                run = spec[var].format(s=s)
                if not os.path.exists(os.path.join(run, "artifacts.npz")):
                    print(f"  missing {run}")
                    continue
                res, _, _ = score_eye(run, device)
                res["variant"] = var
                art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
                # flatness r = |mean|/(|mean|+std) per channel on the training slice: how decisively flat the
                # stationary channels are (labeller cut 0.952, loss target 0.98)
                S_fit = art["S"][: int(art["t_fit"])].astype(np.float64)
                r = np.abs(S_fit.mean(0)) / (np.abs(S_fit.mean(0)) + S_fit.std(0))
                st = [i for i, lab in enumerate(res["labels"]) if lab == "stationary"]
                res["stat_r"] = float(r[st].mean()) if st else float("nan")
                res["n_r98"] = int((r >= 0.98).sum())
                if "q" in art.files:
                    fam_q = [ROLE2FAM[r] for r in art["roles"]]
                    res["q"] = art["q"].tolist()
                    res["agreement"] = float(np.mean([x == y for x, y in zip(fam_q, res["labels"])]))
                    res["q_population"] = {f: fam_q.count(f) for f in FAM_COL}
                    if s == 0:
                        fig_q(run, res["labels"], f"{grp} seed 0 ({var})",
                              os.path.join(a.out, f"q_{grp.replace('=', '')}_{var}_seed0.png"))
                R[grp].append(res)
                extra = (f" | argmax-q pop {res['q_population']} agree {res['agreement']:.2f}" if "q" in res else "")
                extra += f" | stat r {res['stat_r']:.3f} n(r>=.98) {res['n_r98']}"
                print(f"{grp} seed {s} {var:>6}: pop {res['population']} R2/C {res['recon_over_ceiling']:.3f} "
                      f"cyc best/dec {res['cyc_best']:.2f}/{res['cyc_dec']:.2f} cha best/dec "
                      f"{res['cha_best']:.2f}/{res['cha_dec']:.2f}{extra}", flush=True)
    fig_summary(R, variants, a.out)
    with open(os.path.join(a.out, "eye_compare.json"), "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
