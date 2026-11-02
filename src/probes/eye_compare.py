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
from omegaconf import OmegaConf  # noqa: E402

from src.probes.family import PRED_FAMILY_NAMES, label_predictability  # noqa: E402
from src.probes.input_norm_compare import score_eye  # noqa: E402

FAM_COL = {"stationary": "#1baf7a", "cyclic": "#4a3aa7", "chaotic": "#e87ba4", "noise": "#8c8b86"}   # noise: D-035
ROLE2FAM = {"slow": "stationary", "cyclic": "cyclic", "fast": "chaotic", "noise": "noise"}
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
    # W4 (D-034, 2026-10-30): the population is no longer pulled. pop5 = the 2026-10-29 soft default
    # (lambda_pop 5, whole-channel target), pop0 = the new default (l_pop logged only)
    "w4_nopop": {
        "M=10": {"fixed_log": ".tmps/runs/obs_t8000_logflat_seed{s}", "pop5": ".tmps/runs/obs_t8000_soft_popint_seed{s}",
                 "pop0": ".tmps/runs/obs_t8000_soft_pop0_seed{s}", "seeds": range(5)},
    },
    # W2 step 2b (D-035, 2026-10-30): predictability family measure, 4 families, NO population target (user)
    "w2_pred": {
        "M=10": {"fixed_log": ".tmps/runs/obs_t8000_logflat_seed{s}", "pop5": ".tmps/runs/obs_t8000_soft_popint_seed{s}",
                 "pred0": ".tmps/runs/obs_t8000_pred_seed{s}", "seeds": range(5)},
    },
    # W2 follow-up (2026-11-02): TIGHTER cyclic criterion only (p_cut 0.5 -> 0.8, p_target 0.9 -> 0.97), still no
    # population target; read every eye with --p_cut 0.8
    "w2_tight": {
        "M=10": {"pop5": ".tmps/runs/obs_t8000_soft_popint_seed{s}", "pred0": ".tmps/runs/obs_t8000_pred_seed{s}",
                 "pred_tight": ".tmps/runs/obs_t8000_pred_tight_seed{s}", "seeds": range(5)},
    },
    # + noise pushed toward structure (pred.noise_penalty: structure, 2026-11-02); read with --p_cut 0.8
    "w2_noisepen": {
        "M=10": {"pop5": ".tmps/runs/obs_t8000_soft_popint_seed{s}", "pred_tight": ".tmps/runs/obs_t8000_pred_tight_seed{s}",
                 "pred_noisepen": ".tmps/runs/obs_t8000_pred_noisepen_seed{s}", "seeds": range(5)},
    },
    # D-037 (2026-11-02): count the LEVEL as signal (train.signal_moment=second) on top of the D-036 defaults
    "w2_level": {
        "M=10": {"pop5": ".tmps/runs/obs_t8000_soft_popint_seed{s}", "pred_noisepen": ".tmps/runs/obs_t8000_pred_noisepen_seed{s}",
                 "pred_level": ".tmps/runs/obs_t8000_pred_level_seed{s}", "seeds": range(5)},
    },
}
VCOL = ["#8c8b86", "#2a78d6", "#eb6834", "#1baf7a"]
METRICS = [("recon_over_ceiling", "reconstruction R² / ceiling"), ("cyc_best", "cyclic: best single |corr|"),
           ("cha_best", "chaotic: best single |corr|"), ("cyc_dec", "cyclic: decode R² (all K)"),
           ("cha_dec", "chaotic: decode R² (all K)")]


def fig_summary(R: dict, variants: list, out: str):
    fig, axes = plt.subplots(len(R), len(METRICS) + 2, figsize=(3.0 * (len(METRICS) + 2), 3.3 * len(R)),
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
        for ax, key, ttl in [(axes[g, -2], "population", "spectral labeller"),
                             (axes[g, -1], "pred_population", "PREDICTABILITY labeller (D-035)")]:
            _pop_bars(ax, rows, variants, key, f"{grp}: population, {ttl}")
    fig.suptitle("Eye variants, validation [8000,10000) (dots = seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eye_compare_summary.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _pop_bars(ax, rows, variants, key, title):
    """Stacked mean family population per variant, under one labeller (`key` = population | pred_population)."""
    for v, var in enumerate(variants):
        rows_v = [r for r in rows if r["variant"] == var and key in r]
        if not rows_v:
            continue
        bottom = 0.0
        for f, col in FAM_COL.items():
            h = np.mean([r[key].get(f, 0) for r in rows_v])
            ax.bar(v, h, 0.65, bottom=bottom, color=col, edgecolor="white", linewidth=2, label=f if v == 0 else None)
            if h > 0.4:
                ax.text(v, bottom + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=7, color="white")
            bottom += h
    ax.set_xticks(range(len(variants)), variants, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper right")


def fig_q(run: str, labels: list, title: str, path: str):
    """Final q per channel (stacked) + q over training per channel (one small panel each)."""
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    q, qs, steps = art["q"], art["q_snap"], art["snap_steps"]
    K = len(q)
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.3, 1, 1], hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(K)
    for j, (f, col) in list(enumerate(FAM_COL.items()))[: q.shape[1]]:   # 3 (spectral) or 4 (predictability)
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
        for j, col in list(enumerate(FAM_COL.values()))[: q.shape[1]]:
            a.plot(steps, qs[:, i, j], color=col, lw=1.5)
        a.set_ylim(-0.03, 1.03)
        a.set_title(f"ch {i}", fontsize=8)
        a.tick_params(labelsize=6)
        if i % 8:
            a.set_yticklabels([])
    fig.suptitle(title + " — q over training steps (green stat, violet cyc, pink chao, grey noise)", fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_pop_time(groups: dict, variants: list, out: str):
    """Soft population over training: mean q per family vs step, one line per seed, one panel per soft variant.

    WHY: with the population pull removed (D-034) the question is whether the families stay populated or drift
    into one family, and WHEN the split is decided. The final bars alone cannot show a slow drift.
    Dashed lines = the 1/3/6 reference (K=16 -> 2/5/9 channels, as a share of K).
    """
    panels = []                                              # (group, variant, [(seed, steps, qbar[t,3])])
    for grp, spec in groups.items():
        for var in variants:
            curves = []
            for s in spec["seeds"]:
                path = os.path.join(spec[var].format(s=s), "artifacts.npz")
                if not os.path.exists(path):
                    continue
                art = np.load(path, allow_pickle=True)
                if "q_snap" in art.files:                    # fixed-ladder runs have no q: skipped
                    curves.append((s, art["snap_steps"], art["q_snap"].mean(axis=1)))
            if curves:
                panels.append((grp, var, curves))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.6), squeeze=False)
    for ax, (grp, var, curves) in zip(axes[0], panels):
        for s, steps, qbar in curves:
            for j, col in list(enumerate(FAM_COL.values()))[: qbar.shape[1]]:
                ax.plot(steps, qbar[:, j], color=col, lw=1.2, alpha=0.75)
        for j, (f, col) in enumerate(FAM_COL.items()):      # reference share 1/3/6 rounded like the ladder (K=16)
            if j < 3:                                        # noise has no reference share (D-035)
                ax.axhline([2, 5, 9][j] / 16, color=col, ls="--", lw=0.8)
            ax.plot([], [], color=col, label=f)
        ax.set_ylim(0, 1)
        ax.set_xlabel("training step")
        ax.set_ylabel("mean q over channels (share of K)")
        ax.set_title(f"{grp} {var}: population over training", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Soft family population vs training step (lines = seeds, dashed = 2/5/9 of 16)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pop_over_time.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_pred_plane(R: dict, variants: list, out: str, p_cut: float = 0.5):
    """Every eye's channels on the predictability plane (g, p_long), one panel per variant, coloured by the
    predictability label. Dashed = the cuts p 0.5 / g 0.1 (D-035). Shows whether training moved channels to the
    family corners or left them on the mixture continuum (F-30 item 5)."""
    grp_rows = [(g, r) for g, rows in R.items() for r in rows]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.8 * len(variants), 4.4), squeeze=False, sharey=True)
    for ax, var in zip(axes[0], variants):
        for f, col in FAM_COL.items():
            pts = [(gg, p) for _, r in grp_rows if r["variant"] == var
                   for gg, p, lab in zip(r["g"], r["p_long"], r["pred_labels"]) if lab == f]
            if pts:
                ax.scatter(*zip(*pts), s=30, color=col, edgecolor="white", lw=1.2, label=f"{f} ({len(pts)})")
        ax.axhline(p_cut, color="#52514e", ls="--", lw=0.8)       # the labeller's cyclic cut used for this comparison
        ax.axvline(0.1, color="#52514e", ls="--", lw=0.8)
        ax.set_xlabel("g (nonlinear gain, leads 1-22)")
        ax.set_title(var, fontsize=10, loc="left")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0, 0].set_ylabel("p_long = 1 - e_long(64)")
    fig.suptitle("Trained channels on the predictability plane, all seeds (colour = predictability label)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pred_plane.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="soft_roles", choices=list(PRESETS))
    ap.add_argument("--out", default=".tmps/eye_compare")
    ap.add_argument("--p_cut", type=float, default=None,
                    help="cyclic cut of the predictability labeller (default: config). ONE labeller reads all eyes")
    a = ap.parse_args()
    pc = OmegaConf.load("config/config.yaml").train.spectral.pred
    if a.p_cut is not None:
        pc.p_cut = a.p_cut
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
                # D-035: every eye is ALSO read by the predictability labeller (full series, same torch scores as the
                # loss), so spectral-trained and predictability-trained eyes are compared under both readouts
                plab, pd = label_predictability(art["S"], pc=pc)
                res["pred_labels"], res["p_long"], res["g"] = plab, pd["p_long"].tolist(), pd["g"].tolist()
                res["pred_population"] = {f: plab.count(f) for f in PRED_FAMILY_NAMES}
                if "q" in art.files:
                    fam_q = [ROLE2FAM[r] for r in art["roles"]]
                    nf = art["q"].shape[1]
                    ref = res["labels"] if nf == 3 else plab          # the labeller that speaks the eye's measure
                    res["q"] = art["q"].tolist()
                    res["agreement"] = float(np.mean([x == y for x, y in zip(fam_q, ref)]))
                    res["q_population"] = {f: fam_q.count(f) for f in FAM_COL}
                    # final l_pop = KL(2/5/9[/0] of 16 || mean q): how far the population lands from the 1/3/6
                    # reference (an objective when lambda_pop > 0, a readout otherwise); noise has no share
                    pi = np.array([2, 5, 9, 0][:nf]) / 16
                    qbar = np.clip(art["q"].mean(0), 1e-8, None)
                    res["l_pop"] = float((pi[pi > 0] * (np.log(pi[pi > 0]) - np.log(qbar[pi > 0]))).sum())
                    if s == 0:
                        fig_q(run, ref, f"{grp} seed 0 ({var})",
                              os.path.join(a.out, f"q_{grp.replace('=', '')}_{var}_seed0.png"))
                R[grp].append(res)
                extra = (f" | argmax-q pop {res['q_population']} agree {res['agreement']:.2f} l_pop {res['l_pop']:.3f}"
                         if "q" in res else "")
                extra += f" | stat r {res['stat_r']:.3f} n(r>=.98) {res['n_r98']} | pred pop {res['pred_population']}"
                print(f"{grp} seed {s} {var:>6}: pop {res['population']} R2/C {res['recon_over_ceiling']:.3f} "
                      f"cyc best/dec {res['cyc_best']:.2f}/{res['cyc_dec']:.2f} cha best/dec "
                      f"{res['cha_best']:.2f}/{res['cha_dec']:.2f}{extra}", flush=True)
    fig_summary(R, variants, a.out)
    fig_pop_time(groups, variants, a.out)
    fig_pred_plane(R, variants, a.out, float(pc.p_cut))
    with open(os.path.join(a.out, "eye_compare.json"), "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    main()
