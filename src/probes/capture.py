"""What does each channel actually CAPTURE? Mixture of truth modes, or genuinely unexplained? (2026-11-02)

Question (user, after the tight-cyclic test). Many channels of the predictability eye are labelled `noise`. The user's
reading: noise is not a mixed signal -- a noise-labelled channel should be decomposable later (deeper eyes, W1), and
only what remains after that is true noise. But the label is "no long-lead predictability AND no short-range
determinism", which an UNRESOLVED MIXTURE of cycles + chaos can also produce. This probe tells the two apart.

How. A channel s_k = <mask_k, field> is LINEAR in the field, and the field is SSH = sum_j amp_j phi_j and
SST = an AR1 low-pass of the same amps (rev2, `data.sst_tau`), plus observation noise. So regressing each channel on
the truth amplitudes AND their SST responses explains everything except the observation noise:
  explained share split by family (cyclic part / chaotic part)  and  unexplained = 1 - R^2.
A mixture = large explained share spread over several modes; true noise = large unexplained share.
Fitted on the training slice [0, t_fit), scored on validation [t_fit, T) -- out of sample.

Writes capture_<run>.png per run (heatmap |corr| channel x mode, stacked shares, example traces) + a printed table.
Run:  conda run -n oceanai python -m src.probes.capture --runs .tmps/runs/obs_t8000_pred_tight_seed0 ...
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import _ar1_response, _gaussian_blur  # noqa: E402
from src.probes.eye_compare import FAM_COL  # noqa: E402
from src.probes.family import label_predictability  # noqa: E402

UNEXPL = "#d6d5cf"        # unexplained share (light neutral: "nothing the truth can account for")
FAM_ORDER = ("cyclic", "chaotic", "noise", "stationary")


def capture(run: str) -> dict:
    """Per-channel capture numbers for one run dir."""
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra", "config.yaml"))
    S, A, fam = art["S"].astype(np.float64), art["truth_amp"].astype(np.float64), list(art["families"])
    t_fit, T = int(art["t_fit"]), S.shape[0]
    keep = [j for j in range(len(fam)) if A[j].std() > 1e-9]          # the constant mode lives in the level only
    tau = float(cfg.data.sst_tau)
    # regressors: raw amp (SSH) + AR1 response (SST) per mode; weights/private modes do not matter for a regression
    X = np.concatenate([np.stack([A[j] for j in keep], 1),
                        np.stack([_ar1_response(A[j], tau) for j in keep], 1)], axis=1)       # [T, 2*nm]
    owner = keep + keep                                               # regressor column -> mode index
    tr, va = slice(0, t_fit), slice(t_fit, T)
    X1 = np.hstack([np.ones((T, 1)), X])
    beta = np.linalg.lstsq(X1[tr], S[tr], rcond=None)[0]              # [1+2nm, K]
    Sv, Xv = S[va] - S[tr].mean(0), X[va]
    var = Sv.var(0)
    parts = {j: Xv[:, [c for c, o in enumerate(owner) if o == j]] @ beta[1:][[c for c, o in enumerate(owner) if o == j]]
             for j in keep}                                           # per-mode fitted contribution [Tv, K]
    fit = X1[va] @ beta
    grp = {f: sum(parts[j] for j in keep if fam[j] == f) for f in ("cyclic", "chaotic")}
    # Does a channel match a mode's SST view (AR1 low-passed) better than its raw SSH view? A variance split of the
    # fit cannot say (the raw and low-passed versions of a sinusoid are nearly collinear and their parts cancel), so
    # compare the best |corr| against each view directly.
    nm = len(keep)
    corr_sst = np.array([[abs(np.corrcoef(S[va, k], X[va, nm + j])[0, 1]) for j in range(nm)]
                         for k in range(S.shape[1])])
    # runs older than D-035 have no `pred` block: read them with the repo's current predictability settings
    cfg_pc = cfg.train.spectral.get("pred") or OmegaConf.load("config/config.yaml").train.spectral.pred
    labels, pd = label_predictability(S, pc=cfg_pc)
    corr = np.array([[abs(np.corrcoef(S[va, k], A[j, va])[0, 1]) for j in keep] for k in range(S.shape[1])])
    # D-037 readouts: does a channel carry the STATIC structure? level share = mean^2/(mean^2+var) of the raw channel;
    # static overlap = corr(mask, stationary pattern) over cells, SSH layer vs phi_0 and SST layer vs its blurred copy
    # (the generator blurs shared patterns for SST). Sign: which lobe of the pattern the mask sits on.
    Sva = S[va]
    level_share = Sva.mean(0) ** 2 / (Sva.mean(0) ** 2 + Sva.var(0) + 1e-30)
    i_stat = [j for j in range(len(fam)) if fam[j] == "stationary"]
    static_ov = np.full(S.shape[1], np.nan)
    if i_stat:
        phi0 = art["truth_phi"][i_stat[0]].astype(np.float64)
        ref = np.concatenate([phi0.ravel(), _gaussian_blur(phi0, float(cfg.data.sst_blur_sigma)).ravel()])
        M = art["masks"].astype(np.float64).reshape(S.shape[1], -1)
        static_ov = np.array([np.corrcoef(M[k], ref)[0, 1] for k in range(S.shape[1])])
    # the EYE's own final soft vote (what training decided), shown next to the labeller so the two can be compared;
    # older/fixed-ladder runs have no q
    ROLE2FAM = {"slow": "stationary", "cyclic": "cyclic", "fast": "chaotic", "noise": "noise"}
    if "q" in art.files:
        q = art["q"]
        q_fam = [ROLE2FAM[str(r)] for r in art["roles"]]
        q_max, q_stat = q.max(1), q[:, 0]
    else:   # family-free eye (layer 1, D-038): no vote to show, so the predictability labeller names the family
        q_fam, q_max, q_stat = list(labels), np.full(S.shape[1], np.nan), np.full(S.shape[1], np.nan)
    return dict(level_share=level_share, static_ov=static_ov, q_fam=q_fam, q_max=q_max, q_stat=q_stat,
        S=S, va=va, keep=keep, fam=fam, labels=labels, p_long=pd["p_long"], g=pd["g"], corr=corr,
        share_cyc=grp["cyclic"].var(0) / var, share_cha=grp["chaotic"].var(0) / var,
        unexpl=(S[va] - fit).var(0) / var, corr_sst=corr_sst,
        mode_share=np.stack([parts[j].var(0) / var for j in keep], 1),            # [K, nm]
        grp=grp, fit=fit, periods=list(cfg.data.cyclic_periods), cut=float(cfg_pc.p_cut))


def figure(c: dict, title: str, path: str):
    K = len(c["labels"])
    order = sorted(range(K), key=lambda k: (FAM_ORDER.index(c["labels"][k]), -c["p_long"][k]))
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 0.55, 1], width_ratios=[1, 1.25], hspace=0.45, wspace=0.18)

    # 1. heatmap |corr| channel x mode -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(c["corr"][order], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    cyc = [j for j in c["keep"] if c["fam"][j] == "cyclic"]
    xt = [f"cyc P{c['periods'][i % len(c['periods'])]:.0f}" for i in range(len(cyc))] + \
         [f"chao {i}" for i in range(len(c["keep"]) - len(cyc))]
    ax.set_xticks(range(len(xt)), xt, rotation=45, ha="right", fontsize=8)
    # row label = labeller verdict, the eye's own soft vote (q of its argmax family), and the two predictability scores
    vote = [f" (q {c['q_fam'][k][:4]} {c['q_max'][k]:.2f})" if np.isfinite(c["q_max"][k]) else "" for k in range(K)]
    ax.set_yticks(range(K), [f"ch{k} {c['labels'][k][:4]}{vote[k]}  p{c['p_long'][k]:.2f} g{c['g'][k]:+.2f}"
                             for k in order], fontsize=7)
    for t, k in zip(ax.get_yticklabels(), order):
        t.set_color(FAM_COL[c["labels"][k]] if c["labels"][k] != "noise" else "#52514e")
    ax.axvline(len(cyc) - 0.5, color="#0b0b0b", lw=1)
    for i, k in enumerate(order):
        for j in range(len(xt)):
            v = c["corr"][k, j]
            if v >= 0.3:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "#0b0b0b")
    ax.set_title("|corr| of each channel with each truth mode (validation); rows grouped by predictability label",
                 fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

    # 2. stacked shares of the WHOLE signal: level + fluctuation -------------------------------------
    # WHY total power (level^2 + var), not variance: a stationary eye IS a constant level with a small wiggle. A
    # variance-only split hides the level and drew such eyes as "100 % chaotic" (user, 2026-11-02).
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(K)
    lv = c["level_share"][order]
    fl = 1.0 - lv                                              # fluctuation share of total power
    sc, sh, un = (c[k][order] * fl for k in ("share_cyc", "share_cha", "unexpl"))
    ax.bar(x, lv, 0.72, color=FAM_COL["stationary"], edgecolor="white", lw=1.5, label="level (mean²)")
    ax.bar(x, sc, 0.72, bottom=lv, color=FAM_COL["cyclic"], edgecolor="white", lw=1.5, label="cyclic modes")
    ax.bar(x, sh, 0.72, bottom=lv + sc, color=FAM_COL["chaotic"], edgecolor="white", lw=1.5, label="chaotic modes")
    ax.bar(x, un, 0.72, bottom=lv + sc + sh, color=UNEXPL, edgecolor="white", lw=1.5, label="unexplained")
    nmode = (c["mode_share"][order] >= 0.10).sum(1)
    for i, k in enumerate(order):
        ax.text(i, min(lv[i] + sc[i] + sh[i] + un[i], 1.12) + 0.02, f"{nmode[i]}", ha="center", fontsize=7,
                color="#52514e")
    ax.set_xticks(x, [f"ch{k}\n{c['labels'][k][:4]}" for k in order], fontsize=7)
    for t, k in zip(ax.get_xticklabels(), order):
        t.set_color(FAM_COL[c["labels"][k]] if c["labels"][k] != "noise" else "#52514e")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("share of channel power, level² + variance (validation)")
    ax.set_title("what each channel's signal is made of: level + fluctuation (number on top = truth modes "
                 "carrying ≥ 10 % of the fluctuation)", fontsize=9, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left")
    ax.grid(alpha=0.2, axis="y")

    # 3. example traces ----------------------------------------------------------------------------
    picks = []
    for lab, n in (("cyclic", 1), ("chaotic", 1), ("noise", 2)):
        picks += [k for k in order if c["labels"][k] == lab][:n]
    picks = picks[:4] if len(picks) >= 4 else (picks + [k for k in order if k not in picks])[:4]
    # 2b. does each channel carry the STATIC structure? (D-037) ------------------------------------
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(K)
    ax.bar(x, np.abs(c["static_ov"][order]), 0.6, color="#8c8b86", edgecolor="white", lw=1.2)
    ax.set_xticks(x, [f"ch{k} {c['labels'][k][:4]}" for k in order], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("|corr|", fontsize=8)
    ax.set_title("where each eye looks: |corr(mask, static pattern)| over cells (same channel order as above)",
                 fontsize=9, loc="left")
    ax.grid(alpha=0.2, axis="y")

    sub = gs[2, :].subgridspec(1, 4, wspace=0.15)
    w = slice(0, 400)
    for i, k in enumerate(picks):
        ax = fig.add_subplot(sub[0, i])
        s = c["S"][c["va"], k]
        mu, sd = s.mean(), s.std() + 1e-12
        t = np.arange(400)
        ax.plot(t, ((s - mu) / sd)[w], color="#0b0b0b", lw=1.6, label="channel")
        ax.plot(t, (c["grp"]["cyclic"][:, k] / sd)[w], color=FAM_COL["cyclic"], lw=1.2, label="cyclic part")
        ax.plot(t, (c["grp"]["chaotic"][:, k] / sd)[w], color=FAM_COL["chaotic"], lw=1.2, label="chaotic part")
        ax.plot(t, ((s - c["fit"][:, k]) / sd)[w], color="#8c8b86", lw=1.0, label="residual")
        ax.set_title(f"ch{k} labelled {c['labels'][k]}: cyc {c['share_cyc'][k]:.0%} chao {c['share_cha'][k]:.0%} "
                     f"unexpl {c['unexpl'][k]:.0%}", fontsize=8, loc="left")
        ax.set_xlabel("validation step")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.set_ylabel("channel anomaly (z units)")
            ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[".tmps/runs/obs_t8000_pred_tight_seed0",
                                                   ".tmps/runs/obs_t8000_pred_tight_seed2"])
    ap.add_argument("--out", default=".tmps/capture_w2_tight")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        c = capture(run)
        name = os.path.basename(run.rstrip("/"))
        print(f"\n=== {name} (labeller p_cut {c['cut']}) ===")
        print(f"{'ch':>3} {'label':>10} {'p_long':>6} {'g':>6} | {'cyc':>5} {'chao':>5} {'unexpl':>6} | "
              f"modes>=10%  best|corr| SSH / SST | level  static_ov  q_stat")
        for k in range(len(c["labels"])):
            ms = c["mode_share"][k]
            print(f"{k:>3} {c['labels'][k]:>10} {c['p_long'][k]:6.2f} {c['g'][k]:+6.2f} | {c['share_cyc'][k]:5.2f} "
                  f"{c['share_cha'][k]:5.2f} {c['unexpl'][k]:6.2f} | {int((ms >= 0.10).sum()):>10}  "
                  f"{c['corr'][k].max():.2f} / {c['corr_sst'][k].max():.2f} | {c['level_share'][k]:5.2f}  "
                  f"{c['static_ov'][k]:+.2f}  {c['q_stat'][k]:6.2f}")
        figure(c, f"Signal capture, {name}: what each channel is made of (truth regression, validation)",
               os.path.join(a.out, f"capture_{name}.png"))


if __name__ == "__main__":
    main()
