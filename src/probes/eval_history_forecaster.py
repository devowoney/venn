"""Streaming evaluation of the history forecaster on the VALIDATION set (SOP 04).

For each trained forecaster run: one causal pass over the whole record gives, at every t, the forecast the
streaming emulator would make (history initialized on the training set [0, t_tr), grown by appending the
observed states). Only launches whose targets all lie in the validation set [t_tr, T) are scored.

Answers "how much, and how long can the prefrontal cortex predict?":
  - latent channels: skill vs persistence, corr, amplitude per lead; horizon (first lead with corr < 0.5);
    grouped by channel family, labelled on the TRAINING half only;
  - hidden modes (answer key): ridge readout latent -> mode amplitudes fitted on the training set, applied to the
    forecast; ceiling = same readout on the TRUE future latent state;
  - causality check: perturbing the future must not change any earlier forecast.

Run:  python -m src.probes.eval_history_forecaster --runs .tmps/runs_hf/final_seed0 ... --out .tmps/eval_hf/final
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.models.history_forecaster import HistoryForecaster   # noqa: E402
from src.probes.family import amp_ratio, label_family, series_stats   # noqa: E402

FAMS = ("stationary", "cyclic", "chaotic")
COL = {"stationary": "#7a7a7a", "cyclic": "#1f77b4", "chaotic": "#d62728"}


def load_run(run: str, device: str):
    """Rebuild the frozen forecaster + its train-only normalization + the eye-lobe artifacts it was fitted on."""
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    c = ck["cfg"]
    model = HistoryForecaster(ck["K"], leads=c["leads"], d=c["d"], layers=c["layers"], heads=c["heads"],
                              dropout=c["dropout"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    nm = np.load(os.path.join(run, "norm.npz"))
    art = np.load(os.path.join(ck["encoder_run"], "artifacts.npz"))
    return model, nm, art, int(nm["t_tr"])


@torch.no_grad()
def causality_check(model, z: torch.Tensor, t_cut: int) -> float:
    """Max |change| of the forecasts made before t_cut when every state from t_cut on is replaced by noise."""
    a = model(z[None, : t_cut + 200])[0, :t_cut]
    zp = z[: t_cut + 200].clone()
    zp[t_cut:] = torch.randn_like(zp[t_cut:]) * 5.0
    b = model(zp[None])[0, :t_cut]
    return float((a - b).abs().max())


def channel_families(S: np.ndarray, t_tr: int) -> list[str]:
    """Family label of each latent channel, decided on the TRAINING half only (never on validation)."""
    out = []
    for i in range(S.shape[1]):
        x = S[:t_tr, i]
        out.append(label_family(series_stats((x - x.mean()) / (x.std() + 1e-12)), amp_ratio=amp_ratio(x)))
    return out


def lead_scores(pred: np.ndarray, truth: np.ndarray, pers: np.ndarray):
    """pred/truth/pers [N,A,K] over N launches -> skill, corr, amplitude, each [A,K]."""
    mse = ((pred - truth) ** 2).mean(0)
    mse_p = ((pers - truth) ** 2).mean(0)
    skill = 1 - mse / np.maximum(mse_p, 1e-12)
    pc, tc = pred - pred.mean(0), truth - truth.mean(0)
    corr = (pc * tc).mean(0) / (pc.std(0) * tc.std(0) + 1e-12)
    amp = pred.std(0) / (truth.std(0) + 1e-12)
    return skill, corr, amp


def horizon(corr_ak: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """First lead (1-based) at which corr drops below thr, per channel; A+1 = never within the leads."""
    A = corr_ak.shape[0]
    below = corr_ak < thr
    return np.where(below.any(0), below.argmax(0) + 1, A + 1)


def eval_one(run: str, device: str) -> dict:
    model, nm, art, t_tr = load_run(run, device)
    S = art["S"].astype(np.float64)
    T, K = S.shape
    A = model.A
    z = (S - nm["mu"]) / nm["sd"]
    zt = torch.tensor(z, dtype=torch.float32, device=device)

    with torch.no_grad():
        zhat = model(zt[None, : T - 1])[0].cpu().numpy()          # [T-1,A,K]: forecast made at t from z_0..z_t
    caus = causality_check(model, zt, t_tr + 100)

    # launches: every t whose targets t+1..t+A all fall inside the validation set
    L = np.arange(t_tr - 1, T - A)
    idx = L[:, None] + np.arange(1, A + 1)[None, :]                # [N,A] target times
    pred, truth = zhat[L], z[idx]                                   # [N,A,K]
    # PERSISTENCE starts from each forecast's own initial condition: the last observed state z_t at the launch
    # time t, held constant over every lead ("nothing changes from now on"). Its error grows with the lead.
    pers = np.repeat(z[L][:, None, :], A, axis=1)
    skill, corr, amp = lead_scores(pred, truth, pers)
    # RMSE in standardized units (1 = one training-set standard deviation of that channel), per lead and channel.
    # CLIMATOLOGY = always forecast the training mean (0 in standardized units): the "knows nothing" reference.
    rmse = np.sqrt(((pred - truth) ** 2).mean(0))                   # [A,K]
    rmse_p = np.sqrt(((pers - truth) ** 2).mean(0))
    rmse_c = np.sqrt((truth ** 2).mean(0))
    half = len(L) // 2                                              # does a LONGER history help or hurt?
    sk_early = lead_scores(pred[:half], truth[:half], pers[:half])[0]
    sk_late = lead_scores(pred[half:], truth[half:], pers[half:])[0]

    fams = channel_families(S, t_tr)

    # --- hidden modes: can the forecast be read back into the generator's modes? -------------------
    amp_true = art["truth_amp"].astype(np.float64)                  # [n_modes,T]
    mode_fam = [str(f) for f in art["families"]]
    moving = amp_true.std(1) > 1e-9                                 # the constant mode has nothing to forecast
    Y = amp_true[moving].T                                          # [T,m]
    X = np.concatenate([np.ones((T, 1)), z], 1)                     # [T,K+1]
    Xtr, Ytr = X[:t_tr], Y[:t_tr]
    Wr = np.linalg.solve(Xtr.T @ Xtr + 1e-3 * t_tr * np.eye(K + 1), Xtr.T @ Ytr)   # ridge, TRAINING set only
    rd = lambda zz: np.concatenate([np.ones(zz.shape[:-1] + (1,)), zz], -1) @ Wr
    m_pred, m_ceil, m_true = rd(pred), rd(truth), Y[idx]           # [N,A,m]
    mc_pred = lead_scores(m_pred, m_true, m_true)[1]                # corr [A,m]
    mc_ceil = lead_scores(m_ceil, m_true, m_true)[1]

    return dict(run=run, t_tr=t_tr, T=T, n_launch=len(L), causality_maxdiff=caus, fams=fams,
                skill=skill, corr=corr, amp=amp, rmse=rmse, rmse_p=rmse_p, rmse_c=rmse_c, sk_early=sk_early, sk_late=sk_late, hor=horizon(corr),
                mode_fam=[f for f, m in zip(mode_fam, moving) if m], mode_corr=mc_pred, mode_ceil=mc_ceil,
                z=z, zhat=zhat)


def fam_mean(r: dict, key: str, fam: str | None) -> np.ndarray:
    """[A] mean over the channels of one family (all channels if fam is None)."""
    sel = [i for i, f in enumerate(r["fams"]) if fam is None or f == fam]
    return r[key][:, sel].mean(1) if sel else np.full(r[key].shape[0], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = [eval_one(r, args.device) for r in args.runs]
    A = R[0]["skill"].shape[0]
    leads = np.arange(1, A + 1)
    show = [1, 2, 4, 8, 16, 32, 64]

    # ------------------------------ text summary ------------------------------------------------
    lines = [f"history forecaster, validation [{R[0]['t_tr']},{R[0]['T']}), {R[0]['n_launch']} launches/seed, "
             f"{len(R)} seeds", ""]
    for r in R:
        pop = {f: r["fams"].count(f) for f in FAMS}
        lines.append(f"{os.path.basename(r['run'])}: causality max|diff|={r['causality_maxdiff']:.2e}  "
                     f"channels {pop['stationary']}/{pop['cyclic']}/{pop['chaotic']}")
    lines.append("")
    lines += ["RMSE (standardized units: 1 = one training std of the channel). Lower is better.",
              "  model       = history forecaster",
              "  persistence = z_{t+a} := z_t, the forecast's own initial condition held constant",
              "  climatology = z_{t+a} := training mean (0)", ""]
    lines.append(f"{'RMSE':<22}" + "".join(f"  h{a:<5}" for a in show))
    for fam in (None,) + FAMS:
        for key, who in [("rmse", "model"), ("rmse_p", "persistence"), ("rmse_c", "climatology")]:
            v = np.array([fam_mean(r, key, fam) for r in R])
            lines.append(f"  {(fam or 'all') + ' ' + who:<20}" + "".join(f"  {np.nanmean(v[:, a - 1]):.3f}" for a in show))
    lines.append("")
    for key, name in [("skill", "skill vs persistence"), ("corr", "corr"), ("amp", "amplitude")]:
        lines.append(f"{name:<22}" + "".join(f"  h{a:<5}" for a in show))
        for fam in (None,) + FAMS:
            v = np.array([fam_mean(r, key, fam) for r in R])       # [seeds,A]
            lab = fam or "all"
            lines.append(f"  {lab:<20}" + "".join(f" {np.nanmean(v[:, a - 1]):+.3f}" for a in show))
        lines.append("")
    hz = np.concatenate([r["hor"] for r in R])
    lines.append(f"horizon (first lead with corr<0.5, {A + 1} = never): mean {hz.mean():.1f}, per family: "
                 + ", ".join(f"{f} {np.mean([h for r in R for h, ff in zip(r['hor'], r['fams']) if ff == f]):.1f}"
                             for f in FAMS if any(f in r["fams"] for r in R)))
    e = np.array([r["sk_early"].mean(1) for r in R]).mean(0)
    l_ = np.array([r["sk_late"].mean(1) for r in R]).mean(0)
    lines.append("history length: skill early/late half of validation  "
                 + "  ".join(f"h{a} {e[a - 1]:+.3f}/{l_[a - 1]:+.3f}" for a in (1, 16, 64)))
    lines.append("")
    lines.append("hidden modes: forecast corr (ceiling = readout of the TRUE future latent)")
    for fam in ("cyclic", "chaotic"):
        mp = np.array([r["mode_corr"][:, [i for i, f in enumerate(r["mode_fam"]) if f == fam]].mean(1) for r in R])
        mc = np.array([r["mode_ceil"][:, [i for i, f in enumerate(r["mode_fam"]) if f == fam]].mean(1) for r in R])
        lines.append(f"  {fam:<9}" + "".join(f" h{a}:{mp[:, a - 1].mean():.2f}/{mc[:, a - 1].mean():.2f}"
                                          for a in show))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(args.out, "summary.txt"), "w").write(txt + "\n")
    json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k not in ("z", "zhat")}
               for r in R], open(os.path.join(args.out, "results.json"), "w"))

    # ------------------------------ figure ------------------------------------------------------
    # Read top to bottom: (1) how big is the error, against the two "no-skill" references;
    # (2) what individual forecasts look like; (3) phase (corr), amplitude, and the hidden modes.
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.25)
    for j, fam in enumerate(FAMS):
        ax = fig.add_subplot(gs[0, j])
        for key, lab, sty in [("rmse", "history forecaster", dict(color=COL[fam], lw=2.2)),
                              ("rmse_p", "persistence (initial condition held)", dict(color="k", lw=1.4, ls="--")),
                              ("rmse_c", "climatology (training mean)", dict(color="0.55", lw=1.4, ls=":"))]:
            v = np.array([fam_mean(r, key, fam) for r in R])
            if np.isnan(v).all():
                continue
            ax.plot(leads, np.nanmean(v, 0), label=lab, **sty)
            if key == "rmse":
                ax.fill_between(leads, np.nanmin(v, 0), np.nanmax(v, 0), color=COL[fam], alpha=0.18)
        n = sum(r["fams"].count(fam) for r in R)
        ax.set_title(f"RMSE — {fam} channels (n={n}, all seeds)")
        ax.set_xlabel("lead (steps ahead)")
        ax.set_ylabel("RMSE (training std units)")
        ax.set_xlim(1, A)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper left")
    # example forecasts, seed 0: three launches, each with its own initial condition
    r0 = R[0]
    t_a = r0["t_tr"] + 300
    for j, fam in enumerate(FAMS):
        ax = fig.add_subplot(gs[1, j])
        ch = [i for i, f in enumerate(r0["fams"]) if f == fam]
        if not ch:
            ax.set_visible(False)
            continue
        k = ch[0]
        ts = np.arange(t_a - 32, t_a + 3 * 80 + 16)
        ax.plot(ts, r0["z"][ts, k], color="k", lw=1.1, label="truth")
        for n_, t0 in enumerate(range(t_a, t_a + 3 * 80, 80)):
            tt = np.arange(t0 + 1, t0 + A + 1)
            ax.plot(tt, r0["zhat"][t0, :, k], color=COL[fam], lw=2, label="forecast" if n_ == 0 else None)
            ax.plot(tt, np.full(A, r0["z"][t0, k]), color="k", ls="--", lw=1,
                    label="persistence" if n_ == 0 else None)
            ax.plot([t0], [r0["z"][t0, k]], "o", color="k", ms=5,
                    label="initial condition (launch)" if n_ == 0 else None)
        ax.set_title(f"seed 0, channel {k} ({fam}): three {A}-step forecasts", fontsize=10)
        ax.set_xlabel("time step (validation)")
        ax.set_ylabel("z (training std units)")
        ax.legend(fontsize=7, loc="best")
    for j, (key, name, ref) in enumerate([("corr", "correlation with truth", 0.5),
                                          ("amp", "amplitude = forecast std / true std", 1.0)]):
        ax = fig.add_subplot(gs[2, j])
        for fam in FAMS:
            v = np.array([fam_mean(r, key, fam) for r in R])
            if np.isnan(v).all():
                continue
            ax.plot(leads, np.nanmean(v, 0), color=COL[fam], label=fam, lw=2)
            ax.fill_between(leads, np.nanmin(v, 0), np.nanmax(v, 0), color=COL[fam], alpha=0.15)
        ax.axhline(ref, color="k", lw=0.8, ls=":")
        ax.set_xlim(1, A)
        ax.set_xlabel("lead (steps ahead)")
        ax.set_title(name)
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[2, 2])
    for fam in ("cyclic", "chaotic"):
        sel = lambda r: [i for i, f in enumerate(r["mode_fam"]) if f == fam]
        v = np.array([r["mode_corr"][:, sel(r)].mean(1) for r in R])
        c = np.array([r["mode_ceil"][:, sel(r)].mean(1) for r in R])
        ax.plot(leads, v.mean(0), color=COL[fam], lw=2, label=f"{fam} modes: forecast")
        ax.plot(leads, c.mean(0), color=COL[fam], lw=1, ls="--", label=f"{fam} modes: perfect-forecast ceiling")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xlim(1, A)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lead (steps ahead)")
    ax.set_title("hidden generator modes read from the forecast (corr)")
    ax.legend(fontsize=7)
    fig.suptitle(f"History forecaster — validation set [{r0['t_tr']},{r0['T']}), never seen in training; "
                 f"{len(R)} seeds x {r0['n_launch']} forecasts (band = seed range)", fontsize=13)
    out = os.path.join(args.out, "hf_eval.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
