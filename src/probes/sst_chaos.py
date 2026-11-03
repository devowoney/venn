"""W2 step 2c: why does the chaos score g miss chaos seen through SST, and which g fixes it? (calibration only)

Problem (F-31 item 4, D-036 limit 3; layer 2: chaotic children labelled noise). SST is an AR1 low-pass (tau 20) of
the SSH forcing. Through it, 4 of the 6 truth Lorenz modes drop from g 0.38-0.40 (SSH view) to 0.04-0.08, under the
0.1 cut, so a clean single chaotic mode is called noise. F-30 calibrated g on the SSH views only.

WHY the current g fails (two reasons, both about SCALE, not about the dynamics):
  1. g is an ABSOLUTE gain e_lin - e_nl. A smooth signal is almost perfectly linearly predictable at short leads, so
     e_lin(h) is tiny there and the gain cannot exceed it, however deterministic the signal is.
  2. The random-feature kernel width is the median embedded distance, set by the large smooth part of the signal.
     The fine directions of the embedding (the local slope / curvature, where the deterministic forcing shows) are
     far below that width, so the features cannot resolve them.
Chaos stays chaos through a linear filter (it is still a deterministic function of the Lorenz state); only the
measuring stick is off.

Candidates, all with the same windows / leads / ridge as D-035 (only g changes; p_long is untouched):
  g0     current:  mean_{h<=22} (e_lin - e_nl)
  g_rel  RELATIVE gain 1 - sum_h e_nl / sum_h e_lin: the share of the linear error the nonlinear ridge removes
         (fixes reason 1; ratio of sums, so a near-zero e_lin at h=1 cannot blow it up)
  g_wh   WHITENED embedding: the delay vectors are PCA-whitened (on the fit rows) before the random features, so
         every direction gets the same kernel scale (fixes reason 2; the linear ridge is invariant to it)
  g_lag  embedding LAG adapted to the channel's own memory: s = max(2, round(LAG_PER_TAU * tau_e)), tau_e = first
         lag where the autocorrelation falls below 1/e; LAG_PER_TAU set so the raw Lorenz modes keep s = 2
  g_rw   both: relative gain on the whitened embedding
Controls MUST stay below the cut: same-spectrum surrogates, OU and white noise, each also through the SAME AR1
low-pass (a filtered Gaussian process is still Gaussian and linear: no variant may find structure in it).

Signals: truth modes of generator seeds 0-2 (T=10000) in their SSH view and SST view (+0.05 noise), the controls, and
every live child of the layer-2 runs `obs_t8000_l2_seed*` with its best |corr| against each mode's SSH and SST view.
Writes sst_chaos.json + g_variants.png + children.png into --out (default .tmps/sst_chaos/).
Run:  python -m src.probes.sst_chaos
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

from src.data.synthetic import GenConfig, _ar1_response, generate_field  # noqa: E402
from src.probes.predictability import NOISE, phase_surrogate  # noqa: E402
from src.train.predictability import _embed, _ridge_err, make_rff  # noqa: E402

VARIANTS = ("g0", "g_rel", "g_wh", "g_lag", "g_rw")
CHAOS_CLASSES = ("chaotic SSH", "chaotic SST", "chaotic SST+noise")
NOISE_CLASSES = ("surrogate", "surrogate SST", "surrogate SST+noise", "OU", "OU SST", "white", "white SST")
# colours: chaos in the project's chaotic magenta family, noise controls in warm neutrals (dataviz reference slots)
CLS_COL = {"chaotic SSH": "#e87ba4", "chaotic SST": "#b8336a", "chaotic SST+noise": "#7a1f47",
           "surrogate": "#eda100", "surrogate SST": "#c98a00", "surrogate SST+noise": "#8f6200",
           "OU": "#eb6834", "OU SST": "#b84a1c", "white": "#008300", "white SST": "#005c00"}
CHILD_COL = {"chaotic SST view": "#b8336a", "chaotic SSH view": "#e87ba4", "cyclic view": "#4a3aa7",
             "mixture": "#8a8984"}


def tau_e(x: torch.Tensor, max_lag: int = 300) -> torch.Tensor:
    """First lag where the autocorrelation drops below 1/e, per channel. x [K,T] standardized."""
    X = torch.fft.rfft(x, n=2 * x.shape[1])
    ac = torch.fft.irfft(X * X.conj())[:, :max_lag]
    ac = ac / ac[:, :1]
    below = ac < np.exp(-1)
    return torch.where(below.any(1), below.float().argmax(1), torch.full_like(below[:, 0], max_lag, dtype=torch.long))


def g_scores(S: np.ndarray, pc, rff: dict, lag_per_tau: float) -> dict:
    """All g variants for series S [T,K], same rows / halves / ridge as `src.train.predictability.pred_scores`."""
    m_s, s_s, m_l, s_l = int(pc.m_short), int(pc.s_short), int(pc.m_long), int(pc.s_long)
    h_max, h_g = int(pc.h_long), int(pc.h_gain)
    x = torch.as_tensor(S.T, dtype=torch.float64)
    T, half = x.shape[1], x.shape[1] // 2
    x = (x - x[:, :half].mean(1, keepdim=True)) / (x[:, :half].std(1, keepdim=True) + 1e-12)
    te_ = tau_e(x[:, :half])
    lags = torch.clamp(torch.round(lag_per_tau * te_.double()).long(), min=s_s)
    # rows start late enough for the longest adapted window, so every variant scores the same rows
    t0 = max((m_l - 1) * s_l, (m_s - 1) * int(lags.max()))
    t = torch.arange(t0, T - h_max)
    fit, te = t < half - h_max, t >= half
    Y = torch.stack([x[:, t + h] for h in range(1, h_g + 1)], dim=2)

    def errs(E, Y, whiten):
        F_lin = torch.cat([torch.ones(E.shape[0], E.shape[1], 1, dtype=E.dtype), E], dim=2)
        Z = E
        if whiten:   # PCA-whiten on the fit rows; floor the spectrum so a noise-free smooth signal stays finite
            mu = E[:, fit].mean(1, keepdim=True)
            C = (E[:, fit] - mu).transpose(1, 2) @ (E[:, fit] - mu) / int(fit.sum())
            lam, V = torch.linalg.eigh(C)
            lam = lam.clamp_min(1e-6 * lam[:, -1:])
            Z = (E - mu) @ (V / lam.sqrt()[:, None, :])
        sub = Z[:, fit][:, :: max(1, int(fit.sum()) // 400)]
        ell = torch.cdist(sub, sub).flatten(1).median(dim=1).values.clamp_min(1e-6)
        F_nl = torch.cat([F_lin, (2.0 / rff["W"].shape[1]) ** 0.5 * torch.cos((Z @ rff["W"]) / ell[:, None, None]
                                                                              + rff["b"])], dim=2)
        return _ridge_err(F_lin, Y, fit, te, float(pc.lam_lin)), _ridge_err(F_nl, Y, fit, te, float(pc.lam_nl))

    e_lin, e_nl = errs(_embed(x, m_s, s_s, t), Y, whiten=False)
    e_linw, e_nlw = errs(_embed(x, m_s, s_s, t), Y, whiten=True)
    g_lag = torch.empty(x.shape[0], dtype=torch.float64)
    for s in lags.unique():                                   # channels grouped by their adapted lag
        k = lags == s
        el, en = errs(_embed(x[k], m_s, int(s), t), Y[k], whiten=False)
        g_lag[k] = (el - en).mean(1)
    return dict(g0=(e_lin - e_nl).mean(1).numpy(), g_rel=(1 - e_nl.sum(1) / e_lin.sum(1)).numpy(),
                g_wh=(e_linw - e_nlw).mean(1).numpy(), g_lag=g_lag.numpy(),
                g_rw=(1 - e_nlw.sum(1) / e_linw.sum(1)).numpy(), tau_e=te_.numpy(), lag=lags.numpy(),
                e1_lin=e_lin[:, 0].numpy())


def calibration_signals(cfg_data, seed: int, T: int, tau: float):
    """(class, name, series): truth modes in both views + low-passed and raw controls."""
    gc = GenConfig(**{**OmegaConf.to_container(cfg_data, resolve=True), "T": T})
    _, truth = generate_field(gc, seed=seed)
    rng = np.random.default_rng(2000 + seed)
    out = []
    for i, m in enumerate(truth["modes"]):
        if m["family"] != "chaotic":
            continue
        a = m["amp"].astype(np.float64)
        a = (a - a.mean()) / a.std()
        sst = _ar1_response(a, tau)
        out += [("chaotic SSH", f"s{seed} mode{i}", a), ("chaotic SST", f"s{seed} mode{i}", sst),
                ("chaotic SST+noise", f"s{seed} mode{i}", sst + NOISE * rng.standard_normal(T))]
        sur = phase_surrogate(a, rng)
        sur_sst = _ar1_response(sur, tau)
        out += [("surrogate", f"s{seed} sur{i}", sur), ("surrogate SST", f"s{seed} sur{i}", sur_sst),
                ("surrogate SST+noise", f"s{seed} sur{i}", sur_sst + NOISE * rng.standard_normal(T))]
    for r in range(2):
        ou = np.zeros(T)
        for t in range(1, T):
            ou[t] = ou[t - 1] * (1 - 1 / 5) + np.sqrt(2 / 5) * rng.standard_normal()
        w = rng.standard_normal(T)
        out += [("OU", f"s{seed} OU{r}", ou), ("OU SST", f"s{seed} OU{r}", _ar1_response(ou, tau)),
                ("white", f"s{seed} white{r}", w), ("white SST", f"s{seed} white{r}", _ar1_response(w, tau))]
    return out


def child_signals(pattern: str, seeds, tau: float):
    """Every live layer-2 child, with what it matches: a single chaotic mode's SST / SSH view, a cycle, or a mixture."""
    rows = []
    for s in seeds:
        path = os.path.join(pattern.format(s=s), "artifacts.npz")
        if not os.path.exists(path):
            print(f"  missing {path}")
            continue
        art = np.load(path, allow_pickle=True)
        S, A, fam = art["S"].astype(np.float64), art["truth_amp"].astype(np.float64), list(art["families"])
        t_fit = int(art["t_fit"])
        live = [k for k in range(S.shape[1]) if art["kind"][k] == "child" and not art["faded"][k]]
        keep = [j for j in range(len(fam)) if A[j].std() > 1e-9]
        views = {(j, "SSH"): A[j] for j in keep} | {(j, "SST"): _ar1_response(A[j], tau) for j in keep}
        for k in live:
            c = {v: abs(np.corrcoef(S[t_fit:, k], x[t_fit:])[0, 1]) for v, x in views.items()}
            (j, view), best = max(c.items(), key=lambda kv: kv[1])
            if best < 0.95:
                kind = "mixture"
            elif fam[j] == "chaotic":
                kind = f"chaotic {view} view"
            else:
                kind = "cyclic view" if fam[j] == "cyclic" else "mixture"
            rows.append(dict(cls="child", kind=kind, name=f"s{s} ch{k}", role=str(art["roles"][k]),
                             best_corr=float(best), best_mode=f"{fam[j]}{j} {view}", x=S[:t_fit, k]))
    return rows


def calibrate_lag(pc, rff, cfg_data, T, tau):
    """LAG_PER_TAU such that the raw (SSH) Lorenz modes keep the F-30 lag s = 2: s_short / median tau_e(SSH)."""
    xs = [x for s in (0, 1, 2) for c, _, x in calibration_signals(cfg_data, s, T, tau) if c == "chaotic SSH"]
    x = torch.as_tensor(np.stack(xs), dtype=torch.float64)
    x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
    te_ = tau_e(x[:, : T // 2]).double()
    return float(pc.s_short) / float(te_.median()), te_.numpy()


def margin(rows, v):
    """Separation on the test signals: min over chaos classes minus max over noise classes (> 0 = a cut exists)."""
    lo = min(r[v] for r in rows if r["cls"] in CHAOS_CLASSES)
    hi = max(r[v] for r in rows if r["cls"] in NOISE_CLASSES)
    return lo, hi


def fig_variants(rows, children, cuts, out):
    """One panel per variant: g per signal class (dots) and per child kind; dashed = the candidate cut."""
    classes = list(CHAOS_CLASSES) + list(NOISE_CLASSES)
    kinds = [k for k in CHILD_COL if any(c["kind"] == k for c in children)]
    ylab = classes + [f"child: {k}" for k in kinds]
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 0.38 * len(ylab) + 1.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, v in zip(axes, VARIANTS):
        for y, c in enumerate(classes):
            vals = [r[v] for r in rows if r["cls"] == c]
            ax.scatter(vals, y + 0.12 * rng.standard_normal(len(vals)), s=14, color=CLS_COL[c], alpha=0.85, lw=0)
        for y, k in enumerate(kinds, start=len(classes)):
            cs = [c for c in children if c["kind"] == k]
            for c in cs:   # ring = the child is labelled noise by training (the case this step is about)
                ax.scatter(c[v], y + 0.12 * rng.standard_normal(), s=22 if c["role"] == "noise" else 12,
                           facecolor="none" if c["role"] == "noise" else CHILD_COL[k], edgecolor=CHILD_COL[k],
                           lw=1.1)
        ax.axvline(cuts[v], color="#52514e", ls="--", lw=1)
        ax.axhline(len(classes) - 0.5, color="#d6d5cf", lw=0.8)
        ax.axhline(len(CHAOS_CLASSES) - 0.5, color="#d6d5cf", lw=0.8)
        lo, hi = margin(rows, v)
        ax.set_title(f"{v}\nchaos min {lo:+.2f} | noise max {hi:+.2f}", fontsize=10)
        ax.grid(axis="x", color="#ecebe6", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(ylab)), ylab, fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Chaos score variants: test signals (top) and layer-2 children (bottom; ring = trained label 'noise')."
                 " Dashed = cut.", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "g_variants.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--children", default=".tmps/runs/obs_t8000_l2_seed{s}")
    ap.add_argument("--child_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=".tmps/sst_chaos")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = OmegaConf.load("config/config.yaml")
    pc, tau = cfg.train.spectral.pred, float(cfg.data.sst_tau)
    rff = make_rff(int(pc.m_short), int(pc.n_rff), seed=0)
    lag_per_tau, te_ssh = calibrate_lag(pc, rff, cfg.data, a.T, tau)
    print(f"tau_e of the SSH Lorenz modes: {np.sort(te_ssh)} -> LAG_PER_TAU {lag_per_tau:.3f}")

    sigs = [s for seed in a.seeds for s in calibration_signals(cfg.data, seed, a.T, tau)]
    G = g_scores(np.stack([x for _, _, x in sigs], 1), pc, rff, lag_per_tau)
    rows = [dict(cls=c, name=n, **{k: float(G[k][i]) for k in G}) for i, (c, n, _) in enumerate(sigs)]
    children = child_signals(a.children, a.child_seeds, tau)
    # children are scored on the TRAINING slice, as layer 2 does (each run its own length; same scores per child)
    for c in children:
        Gc = g_scores(c.pop("x")[:, None], pc, rff, lag_per_tau)
        c.update({k: float(Gc[k][0]) for k in Gc})

    # candidate cut per variant: midway in the gap if there is one, else the current 0.1 (g0 keeps the D-035 cut)
    cuts = {}
    for v in VARIANTS:
        lo, hi = margin(rows, v)
        cuts[v] = 0.1 if v == "g0" else (0.5 * (lo + hi) if lo > hi else 0.1)

    print("\n=== test signals: median [min, max] per class ===")
    print(f"{'class':>20} {'n':>3} " + " ".join(f"{v:>22}" for v in VARIANTS) + "   tau_e  lag")
    for c in list(CHAOS_CLASSES) + list(NOISE_CLASSES):
        rs = [r for r in rows if r["cls"] == c]
        print(f"{c:>20} {len(rs):>3} " + " ".join(
            f"{np.median([r[v] for r in rs]):+6.2f} [{min(r[v] for r in rs):+5.2f},{max(r[v] for r in rs):+5.2f}]"
            for v in VARIANTS) + f"  {np.median([r['tau_e'] for r in rs]):5.0f} {np.median([r['lag'] for r in rs]):4.0f}")
    print("\n=== separation and cut ===")
    for v in VARIANTS:
        lo, hi = margin(rows, v)
        ok = sum((r[v] >= cuts[v]) == (r["cls"] in CHAOS_CLASSES) for r in rows)
        print(f"{v:>6}: chaos min {lo:+.3f}  noise max {hi:+.3f}  gap {lo - hi:+.3f}  cut {cuts[v]:.3f}  "
              f"correct {ok}/{len(rows)}")
    print("\n=== layer-2 children: how many cross the cut, per kind (trained role in brackets) ===")
    for k in CHILD_COL:
        cs = [c for c in children if c["kind"] == k]
        if not cs:
            continue
        roles = {r: sum(c["role"] == r for c in cs) for r in ("slow", "cyclic", "fast", "noise")}
        print(f"  {k:>17} n={len(cs):>2} roles {roles} | above cut: " +
              " ".join(f"{v} {sum(c[v] >= cuts[v] for c in cs)}" for v in VARIANTS))
    print("\n  noise-labelled children that are a clean chaotic view:")
    for c in children:
        if c["role"] == "noise" and c["kind"].startswith("chaotic"):
            print(f"    {c['name']:>9} {c['best_mode']:>15} |corr| {c['best_corr']:.3f} tau_e {c['tau_e']:3.0f} | "
                  + " ".join(f"{v} {c[v]:+.2f}" for v in VARIANTS))

    fig_variants(rows, children, cuts, a.out)
    with open(os.path.join(a.out, "sst_chaos.json"), "w") as f:
        json.dump(dict(lag_per_tau=lag_per_tau, cuts=cuts, signals=rows, children=children), f, indent=1)


if __name__ == "__main__":
    main()
