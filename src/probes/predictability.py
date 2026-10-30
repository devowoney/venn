"""W2 step 2a: calibrate a PREDICTABILITY SCORE for the eye's chaos test, before it goes anywhere near training.

Question (structure_review.md W2). The eye calls a channel cyclic or chaotic by SPECTRAL SHAPE (`structure_share`). That
cannot tell chaos from coloured noise, and it calls a quasi-periodic signal chaotic. Chaos is a property of the DYNAMICS:
deterministic at short range, unpredictable past ~one Lyapunov time (~22 steps on this testbed, F-23). So score each
channel by how well its OWN past predicts its future, at leads h = 1..64:

    e(h) = test MSE of a regression  x[t+h] ~ f(delay embedding of x at t)  /  var(x)        (0 = perfect, 1 = useless)

This is an OBSERVER-side measuring stick (one channel, its own past, used only for a family label). It is not module 2.

Two versions of f, both closed-form ridge regressions (so either can later sit inside the loss like `l_recon`):
  e_lin  f linear in the embedding. WHY it is not enough on its own: a linear predictor's error is fixed by the power
         spectrum (Wiener-Kolmogorov), so Lorenz and a noise with the SAME spectrum give the same e_lin. It does fix the
         quasi-periodic case (a sum of tones is linearly predictable).
  e_nl   f = linear + random Fourier features of the embedding (kernel ridge). Only a nonlinear f can exploit
         determinism, so chaos shows up as a NONLINEAR GAIN g = e_lin - e_nl at short leads that surrogates do not have.

Readouts per signal: d = 1 - e(1) (determinism), g = mean_{h<=22}(e_lin - e_nl) (nonlinear gain), both on a SHORT
window, and p_long = 1 - e_long(64), a linear score on a LONG window (long-lead predictability; see M_LONG for why).
Cross-checks (post-hoc only, options B/C/D of the review): Gottwald-Melbourne 0-1 test K, Rosenstein largest Lyapunov
exponent, and a surrogate rank test on g (19 phase-randomized surrogates: same spectrum, no determinism).

Signals: the generator's truth modes (seeds 0-2, T=10000), same-spectrum surrogates of the Lorenz modes, OU (tau 5,
the chaotic modes' memory), a 3-tone quasi-periodic signal, white noise, truth modes + 0.05 noise, and the trained
channels of the pop5 eye (`obs_t8000_soft_popint_seed*`) with their current labeller verdict.

Writes predictability.json + curves.png / planes.png / checks.png into --out (default .tmps/predictability/).
Run:  conda run -n oceanai python -m src.probes.predictability
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from src.data.synthetic import GenConfig, generate_field  # noqa: E402
from src.probes.family import amp_ratio, label_family, series_stats  # noqa: E402

H_MAX = 64          # longest lead scored (= the history forecaster's horizon)
H_LYAP = 22         # ~one Lyapunov time of the testbed's Lorenz modes, in steps (F-23)
M_EMB, S_EMB = 8, 2  # SHORT delay embedding (span 14) for e_lin / e_nl / g: the chaotic memory is tau_e 3-7 steps
M_LONG, S_LONG = 16, 8  # LONG embedding (span 120) for the linear long-lead score p_long. WHY two windows (sweep
#                      2026-10-30, seed 0): span 14 leaves a 3-tone quasi-periodic signal at p_lin(64) 0.33 (too
#                      little past to pin three long periods), span 120 gives 1.00; but at span 120 the random
#                      features no longer find the chaotic determinism (g +0.37 at span 14 -> -0.03).
N_RFF = 512         # random Fourier features of the nonlinear score
NOISE = 0.05        # observation noise of the testbed, in units of a signal's std
N_SUR = 19          # surrogates for the rank test (p >= 0.05 resolution)

# one colour per signal CLASS, fixed order (dataviz reference slots); truth families keep the project's colours
CLASSES = {
    "cyclic":            "#4a3aa7",   # truth sinusoids (slot 7 violet = project cyclic)
    "cyclic+noise":      "#2a78d6",
    "quasi-periodic":    "#1baf7a",
    "chaotic":           "#e87ba4",   # truth Lorenz coords (slot 5 magenta = project chaotic)
    "chaotic+noise":     "#e34948",
    "surrogate":         "#eda100",   # same spectrum as a Lorenz mode, no determinism
    "OU":                "#eb6834",
    "white":             "#008300",
}
TRAINED_MARK = {"cyclic": "o", "chaotic": "^", "stationary": "s"}   # trained channels: grey, shape = labeller verdict


# ----------------------------------------------------------------------------------------------- the score itself
def _embed(x: np.ndarray, m: int, s: int, t0: int):
    """Delay embedding rows E[t] = x[t], x[t-s], ..., x[t-(m-1)s] and targets Y[t, h-1] = x[t+h], for t >= t0."""
    t = np.arange(t0, len(x) - H_MAX)
    E = np.stack([x[t - j * s] for j in range(m)], axis=1)
    Y = np.stack([x[t + h] for h in range(1, H_MAX + 1)], axis=1)
    return t, E, Y


def _ridge(F_fit, Y_fit, F_te, lam):
    """Closed-form ridge, all leads at once (one solve, H_MAX right-hand sides)."""
    G = F_fit.T @ F_fit
    G[np.diag_indices_from(G)] += lam * len(F_fit)
    return F_te @ np.linalg.solve(G, F_fit.T @ Y_fit)


def pred_curves(x: np.ndarray, rng: np.random.Generator, long: bool = False):
    """e_lin(h), e_nl(h) [, e_long(h)] for h = 1..H_MAX. Fit on the first half of the record, score on the second.

    WHY out-of-sample: 512 features fitted in-sample would "predict" white noise. The halves are separated by H_MAX
    steps so no fit target leaks into the test window.
    """
    x = np.asarray(x, dtype=np.float64)
    half = len(x) // 2
    x = (x - x[:half].mean()) / (x[:half].std() + 1e-12)
    t0 = (M_LONG - 1) * S_LONG                         # same rows for both windows, so the scores compare
    t, E, Y = _embed(x, M_EMB, S_EMB, t0)
    fit, te = t < half - H_MAX, t >= half
    one = np.ones((len(E), 1))
    F_lin = np.hstack([one, E])
    # random Fourier features approximate a Gaussian kernel of width ell = median distance between embedded points
    sub = E[fit][rng.choice(fit.sum(), 500, replace=False)]
    ell = np.median(np.linalg.norm(sub[:, None] - sub[None], axis=-1)) + 1e-12
    W = rng.standard_normal((M_EMB, N_RFF)) / ell
    b = rng.uniform(0, 2 * np.pi, N_RFF)
    F_nl = np.hstack([F_lin, np.sqrt(2.0 / N_RFF) * np.cos(E @ W + b)])
    var = Y[te].var(axis=0) + 1e-12
    e_lin = ((_ridge(F_lin[fit], Y[fit], F_lin[te], 1e-6) - Y[te]) ** 2).mean(0) / var
    e_nl = ((_ridge(F_nl[fit], Y[fit], F_nl[te], 1e-4) - Y[te]) ** 2).mean(0) / var
    if not long:
        return e_lin, e_nl
    _, EL, _ = _embed(x, M_LONG, S_LONG, t0)
    F_long = np.hstack([one, EL])
    e_long = ((_ridge(F_long[fit], Y[fit], F_long[te], 1e-6) - Y[te]) ** 2).mean(0) / var
    return e_lin, e_nl, e_long


def readouts(e_lin, e_nl, e_long=None) -> dict:
    r = dict(d_lin=1 - e_lin[0], d_nl=1 - e_nl[0], p_lin=1 - e_lin[-1], p_nl=1 - e_nl[-1],
             p22_nl=1 - e_nl[H_LYAP - 1], g=float(np.mean((e_lin - e_nl)[:H_LYAP])))
    if e_long is not None:
        r["p_long"] = 1 - e_long[-1]                  # long-window LINEAR predictability at lead 64
    return r


# ----------------------------------------------------------------------------------------------- cross-checks
def phase_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same power spectrum, random Fourier phases: a Gaussian linear process with x's spectrum (Theiler)."""
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x)) + x.mean()


def zero_one_test(x: np.ndarray, rng: np.random.Generator, stride: int = 4, n_c: int = 100) -> float:
    """Gottwald-Melbourne 0-1 test (correlation method, 2009). ~1 chaotic, ~0 regular. Noise ALSO gives ~1.

    The record is undersampled by `stride` first: an oversampled flow reads as regular to this test.
    """
    phi = np.asarray(x, dtype=np.float64)[::stride]
    phi = phi - phi.mean()
    N = len(phi)
    n_cut = N // 10
    j = np.arange(1, N + 1)
    n = np.arange(1, n_cut + 1)
    K = []
    for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, n_c):
        p, q = np.cumsum(phi * np.cos(j * c)), np.cumsum(phi * np.sin(j * c))
        M = np.array([np.mean((p[k:] - p[:-k]) ** 2 + (q[k:] - q[:-k]) ** 2) for k in n])
        D = M - phi.var() * (1 - np.cos(n * c)) / (1 - np.cos(c))   # remove the oscillatory term
        K.append(np.corrcoef(n, D)[0, 1])
    return float(np.median(K))


def rosenstein(x: np.ndarray, n_pts: int = 5000, theiler: int = 50, k_max: int = 40, fit=(1, 20)) -> float:
    """Largest Lyapunov exponent per step (Rosenstein 1993): slope of the mean log distance between nearest neighbours.

    Lorenz at the testbed's sampling (0.05 time units per step) should give ~0.9 * 0.05 = 0.045 / step.
    A regular signal gives ~0; noise gives a spurious positive slope too, which is why this is a cross-check only.
    """
    x = np.asarray(x[:n_pts], dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    t0 = (M_EMB - 1) * S_EMB      # short window: the neighbours must be close in the chaotic sense
    E = np.stack([x[t0 - j * S_EMB:len(x) - j * S_EMB] for j in range(M_EMB)], axis=1)
    n = len(E) - k_max
    _, idx = cKDTree(E[:n]).query(E[:n], k=2 * theiler + 3)
    i_all = np.arange(n)
    ok = np.abs(idx - i_all[:, None]) > theiler                 # skip temporal neighbours
    has = ok.any(axis=1)
    nn = idx[i_all, ok.argmax(axis=1)][has]
    i0 = i_all[has]
    curve = [np.mean(np.log(np.linalg.norm(E[i0 + k] - E[nn + k], axis=1) + 1e-12)) for k in range(k_max)]
    k = np.arange(fit[0], fit[1])
    return float(np.polyfit(k, np.array(curve)[k], 1)[0])


def surrogate_p(x: np.ndarray, g_real: float, rng: np.random.Generator) -> float:
    """Rank test: share of same-spectrum surrogates whose nonlinear gain g reaches the real one. Small p = determinism."""
    g_sur = [readouts(*pred_curves(phase_surrogate(x, rng), rng))["g"] for _ in range(N_SUR)]
    return float((1 + sum(gs >= g_real for gs in g_sur)) / (N_SUR + 1))


def labeller_verdict(x: np.ndarray) -> tuple[str, float]:
    """The CURRENT readout (flatness gate, then structure share) -- what W2 wants to replace."""
    st = series_stats(x)
    return label_family(st, amp_ratio=amp_ratio(x)), float(st["cyclic_share"])


# ----------------------------------------------------------------------------------------------- signal sets
def calibration_signals(cfg_data, seed: int, T: int):
    """Truth modes of one generator seed + test signals built around them. Yields (class, name, series)."""
    gc = GenConfig(**{**OmegaConf.to_container(cfg_data, resolve=True), "T": T})
    _, truth = generate_field(gc, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    out = []
    for i, m in enumerate(truth["modes"]):
        a = m["amp"].astype(np.float64)
        if m["family"] == "stationary":
            continue                                   # constant: the flatness gate decides it, W2 does not touch it
        out.append((m["family"], f"s{seed} mode{i}", a))
        out.append((m["family"] + "+noise", f"s{seed} mode{i}+noise", a + NOISE * rng.standard_normal(T)))
        if m["family"] == "chaotic":
            out.append(("surrogate", f"s{seed} sur(mode{i})", phase_surrogate(a, rng)))
    for r in range(2):
        ou = np.zeros(T)
        for t in range(1, T):                          # tau 5 = the chaotic modes' memory (tau_e 3-7)
            ou[t] = ou[t - 1] * (1 - 1 / 5) + np.sqrt(2 / 5) * rng.standard_normal()
        out.append(("OU", f"s{seed} OU{r}", ou))
        tt = np.arange(T)                              # incommensurate periods 60, 60*phi, 60*phi^2
        qp = sum(np.sin(2 * np.pi * tt / P + rng.uniform(0, 2 * np.pi)) for P in (60.0, 97.08, 157.08))
        out.append(("quasi-periodic", f"s{seed} QP{r}", qp + NOISE * rng.standard_normal(T)))
        out.append(("white", f"s{seed} white{r}", rng.standard_normal(T)))
    return out


def trained_signals(pattern: str, seeds):
    out = []
    for s in seeds:
        path = os.path.join(pattern.format(s=s), "artifacts.npz")
        if not os.path.exists(path):
            print(f"  missing {path}")
            continue
        S = np.load(path, allow_pickle=True)["S"].astype(np.float64)
        out += [(f"s{s} ch{k}", S[:, k]) for k in range(S.shape[1])]
    return out


# ----------------------------------------------------------------------------------------------- figures
def fig_curves(rows, out):
    """e(h) per class: dashed linear, solid nonlinear. One panel per class + one per labeller verdict of trained chans."""
    groups = [(c, [r for r in rows if r["cls"] == c], CLASSES[c]) for c in CLASSES]
    for lab in ("cyclic", "chaotic", "stationary"):
        groups.append((f"trained, labelled {lab}", [r for r in rows if r["cls"] == "trained" and r["label"] == lab],
                       "#52514e"))
    groups = [g for g in groups if g[1]]
    nc = 4
    nr = int(np.ceil(len(groups) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), sharex=True, sharey=True, squeeze=False)
    h = np.arange(1, H_MAX + 1)
    for ax, (title, rs, col) in zip(axes.flat, groups):
        for r in rs:
            ax.plot(h, r["e_lin"], color=col, lw=1.0, ls="--", alpha=0.55)
            ax.plot(h, r["e_nl"], color=col, lw=2.0, alpha=0.75)
            ax.plot(h, r["e_long"], color=col, lw=1.0, ls=":", alpha=0.9)
        ax.axvline(H_LYAP, color="#8c8b86", lw=0.8, ls=":")
        ax.set_title(f"{title} (n={len(rs)})", fontsize=9, loc="left")
        ax.set_ylim(-0.02, 1.25)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("lead h (steps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("e(h) = test MSE / var")
    fig.suptitle(f"Predictability curves. Span-{(M_EMB - 1) * S_EMB} window: dashed = linear e_lin, solid = nonlinear "
                 f"e_nl. Span-{(M_LONG - 1) * S_LONG} window: dotted = linear e_long. Grey line = 1 Lyapunov time "
                 f"({H_LYAP} steps)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curves.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_planes(rows, out):
    """Where each signal class lands: (determinism d_nl, long-lead p_nl) and (nonlinear gain g, long-lead p_lin)."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    panels = [("d_nl", "p_long", "d = 1 - e_nl(1)  (short-lead determinism)", "p_long = 1 - e_long(64)"),
              ("g", "p_long", "g = mean_{h<=22} (e_lin - e_nl)  (nonlinear gain)", "p_long = 1 - e_long(64)"),
              ("g", "cyc_share", "g (nonlinear gain)", "structure_share (CURRENT cyclic/chaotic test, cut 0.5)")]
    for ax, (xk, yk, xl, yl) in zip(axes, panels):
        tr = [r for r in rows if r["cls"] == "trained"]
        for lab, mk in TRAINED_MARK.items():
            rs = [r for r in tr if r["label"] == lab]
            if rs:
                ax.scatter([r[xk] for r in rs], [r[yk] for r in rs], marker=mk, s=40, facecolor="none",
                           edgecolor="#8c8b86", lw=1.2, label=f"trained, labelled {lab}")
        for c, col in CLASSES.items():
            rs = [r for r in rows if r["cls"] == c]
            ax.scatter([r[xk] for r in rs], [r[yk] for r in rs], s=46, color=col, edgecolor="white", lw=1.5, label=c)
        if yk == "cyc_share":
            ax.axhline(0.5, color="#52514e", lw=0.8, ls="--")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Signal classes on the predictability planes (truth seeds 0-2 + trained pop5 channels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "planes.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_checks(rows, out):
    """Cross-checks B/C/D per class, next to the gain g: do the independent chaos tests agree with the score?"""
    names = list(CLASSES) + ["trained"]
    keys = [("g", "nonlinear gain g"), ("K01", "0-1 test K (B)"), ("lyap", "Rosenstein λ / step (C)"),
            ("p_sur", "surrogate rank p on g (D)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 4.2))
    rng = np.random.default_rng(0)
    for ax, (k, title) in zip(axes, keys):
        for i, c in enumerate(names):
            v = [r[k] for r in rows if r["cls"] == c and r.get(k) is not None]
            if not v:
                continue
            ax.scatter(rng.uniform(-0.18, 0.18, len(v)) + i, v, s=14, color=CLASSES.get(c, "#8c8b86"))
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.5)
        if k == "lyap":
            ax.axhline(0.045, color="#52514e", lw=0.8, ls="--")
        if k == "p_sur":
            ax.axhline(0.05, color="#52514e", lw=0.8, ls="--")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Independent chaos checks per class (bar = median; dashed: Lorenz λ ≈ 0.045, p = 0.05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "checks.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=10000)
    ap.add_argument("--trained", default=".tmps/runs/obs_t8000_soft_popint_seed{s}")
    ap.add_argument("--trained_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no_checks", action="store_true", help="skip the slow cross-checks B/C/D")
    ap.add_argument("--out", default=".tmps/predictability")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = OmegaConf.load("config/config.yaml")
    rng = np.random.default_rng(0)

    sigs = []
    for s in a.seeds:
        sigs += calibration_signals(cfg.data, s, a.T)
    sigs += [("trained", n, x) for n, x in trained_signals(a.trained, a.trained_seeds)]

    rows = []
    for cls, name, x in sigs:
        e_lin, e_nl, e_long = pred_curves(x, rng, long=True)
        lab, cs = labeller_verdict(x)
        r = dict(cls=cls, name=name, label=lab, cyc_share=cs, e_lin=e_lin.tolist(), e_nl=e_nl.tolist(),
                 e_long=e_long.tolist(), **readouts(e_lin, e_nl, e_long))
        if not a.no_checks:
            r.update(K01=zero_one_test(x, rng), lyap=rosenstein(x), p_sur=surrogate_p(x, r["g"], rng))
        rows.append(r)
        print(f"{cls:>15} {name:>18}: label {lab:>10} cyc_share {cs:.2f} | d_lin {r['d_lin']:.3f} d_nl {r['d_nl']:.3f}"
              f" p_long {r['p_long']:.2f} g {r['g']:+.3f}"
              + ("" if a.no_checks else f" | K {r['K01']:.2f} λ {r['lyap']:.3f} p_sur {r['p_sur']:.2f}"), flush=True)

    print("\n=== medians per class ===")
    keys = ["d_lin", "d_nl", "p_lin", "p_long", "g", "cyc_share"] + ([] if a.no_checks else ["K01", "lyap", "p_sur"])
    print(f"{'class':>16} {'n':>3} " + " ".join(f"{k:>9}" for k in keys) + "  labeller verdicts")
    for c in list(CLASSES) + ["trained"]:
        rs = [r for r in rows if r["cls"] == c]
        if rs:
            labs = {lab: sum(r["label"] == lab for r in rs) for lab in ("stationary", "cyclic", "chaotic")}
            print(f"{c:>16} {len(rs):>3} " + " ".join(f"{np.median([r[k] for r in rs]):9.3f}" for k in keys)
                  + f"  {labs}")

    fig_curves(rows, a.out)
    fig_planes(rows, a.out)
    if not a.no_checks:
        fig_checks(rows, a.out)
    with open(os.path.join(a.out, "predictability.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
