"""Family labeller: one scalar series -> {stationary, cyclic, chaotic}.

Post-hoc and reversible (D-004): this labels the channels, it never gates training.

## What the three families MEAN (user, 2026-09-01 -- D-025; this superseded my first version)

  stationary = CONSTANT. A signal that does not move: flat in time, living in the field's time
               MEAN. Not "slow".
  cyclic     = it moves and it has memory: a clean oscillation, and ALSO a slow red drift, which
               the user ruled belongs here because it is not constant.
  chaotic    = it moves fast and broadband: short memory, no spectral line.

My first version of this module got the first family wrong. It defined stationary as "long memory,
no spectral line", i.e. slow red noise -- so a tau=200 OU drift counted as stationary while a truly
constant signal (were one present) would have been unlabellable. That is a different taxonomy from
the project's, and it produced the misleading population 10/1/5 for the baseline run. Flatness now
comes FIRST and is decided on amplitude, not on timescale.

## Why the old rule was broken too (finding F-11)

Before either version, `evaluate.py` called a series cyclic whenever its single largest FFT bin held
more than `peak_thr` of the power. A red spectrum piles its power in the LOWEST resolvable bin, so a
slow process was always reported as "a clean oscillation of period T/2" -- the labeller returned
`cyclic` even when handed the generator's own stationary amplitude series. Every
`population.stationary = 0` we ever measured was therefore unfalsifiable.

## The three questions, and what answers them

  1. does it MOVE?  `amp_ratio` = std/|mean| on the RAW series (see that function). Constant -> 0.
     This cannot be answered from a standardized series at all, which is why the flat test is
     skipped rather than guessed when the caller passes `amp_ratio=None`.
  2. is the power in a LINE or a HUMP?  `line_frac` = power within +-2 bins of the peak, over total.
     A sinusoid is one bin wide (0.94 on the testbed); a red slope (0.67) or a chaotic hump
     (0.10-0.46) is tens of bins wide. Width, not height, is the discriminator.
  3. how LONG is the memory?  `tau_e` = first lag whose autocorrelation drops below 1/e. The
     cyclic modes measure 12/27/57 and the chaotic ones 3-7, so the split at 10 is a margin.

Calibration on the ten hidden truth modes (seed 0) is checked by `evaluate.py` on every run and
reported as `labeller_truth_accuracy` -- if that is not 1.0, believe the probe over the model.
"""
from __future__ import annotations

import numpy as np

FAMILIES = ("stationary", "cyclic", "chaotic")

# defaults sit mid-gap in the truth calibration above; exposed so the probe can sweep them
LINE_THR = 0.80    # line_frac above this = power is in a spectral LINE -> a real oscillation
TAU_THR = 10       # tau_e (steps) above this = it has memory -> cyclic rather than chaotic.
#                    Truth margin: the cyclic modes measure 12/27/57, the chaotic ones 3-7.
KPK_MIN = 4        # need >=4 whole cycles in the window before calling something cyclic
FLAT_THR = 0.05    # amp_ratio below this = the series does not MOVE -> stationary (D-025). A truly
#                    constant mode measures 0.000; the next-flattest thing in the testbed is ~1.0.


def series_stats(x: np.ndarray, max_lag: int = 500, prom_win: int = 20) -> dict:
    """Descriptors of ONE series [T]. Standardizes internally, so input scale is irrelevant."""
    T = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)

    # --- spectrum. Bin 0 (the mean) is zeroed: it carries no dynamics and would dominate. ---
    p = np.abs(np.fft.rfft(x)) ** 2
    p[0] = 0.0
    k_pk = int(np.argmax(p))
    p_tot = p.sum() + 1e-30
    # LINE vs HUMP: a +-2-bin window around the peak. Width, not height, is the discriminator.
    line_frac = float(p[max(1, k_pk - 2):k_pk + 3].sum() / p_tot)
    # peak prominence over the LOCAL background (kept as a diagnostic; line_frac is the decider)
    lo, hi = max(1, k_pk - prom_win), min(len(p), k_pk + prom_win + 1)
    bg = np.concatenate([p[lo:k_pk], p[k_pk + 1:hi]])
    prom = float(p[k_pk] / (np.median(bg) + 1e-30)) if bg.size else float("inf")
    peak_frac = float(p[k_pk] / p_tot)                       # legacy descriptor, no longer decisive
    period = float(T / k_pk) if k_pk > 0 else float("inf")

    # --- autocorrelation (FFT, zero-padded to 2T so the wrap-around does not alias) ---
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * T)) ** 2)[:max_lag]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac < np.exp(-1.0))[0]
    tau_e = int(below[0]) if len(below) else max_lag
    gap1 = float(np.var(np.diff(x)))                         # = 2(1-rho1) at unit variance

    out = dict(gap1=gap1, rho1=float(1.0 - gap1 / 2.0), tau_e=tau_e, period=period,
               k_pk=k_pk, peak_frac=peak_frac, line_frac=line_frac, prominence=prom,
               ac_min=float(ac.min()))
    out.update(decompose(x))            # D-026 structure shares; `x` is standardized, which is fine
    #                                     here because all three shares are RATIOS of its power
    return out


def decompose(x: np.ndarray, kpk_min: int = KPK_MIN) -> dict:
    """Split ONE series into level / trend / oscillation / broadband residual (D-026).

    Classification must follow a channel's GLOBAL STRUCTURE, not its small fluctuation and drift
    (user, 2026-09-01). A channel that visibly oscillates while carrying fast noise and a wandering
    baseline is CYCLIC; the noise and the baseline are not what it is. Statistics read off the raw
    series answer the opposite question, because noise and drift dominate them -- which is how my
    earlier ACF-recurrence test called visibly oscillating channels non-cyclic.

    The split, all in the periodogram of the fluctuation (so the parts are additive in power):

      level    = mean(x)                    the static offset      -> handled by `amp_ratio`
      trend    = bins 1 .. kpk_min-1        fewer than kpk_min whole cycles in the record: at that
                                            resolution a drift and a "cycle" are indistinguishable,
                                            so this band IS the drift band
      osc      = the dominant peak's HALF-POWER band (bins >= kpk_min), widened to at least +-2 bins.
                 A half-power bandwidth adapts to the peak: a pure tone gives ~1 bin, a quasi-
                 periodic hump gives a wide band and still counts as oscillation. A fixed +-2 window
                 would have scored a broad hump as mostly residual, i.e. called it chaotic.
      residual = whatever is left in bins >= kpk_min: genuinely broadband

    Returns the variance SHARES of trend / osc / residual (summing to 1 over the fluctuation), plus
    `cyclic_share = trend + osc`. Trend counts toward cyclic per D-025: a drift is not constant, so
    it cannot be stationary, and it is not broadband, so it is not chaotic.

    Known limit: a non-sinusoidal periodic signal puts power in harmonics, which land in `residual`
    and bias it toward chaotic. Harmonic folding is a deferred refinement.
    """
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    p = np.abs(np.fft.rfft(xc)) ** 2
    p[0] = 0.0
    tot = p.sum()
    if tot <= 0:                                          # a perfectly constant series
        return dict(trend_share=0.0, osc_share=0.0, residual_share=0.0, cyclic_share=0.0,
                    osc_k=0, osc_lo=0, osc_hi=0)

    n = len(p)
    k0 = min(max(1, kpk_min), n - 1)
    trend = p[1:k0].sum()

    hi_band = p[k0:]
    if hi_band.size and hi_band.max() > 0:
        k_pk = int(np.argmax(hi_band)) + k0
        half = p[k_pk] / 2.0
        lo = k_pk                                          # grow left while above half power
        while lo - 1 >= k0 and p[lo - 1] >= half:
            lo -= 1
        hi = k_pk                                          # grow right while above half power
        while hi + 1 < n and p[hi + 1] >= half:
            hi += 1
        lo, hi = max(k0, min(lo, k_pk - 2)), min(n - 1, max(hi, k_pk + 2))    # at least +-2 bins
        osc = p[lo:hi + 1].sum()
    else:
        k_pk, lo, hi, osc = 0, 0, 0, 0.0

    residual = max(0.0, tot - trend - osc)
    return dict(trend_share=float(trend / tot), osc_share=float(osc / tot),
                residual_share=float(residual / tot),
                cyclic_share=float((trend + osc) / tot),
                osc_k=int(k_pk), osc_lo=int(lo), osc_hi=int(hi))


def label_family(st: dict, line_thr: float = LINE_THR, tau_thr: int = TAU_THR,
                 kpk_min: int = KPK_MIN, amp_ratio: float | None = None,
                 flat_thr: float = FLAT_THR) -> str:
    """Label one series from its `series_stats`. Order matters: FLATNESS is tested first.

    stationary -> FLAT. `amp_ratio` -- how much this series MOVES relative to the static level it
                  sits on -- is below `flat_thr`. The user's original definition, restored
                  2026-09-01 (D-025): a stationary signal is a CONSTANT one, not a slow one.
    cyclic     -> it moves, and its fluctuation is STRUCTURED: trend + oscillation outweigh the
                  broadband residual (D-026). Drift counts as structure, not as noise.
    chaotic    -> it moves, and its fluctuation is mostly broadband residual.

    NOTE on why the flat test is a strict THRESHOLD and not part of the dominance argmax. Adding the
    level as a fourth share and taking the argmax looks tidier, but `level^2 > var` is only
    `amp_ratio < 1` -- so a channel fluctuating at 50% of its level would come out "stationary".
    A real example: ch14 of run 20260901_080655 has amp_ratio 0.526 (level share 0.78) and is
    plainly broadband chaotic. The level therefore gates, and the dominance vote decides the shape
    of whatever is left moving.

    `amp_ratio=None` means the caller cannot measure flatness, and the flat test is then SKIPPED
    rather than guessed. This matters: a bare standardized series carries NO amplitude information
    (standardizing a constant divides by ~0 and amplifies its noise into garbage), so flatness is
    only decidable against an external reference -- the field's own variability for a channel, or
    the other modes' amplitudes for a truth series.
    """
    if amp_ratio is not None and amp_ratio < flat_thr:
        return "stationary"
    if "cyclic_share" in st:
        # D-026: dominance of STRUCTURE over the fluctuation. trend+osc vs broadband residual.
        return "cyclic" if st["cyclic_share"] >= st["residual_share"] else "chaotic"
    # fallback for callers that did not run `decompose` (kept so old artifacts still label)
    if st["line_frac"] > line_thr and st["k_pk"] >= kpk_min:
        return "cyclic"
    return "cyclic" if st["tau_e"] >= tau_thr else "chaotic"


def amp_ratio(x: np.ndarray) -> float:
    """How much does this series MOVE, relative to the static level it sits on?

    `std_t(x) / |mean_t(x)|` -- the coefficient of variation, on the RAW series (never standardized;
    standardizing destroys exactly the information this measures). One definition that works for
    both sides of the comparison:

      * a truth amplitude series: the constant mode gives std 0 / mean 1 -> **0.0**, while every
        other mode is `_standardize`-d to mean~0 / std 1 -> a huge ratio;
      * a learned channel `s_i = <mask_i, field>`: a channel sitting on a static pattern has a
        coherent level (the mask SUMS the offset, ~count_i) against incoherent fluctuation, so the
        ratio is small; a channel reading anomalies has a near-zero mean and so a large ratio.

    Threshold `FLAT_THR = 0.05`: nothing in this testbed lands between 0.0 and ~1.0, so the choice
    is a margin rather than a fit. The principled reading of 0.05 is "the fluctuation is at most a
    twentieth of the level" -- unambiguously a constant with a little noise on it.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(x.std() / (abs(x.mean()) + 1e-12))


def label_series(x: np.ndarray, **kw) -> str:
    """Convenience: descriptors + label in one call."""
    return label_family(series_stats(x), **kw)


def population(labels) -> dict:
    """Count labels into the family histogram (the thing compared against the 1/3/6 target)."""
    labels = list(labels)
    return {f: sum(l == f for l in labels) for f in FAMILIES}
