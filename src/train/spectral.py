"""Spectral band ladder for the encoder objective (D-024).

WHY (finding F-11/F-12). The v0 objective asks every channel for the SAME thing: minimize the mean
slowness `mean_i var(ds_i)/var(s_i)`. One shared optimum means the K channels compete for the same
globally-slowest content, so raising K only buys more near-duplicates of the dominant regime -- it
is not a lever on WHICH dynamical families appear. Measured on the K sweep 20260831_0736*, with a
labeller that can actually emit "stationary": population 3/0/1 (K=4), 4/2/2 (K=8), 12/0/4 (K=16),
22/3/7 (K=32), 36/10/18 (K=64). Slow channels dominate at every K and clean cyclic channels are
almost absent. More kernels does not redistribute the families.

THE FIX, by analogy with the spatial size ladder (D-020). The size ladder solved the same shape of
problem: left to a shared objective every mask shrank to a tiny patch, so we ASSIGNED each channel
a target footprint on a geometric ladder and penalized only outside a tolerance. Here we assign
each channel a target TIMESCALE BAND, and ask it to put its power there:

    L_band = mean_i relu(band_target - bandfrac_i)^2      bandfrac_i = in-band power / total power
    L_line = mean_i  { relu(line_target - linefrac_i)^2   if channel i is a CYCLIC rung
                     { relu(linefrac_i  - line_cap  )^2   otherwise (slow and fast rungs)

`L_band` places a channel in a frequency band; `L_line` says what SHAPE its spectrum should have
there. Both are needed because the families OVERLAP in timescale on this testbed -- the cyclic mode
of period 300 is slower than the stationary OU (tau=200), so frequency alone cannot separate them.
What separates them is line-vs-hump: `linefrac` (power within +-2 bins of the peak) is 0.94 for the
truth's cyclic modes and 0.67 for its stationary one (see `src/probes/family.py`). So a slow rung is
asked for low-frequency power WITHOUT a line, a cyclic rung for a line inside its own octave, and a
fast rung for high-frequency power without a line.

Properties that keep this compatible with the rest of the loss:
  * every quantity is a RATIO of powers -> invariant to mask scale in value AND gradient, so it
    cannot be gamed by shrinking a mask (the D-017 rule that three earlier collapses violated);
  * every term is a one-sided HINGE -> silent for a channel already in its band, so it guides
    instead of pinning (D-020's tolerance philosophy);
  * nothing here assumes exponential memory. An earlier draft matched a target lag-L
    autocorrelation `exp(-L/tau_i)`; that is wrong for a cyclic channel, whose ACF oscillates and
    whose long-lag gap can reach 4 (twice the exponential maximum), so it would have actively
    suppressed the cyclic family it was meant to create.

The labeller stays post-hoc and reversible (D-004): the ladder never asserts a channel's family, it
only shapes where in the spectrum the channel looks. The population readout still comes from
`src/probes/family.py`.
"""
from __future__ import annotations

import numpy as np
import torch

ROLES = ("slow", "cyclic", "fast")


def rung_roles(K: int, pop_target=(1, 3, 6)) -> list[str]:
    """Assign each channel a role, in the proportion of the target population (CLAUDE.md 1/3/6).

    Order is slow -> cyclic -> fast, which deliberately lines up with the size ladder (D-020,
    channel 0 = largest footprint): in this testbed and in the ocean, the slow modes are the
    large-scale ones, so the two ladders reinforce rather than fight each other.

    Rounding: the largest-remainder method, then any leftover channel goes to `fast` -- the
    curriculum's starting family (D-007 begins at 0/0/100), so a spare channel defaults to the
    least constrained role.
    """
    w = np.asarray(pop_target, dtype=float)
    w = w / w.sum()
    raw = w * K
    n = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - n)):                       # largest remainder first
        if n.sum() >= K:
            break
        n[j] += 1
    n[2] += K - int(n.sum())                               # any leftover -> fast
    return ["slow"] * int(n[0]) + ["cyclic"] * int(n[1]) + ["fast"] * int(n[2])


def band_plan(K: int, win_len: int, roles: list[str], slow_period_min: float,
              fast_period_max: float, cyclic_period_max: float | None = None,
              device="cpu") -> torch.Tensor:
    """Per-channel band indicator over the rFFT bins of a `win_len` window -> [K, n_bins] float.

    Bin k of a length-`win_len` window has period `win_len / k`, so a period range maps to a
    contiguous bin range. Bin 0 (the window mean) is always excluded: it carries no dynamics and
    would otherwise let a channel satisfy `L_band` with a constant.

      slow rung   -> periods >= slow_period_min                       (the low-frequency band)
      cyclic rung -> its own octave, geometric from cyclic_period_max down to fast_period_max, so
                     the n_cyc rungs tile the range instead of all chasing one period
      fast rung   -> periods <= fast_period_max                       (the high-frequency band)

    The cyclic range is DELIBERATELY allowed to overlap the slow band (`cyclic_period_max` defaults
    to `slow_period_min` but should usually be larger). On this testbed the cyclic modes have
    periods 60/140/300 while the stationary OU has tau=200: the families are interleaved in
    frequency, so a partition would make the two longest cycles unreachable by any cyclic rung.
    Overlapping bands are fine -- masks were never required to be disjoint (D-013) -- and it is
    `L_line`, not the band, that separates a slow LINE from a slow HUMP.

    Every band is widened to at least one bin, so no channel is handed an impossible target.
    """
    cyclic_period_max = slow_period_min if cyclic_period_max is None else cyclic_period_max
    n_bins = win_len // 2 + 1
    W = torch.zeros(K, n_bins, device=device)
    k_slow = max(1, int(np.floor(win_len / slow_period_min)))          # slow band = bins 1..k_slow
    k_fast = min(n_bins - 1, int(np.ceil(win_len / fast_period_max)))  # fast band = k_fast..end

    n_cyc = sum(r == "cyclic" for r in roles)
    # geometric octave edges across the mid range; n_cyc bands -> n_cyc+1 edges (long -> short)
    edges = np.geomspace(cyclic_period_max, fast_period_max, max(2, n_cyc + 1))
    c = 0
    for i, role in enumerate(roles):
        if role == "slow":
            lo, hi = 1, k_slow
        elif role == "fast":
            lo, hi = k_fast, n_bins - 1
        else:
            p_hi, p_lo = float(edges[c]), float(edges[c + 1])          # p_hi > p_lo (periods)
            lo = max(1, int(np.floor(win_len / p_hi)))
            hi = min(n_bins - 1, int(np.ceil(win_len / p_lo)))
            c += 1
        hi = max(hi, lo)                                               # never an empty band
        W[i, lo:hi + 1] = 1.0
    return W


def flat_ratio(S_all: torch.Tensor) -> torch.Tensor:
    """`r_i = |mean_t s_i| / (|mean_t s_i| + std_t s_i)` [K], differentiable: 1 = a constant, 0 = pure anomaly.

    The labeller's flatness test in differentiable form (`r = 1/(1+amp_ratio)`). Shared by `level_term`
    and the soft family assignment `family_q`, so both read flatness the same way.
    """
    level = S_all.mean(dim=0).abs()
    fluct = S_all.std(dim=0)
    return level / (level + fluct + 1e-12)


def log_amp(S_all: torch.Tensor) -> torch.Tensor:
    """`log(std_t s_i / |mean_t s_i|)` [K] = log of the labeller's `amp_ratio`, differentiable and scale-free.

    WHY a log (SOP 02 `flat_measure: log`, 2026-10-29). Flatness only becomes meaningful within a few percent of
    a constant, where r = 1/(1+amp_ratio) is squeezed into 0.95..1.0: there the squared r-hinge has ~1e-5 left
    and every trained eye -- fixed ladder AND soft -- stalled at r 0.95-0.97 although a mask can reach 0.995 on
    this field (measured). In log units each FACTOR of flatness counts the same, near the target or far from it.
    """
    level = S_all.mean(dim=0).abs()
    fluct = S_all.std(dim=0)
    return torch.log(fluct + 1e-12) - torch.log(level + 1e-12)


def flat_penalty(S_all: torch.Tensor, measure: str, flat_target: float, amp_target: float) -> torch.Tensor:
    """Per-channel flatness hinge [K] (unweighted). `ratio`: relu(flat_target - r)^2 (D-025, the historical one);
    `log`: relu(log amp_ratio - log amp_target)^2, i.e. ask the fluctuation to be at most `amp_target` x the level."""
    if measure == "log":
        return torch.relu(log_amp(S_all) - float(np.log(amp_target))) ** 2
    return torch.relu(flat_target - flat_ratio(S_all)) ** 2


def level_term(S_all: torch.Tensor, is_slow: torch.Tensor, flat_target: float, measure: str = "ratio",
               amp_target: float = 0.02):
    """FLATNESS hinge on the slow rungs: make the channel a static LEVEL, not a slow wiggle (D-025).

    `r_i = level_i / (level_i + fluct_i)`, with `level_i = |mean_t s_i|` and `fluct_i = std_t s_i`.
    r -> 1 for a perfectly constant channel, -> 0 for a pure anomaly channel. Penalty is the
    one-sided hinge `relu(flat_target - r_i)^2`, averaged over the slow rungs only.

    WHY this term has to exist at all. A constant lives ENTIRELY in the channel mean, and every
    other term in the loss uses the CENTERED channel (`sc = s_t - mu`): slowness, whitening,
    reconstruction and the energy floor are all computed on anomalies, so none of them can see a
    static signal. Worse, two terms actively forbid one -- `l_var = relu(1 - std(s_i))^2` demands
    every channel keep unit normalized std, and `l_energy` demands its cells be energetic in TIME.
    A stationary channel is therefore not merely unrewarded but penalized. So the slow rungs get
    this term and are exempted from `l_var` / `l_energy` / `L_band` / `L_line`, which all encode
    "must fluctuate" assumptions (see train.py). `L_level` is what keeps them on signal instead of
    in an empty corner -- it replaces the protection `l_energy` was giving (F-6).

    Uses the FULL series rather than a window: "constant" is a statement about all of T, and a
    window mean would let a slow drift pass as flat inside each window.

    Both `level` and `fluct` are homogeneous of degree 1 in the mask scale, so `r` is scale-free in
    value AND gradient (the D-017 rule).
    """
    r = flat_ratio(S_all)                                   # [K] in [0,1], for logging in either measure
    pen = torch.where(is_slow, flat_penalty(S_all, measure, flat_target, amp_target), torch.zeros_like(r))
    return pen.sum() / is_slow.sum().clamp_min(1), r.detach()


def memory_term(S_win: torch.Tensor, is_slow: torch.Tensor, lag: int, target: float):
    """Long-lag autocorrelation hinge on the SLOW rungs: `relu(target - rho_i(lag))^2`.

    WHY a second term for the slow rungs (measured, run 20260831_091525). With the band hinge alone
    the slow rungs did take the "stationary" label but locked onto the WRONG content: tau_e 36/27
    against the truth's 106, best-matching hidden mode a chaotic one at corr 0.50/0.33, while two
    CYCLIC rungs picked up the actual OU mode. "Power at periods >= 128" is too weak a target --
    any large red-ish footprint satisfies it.

    A long-lag autocorrelation is the sharp version of the same question, and it discriminates
    exactly where the band cannot. At lag 64 on this testbed:
        OU tau=200     rho = exp(-64/200)          = +0.73   <- wanted
        cycle P=300    rho = cos(2*pi*64/300)      = +0.23
        cycle P=140    rho = cos(2*pi*64/140)      = -0.96
        cycle P=60     rho = cos(2*pi*64/60)       = +0.91   (aliased -- see below)
    A cycle whose period nearly divides the lag aliases back to rho ~ 1, so the hinge alone would
    admit period-60 content; that is what `line_cap` on the same rung is for. The two terms are
    complementary: memory says "still correlated a long time later", line_cap says "and not because
    you are a metronome".

    Still a ratio (rho = covariance / variance) -> scale-free in value and gradient (D-017).
    """
    xc = S_win - S_win.mean(dim=1, keepdim=True)           # [n_win,L,K]
    v = (xc ** 2).mean(dim=(0, 1)).clamp_min(1e-20)        # [K] pooled variance
    rho = (xc[:, :-lag] * xc[:, lag:]).mean(dim=(0, 1)) / v                     # [K]
    pen = torch.where(is_slow, torch.relu(target - rho) ** 2, torch.zeros_like(rho))
    # mean over the SLOW rungs only, so the weight does not depend on how many fast rungs exist
    n_slow = is_slow.sum().clamp_min(1)
    return pen.sum() / n_slow, rho.detach()


def structure_share(S_all: torch.Tensor, kpk_min: int = 4) -> torch.Tensor:
    """Differentiable `cyclic_share = (trend + osc) / total` per channel [K], on the full series.

    Mirrors `src/probes/family.decompose` exactly (see `structure_term` for the why). Shared by
    `structure_term` and the soft family assignment `family_q`.
    """
    xc = S_all - S_all.mean(dim=0, keepdim=True)            # [T,K]; the level is L_level's business
    p = torch.fft.rfft(xc, dim=0).abs() ** 2                # [n_bins,K]
    p = p.transpose(0, 1).clone()                           # [K,n_bins]
    p[:, 0] = 0.0                                           # DC carries no dynamics
    K, n = p.shape
    tot = p.sum(dim=1).clamp_min(1e-20)                     # [K] differentiable normalizer
    k0 = min(max(1, kpk_min), n - 1)                        # below k0: fewer than kpk_min cycles in
    trend = p[:, 1:k0].sum(dim=1)                           #          the record -> the DRIFT band

    # --- the dominant peak's HALF-POWER band, above k0 (mirrors family.decompose exactly) --------
    pd = p.detach()
    hi_band = pd[:, k0:]
    k_pk = hi_band.argmax(dim=1) + k0                       # [K]
    half = torch.gather(pd, 1, k_pk[:, None]) / 2.0         # [K,1]
    ar = torch.arange(n, device=p.device)[None, :].expand(K, n)     # [K,n] bin indices
    below = pd < half                                       # [K,n] outside the half-power band
    below[:, :k0] = True                                    # the search stops at the drift band
    # left edge = one past the LAST sub-half bin at or before the peak; right edge symmetrically.
    # This picks the CONTIGUOUS run containing the peak, which a plain `pd >= half` mask would not.
    lo = torch.where(below & (ar <= k_pk[:, None]), ar, torch.full_like(ar, -1)).max(dim=1).values + 1
    hi = torch.where(below & (ar >= k_pk[:, None]), ar, torch.full_like(ar, n)).min(dim=1).values - 1
    lo = torch.minimum(lo, k_pk - 2).clamp_min(k0)          # widened to at least +-2 bins, so a pure
    hi = torch.maximum(hi, k_pk + 2).clamp_max(n - 1)       # tone is not scored as a single bin
    in_osc = ((ar >= lo[:, None]) & (ar <= hi[:, None])).float()   # detached membership
    osc = (p * in_osc).sum(dim=1)

    return (trend + osc) / tot


def structure_term(S_all: torch.Tensor, is_cyclic: torch.Tensor, is_fast: torch.Tensor,
                   struct_target: float, struct_cap: float, kpk_min: int = 4):
    """Shape hinge written in the LABELLER'S OWN QUANTITY: the structure share `trend + osc` (D-027).

    WHY this replaces `L_line`. The readout decides cyclic-vs-chaotic by `trend + osc >= residual`,
    i.e. a cut at structure share 0.5 (D-026). The training hinge asked a DIFFERENT question --
    `linefrac`, the power within +-2 bins of the peak, capped at 0.75 -- and measured on 512-step
    windows rather than the record. The two never met. Measured on run 20260901_080506:

        fast rung        7    8    9   10   11   12   13   14   15
        linefrac_win  0.15 0.40 0.44 0.16 0.43 0.43 0.39 0.45 0.26   <- cap 0.75: SILENT on all nine
        struct share  0.16 0.39 0.41 0.16 0.33 0.41 0.38 0.44 0.36   <- cut  0.5: six within 0.15

    So nothing in the loss ever pushed a fast rung to be broadband, and six of nine drifted up to
    the readout's decision boundary and sat there -- the "ambiguous cyclic-or-chaotic" channels.
    This is the F-13/`flat_target` lesson recurring in a harder form: there the objective and the
    readout disagreed on the THRESHOLD, here they disagree on the QUANTITY.

    The bar is calibrated on the physics, not chosen: the generator's own chaotic modes score
    structure 0.17-0.35 (median 0.21) and its cyclic modes 0.94, so a `struct_cap` of 0.30 asks a
    fast rung for no more oscillatory character than the hidden chaotic modes actually have, and a
    `struct_target` of 0.85 sits just under what the hidden cyclic modes reach. Both are one-sided
    hinges with a MARGIN either side of 0.5, so a satisfied channel is unambiguous rather than
    merely on the right side of a coin flip.

    Computed on the FULL series, not on windows, so it is the same measurement the probe makes.
    (`L_level` already does this; "how structured is this channel" is a claim about the record.)

    Every quantity is a ratio of powers -> scale-free in value and gradient (D-017). The band edges
    and the peak index are DETACHED -- they are integer set memberships with no gradient of their
    own -- while the power summed inside them is differentiable, which is what carries the signal.
    """
    struct = structure_share(S_all, kpk_min)                # [K] the readout's `cyclic_share`
    pen = torch.where(is_cyclic, torch.relu(struct_target - struct) ** 2,
                      torch.where(is_fast, torch.relu(struct - struct_cap) ** 2,
                                  torch.zeros_like(struct)))       # slow rungs: L_level's job
    n_act = (is_cyclic | is_fast).sum().clamp_min(1)
    return pen.sum() / n_act, struct.detach()


def window_power(S_win: torch.Tensor) -> torch.Tensor:
    """Window-averaged periodogram [K, n_bins] of `S_win` [n_win, L, K], DC bin zeroed.

    Each window's own mean is removed, not the global mean: a slow drift across windows would
    otherwise land in bin 0. Shared by `spectral_terms` and `family_q`.
    """
    xc = S_win - S_win.mean(dim=1, keepdim=True)
    p = torch.fft.rfft(xc, dim=1).abs() ** 2               # [n_win, n_bins, K]
    p = p.mean(dim=0).transpose(0, 1).clone()              # average the periodograms -> [K, n_bins]
    p[:, 0] = 0.0                                          # drop the DC bin (see band_plan)
    return p


def line_fraction(p: torch.Tensor) -> torch.Tensor:
    """Power within +-2 bins of the peak over total power [K]. `argmax` is detached (an index has no
    gradient anyway); the SUM over that window is differentiable, which is what carries the signal."""
    tot = p.sum(dim=1).clamp_min(1e-20)
    n_bins = p.shape[1]
    k_pk = p.detach().argmax(dim=1)                        # [K]
    offs = torch.arange(-2, 3, device=p.device)
    idx = (k_pk[:, None] + offs[None, :]).clamp(0, n_bins - 1)   # [K,5]
    return torch.gather(p, 1, idx).sum(dim=1) / tot


def spectral_terms(S_win: torch.Tensor, band: torch.Tensor, is_cyclic: torch.Tensor,
                   band_target: float, line_target: float, line_cap: float,
                   chan_w: torch.Tensor | None = None, line_w: torch.Tensor | None = None):
    """Band-placement + line-shape hinges from windowed channel series.

    S_win     [n_win, L, K]  contiguous windows of the channel series (GRAD-CARRYING)
    band      [K, n_bins]    band indicator from `band_plan`
    is_cyclic [K] bool       which rungs are asked FOR a spectral line (the others are capped)
    chan_w    [K] float      per-channel weight, for EXEMPTING rungs. Under the D-025 definition the
                             stationary rungs are asked to be constant, and a constant has all of
                             its power in the DC bin that this function deliberately zeroes -- so
                             both hinges here would be scored on its leftover noise. Pass 0 for
                             those channels; `L_level` drives them instead.
    line_w    [K] float      separate weight for `L_line` only (defaults to `chan_w`). Under D-027
                             with `struct_rungs: fast`, the FAST rungs' shape is owned by
                             `L_struct`, so `L_line` is kept on the cyclic rungs alone. It only
                             selects rungs: the average is still over `chan_w`, so the per-rung
                             weight does not change with how many rungs are selected.

    Returns (l_band, l_line, diag) with diag holding per-channel `bandfrac` / `linefrac` for logging.
    """
    p = window_power(S_win)                                # [K, n_bins], DC dropped
    tot = p.sum(dim=1).clamp_min(1e-20)                    # [K] differentiable -> ratios are
    #                                                        scale-free in the gradient too (D-017)

    bandfrac = (p * band).sum(dim=1) / tot                 # [K] fraction of power inside the band
    w = torch.ones_like(bandfrac) if chan_w is None else chan_w
    w_sum = w.sum().clamp_min(1e-6)
    l_band = ((torch.relu(band_target - bandfrac) ** 2) * w).sum() / w_sum

    linefrac = line_fraction(p)                            # [K] line concentration around the peak

    pen = torch.where(is_cyclic,
                      torch.relu(line_target - linefrac) ** 2,      # cyclic rungs: BE a line
                      torch.relu(linefrac - line_cap) ** 2)         # others: do NOT be a line
    # `line_w` SELECTS which rungs get the line hinge, but the average stays over all dynamic rungs
    # (`w_sum`), so each selected rung keeps the per-rung weight it had in D-024. Averaging over the
    # selected rungs only made the hinge 14/5 = 2.8x stronger per cyclic rung, which pinned them on the
    # period-60 cycle they start on before L_band could move them to their octaves (seed 0 diagnosis).
    lw = w if line_w is None else line_w
    l_line = (pen * lw).sum() / w_sum

    return l_band, l_line, dict(bandfrac=bandfrac.detach(), linefrac=linefrac.detach())


# ---------------------------------------------------------------------------------------------------------
# SOFT family assignment (SOP 02 `role_mode: soft`, user 2026-10-29)
# ---------------------------------------------------------------------------------------------------------
FAMILY_ROLE = ("slow", "cyclic", "fast")     # q column -> the role name every probe already understands


def family_q(r: torch.Tensor, c: torch.Tensor, alpha: float, beta: float, tau: float,
             r0: float, c0: float = 0.5) -> torch.Tensor:
    """Soft family membership per channel, inferred from its own signal: q [K,3] = (stat, cyc, chao), rows sum to 1.

    Follows the labeller's decision ORDER, so argmax q is the post-hoc label made differentiable:
        q_stat = sigma(alpha (r - r0) / tau)                         flat first   (labeller: amp_ratio < FLAT_THR)
        q_cyc  = (1 - q_stat) * sigma(beta (c - c0) / tau)           then structure share vs 1/2
        q_chao = (1 - q_stat) * (1 - sigma(beta (c - c0) / tau))     (labeller: cyclic_share >= residual)
    A categorical distribution, i.e. a softmax over the logits log q. WHY nested and not one flat softmax over
    [alpha(r-r0), beta(c-1/2), beta(1/2-c)]: measured on trained eyes, a truly flat channel (r=0.96, c=0.28) gets
    logits ~[0.03, -2.2, +2.2] and would be called CHAOTIC -- flatness has to be decided before shape, exactly as
    the labeller does. q_stat crosses 1/2 exactly at the labeller's flat cut r0.
    No parameters: the only knobs are the logit scales alpha, beta and the temperature tau. alpha is kept soft
    (5) so a channel far from flat still feels the population pull (untrained eyes measure r 0.03-0.65).
    """
    q_stat = torch.sigmoid(alpha * (r - r0) / tau)
    p_cyc = torch.sigmoid(beta * (c - c0) / tau)
    return torch.stack([q_stat, (1 - q_stat) * p_cyc, (1 - q_stat) * (1 - p_cyc)], dim=1)


def soft_q(S_all: torch.Tensor, sp, tau: float | None = None) -> torch.Tensor:
    """q [K,3] for a channel series, in the configured flatness measure. One place, so training and the saved
    final q are computed the same way.

    `flat_measure: ratio` -> family_q as above. `flat_measure: log` -> the stationary vote is read in log units:
        q_stat = sigma(alpha_log (log amp0 - log amp_ratio) / tau),  amp0 = (1 - r0) / r0  (= the labeller's 0.05)
    so it still crosses 1/2 exactly at the labeller's cut, but r = 0.965 (amp 0.036) is no longer "half stationary":
    with alpha_log = 2, tau = 0.2 it gives q_stat 0.96, and amp 0.02 gives ~1.
    """
    so = sp.soft
    t = float(so.tau) if tau is None else tau
    r, c = flat_ratio(S_all), structure_share(S_all)
    q = family_q(r, c, float(so.alpha), float(so.beta), t, float(so.r0))
    if str(sp.get("flat_measure", "ratio")) == "log":
        amp0 = (1.0 - float(so.r0)) / float(so.r0)
        q_stat = torch.sigmoid(float(so.alpha_log) * (float(np.log(amp0)) - log_amp(S_all)) / t)
        p_cyc = torch.sigmoid(float(so.beta) * (c - 0.5) / t)
        q = torch.stack([q_stat, (1 - q_stat) * p_cyc, (1 - q_stat) * (1 - p_cyc)], dim=1)
    return q


def soft_family_terms(S_all: torch.Tensor, S_win: torch.Tensor, band3: torch.Tensor, sp, pop: torch.Tensor,
                      tau: float | None = None):
    """Every family's own penalty, weighted by the channel's q instead of a fixed role mask.

    S_all [t_fit,K] full training series; S_win [n_win,L,K] windows; band3 [3,n_bins] one band row per
    family (slow / whole cyclic range / fast); `sp` = cfg.train.spectral; pop [3] target population.

    Per channel and family (all already per-channel vectors in the fixed ladder, same hinges, same lambdas):
      stationary : lambda_level  * relu(flat_target - r)^2   (flat_measure: log -> lambda_level_log * log-amp hinge)
      cyclic     : lambda_line   * relu(line_target - linefrac)^2 + lambda_band * relu(band_target - bandfrac_cyc)^2
      chaotic    : lambda_struct * relu(c - struct_cap)^2          + lambda_band * relu(band_target - bandfrac_fast)^2
    The cyclic shape stays on `linefrac`, not the structure share: structure on cyclic rungs collapsed them all
    onto the period-60 cycle (D-027 measurement).

    L_family = mean_i sum_f q_if * pen_if   -- q NOT detached: each channel slides toward its nearest family.
    L_pop    = KL(pop || mean_i q_i)        -- keeps the families populated (emergence alone failed, F-5/F-12).
    L_ent    = mean_i H(q_i)                -- decisiveness. WHY: L_pop sees only the MEAN of q, and the first soft
               runs met it by spreading q_stat 0.1-0.5 over many channels with none actually flat. Low entropy
               per channel forces the target share to be carried by channels that commit.
    `tau` overrides the config temperature (annealing). Returns (l_family, l_pop, l_ent, q, diag).
    """
    r = flat_ratio(S_all)                                   # [K]
    c = structure_share(S_all)                              # [K]
    q = soft_q(S_all, sp, tau)                              # [K,3] in the configured flatness measure
    measure = str(sp.get("flat_measure", "ratio"))
    lam_flat = float(sp.lambda_level_log) if measure == "log" else float(sp.lambda_level)

    p = window_power(S_win)                                 # [K,n_bins]
    tot = p.sum(dim=1).clamp_min(1e-20)
    bandfrac = (p @ band3.T) / tot[:, None]                 # [K,3] power share in each family's band
    linefrac = line_fraction(p)                             # [K]

    pen = torch.stack([
        lam_flat * flat_penalty(S_all, measure, float(sp.flat_target), float(sp.get("flat_amp_target", 0.02))),
        float(sp.lambda_line) * torch.relu(float(sp.line_target) - linefrac) ** 2
        + float(sp.lambda_band) * torch.relu(float(sp.band_target) - bandfrac[:, 1]) ** 2,
        float(sp.lambda_struct) * torch.relu(c - float(sp.struct_cap)) ** 2
        + float(sp.lambda_band) * torch.relu(float(sp.band_target) - bandfrac[:, 2]) ** 2,
    ], dim=1)                                               # [K,3]
    l_family = (q * pen).sum(dim=1).mean()
    qbar = q.mean(dim=0).clamp_min(1e-8)                    # soft population, sums to 1
    l_pop = (pop * (pop.clamp_min(1e-8).log() - qbar.log())).sum()
    l_ent = -(q * q.clamp_min(1e-8).log()).sum(dim=1).mean()
    return l_family, l_pop, l_ent, q, dict(r=r.detach(), c=c.detach(), linefrac=linefrac.detach(),
                                    bandfrac=bandfrac.detach(), pen=pen.detach())
