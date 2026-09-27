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


def level_term(S_all: torch.Tensor, is_slow: torch.Tensor, flat_target: float):
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
    level = S_all.mean(dim=0).abs()                         # [K]
    fluct = S_all.std(dim=0)                                # [K]
    r = level / (level + fluct + 1e-12)                     # [K] in [0,1]
    pen = torch.where(is_slow, torch.relu(flat_target - r) ** 2, torch.zeros_like(r))
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

    struct = (trend + osc) / tot                            # [K] the readout's `cyclic_share`
    pen = torch.where(is_cyclic, torch.relu(struct_target - struct) ** 2,
                      torch.where(is_fast, torch.relu(struct - struct_cap) ** 2,
                                  torch.zeros_like(struct)))       # slow rungs: L_level's job
    n_act = (is_cyclic | is_fast).sum().clamp_min(1)
    return pen.sum() / n_act, struct.detach()


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
                             `L_struct`, so `L_line` is kept on the cyclic rungs alone.

    Returns (l_band, l_line, diag) with diag holding per-channel `bandfrac` / `linefrac` for logging.
    """
    xc = S_win - S_win.mean(dim=1, keepdim=True)           # remove each window's mean, not the
    #                                                        global mean: a slow drift across
    #                                                        windows would otherwise land in bin 0
    p = torch.fft.rfft(xc, dim=1).abs() ** 2               # [n_win, n_bins, K]
    p = p.mean(dim=0).transpose(0, 1)                      # average the periodograms -> [K, n_bins]
    p = p.clone()
    p[:, 0] = 0.0                                          # drop the DC bin (see band_plan)
    tot = p.sum(dim=1).clamp_min(1e-20)                    # [K] differentiable -> ratios are
    #                                                        scale-free in the gradient too (D-017)

    bandfrac = (p * band).sum(dim=1) / tot                 # [K] fraction of power inside the band
    w = torch.ones_like(bandfrac) if chan_w is None else chan_w
    w_sum = w.sum().clamp_min(1e-6)
    l_band = ((torch.relu(band_target - bandfrac) ** 2) * w).sum() / w_sum

    # line concentration: power within +-2 bins of the peak. `argmax` is detached (an index has no
    # gradient anyway); the SUM over that window is differentiable, which is what carries the signal.
    K_, n_bins = p.shape
    k_pk = p.detach().argmax(dim=1)                        # [K]
    offs = torch.arange(-2, 3, device=p.device)
    idx = (k_pk[:, None] + offs[None, :]).clamp(0, n_bins - 1)   # [K,5]
    linefrac = torch.gather(p, 1, idx).sum(dim=1) / tot     # [K]

    pen = torch.where(is_cyclic,
                      torch.relu(line_target - linefrac) ** 2,      # cyclic rungs: BE a line
                      torch.relu(linefrac - line_cap) ** 2)         # others: do NOT be a line
    lw = w if line_w is None else line_w
    l_line = (pen * lw).sum() / lw.sum().clamp_min(1e-6)

    return l_band, l_line, dict(bandfrac=bandfrac.detach(), linefrac=linefrac.detach())
