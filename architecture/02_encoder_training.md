# SOP 02 — v0 Encoder + Training (slowness + whitening)

> Golden Rule: update this SOP before changing `src/models/encoder.py` or `src/train/train.py`.
> Realizes the settled v0 core (CLAUDE.md): learnable selection masks, slowness loss, whitening
> constraint, online pairwise SGD. Defaults recorded in D-015; all are Hydra-tunable.

## Goal

Learn K selection masks whose K scalar channels are **slow** (temporally coherent) under a
**whitening** constraint (unit variance + decorrelation). Whitening blocks the trivial optima
(M=0 collapse, duplicate kernels); decorrelation is what drives channel diversity (D-013).

## Tensors & call order

```
field[T,V,H,W]  --SelectionEncoder-->  S[T,K]   (s_i = <mask_i, field>, norm="none")
   per training step:
     idx ~ U[0,T-2], batch B          -> pairs (t, t+1)
     s_t, s_tp1 = enc(field[idx]), enc(field[idx+1])   # [B,K] each
     C       = cov(center(s_t))                         # [K,K], DIFFERENTIABLE
     v       = diag(C)                                  # [K] channel variance
     R       = C / sqrt(v_i v_j)                        # [K,K] correlation, unit diagonal
   mode "corr" (DEFAULT, D-017):
     L_slow   = mean_i var(s_tp1 - s_t)_i / v_i         # SFA Rayleigh quotient
     L_white  = mean_{i!=j} R_ij^2                      # decorrelation
     L_var    = mean_i relu(1 - std(s_i/sqrt(count_i)))^2  # anti-death hinge
     L_energy = mean_i relu(1 - e_i/e_ref)^2            # energy floor, e_i = v_i/count_i^2  (D-019)
     L_size   = mean_i relu(|log(count_i/tgt_i)| - log(tol))^2   # SCALE LADDER, tgt_i geometric (D-020)
     W        = argmin_W ‖X - [1,S]W‖^2                 # per-batch closed form, DIFFERENTIABLE
     L_recon  = ‖X - [1,S]W‖^2 / ‖X‖^2                  # = 1 - R^2, coverage       (D-019)
     L        = λ_slow·L_slow + λ_white·L_white + λ_var·L_var
                + λ_energy·L_energy + λ_recon·L_recon + λ_size·L_size
                + ladder terms: L_band, L_line (cyclic rungs), L_struct (fast rungs), L_level
                  (stationary rungs) -- see the D-024 / D-025 / D-027 sections below
   mode "hard" / "soft": ABLATIONS. Both collapse -- see "Failure modes" below.
```

**Why an energy/coverage term is REQUIRED, not optional (D-019, finding F-6):** `L_slow` is a
ratio, so it is blind to amplitude. Left alone it walks every kernel into the lowest-energy cells
of the domain (the edges), where the field is the far tail of the big slow mode: tiny amplitude,
almost purely slow, near-perfect ratio. Scale-invariance is mandatory to prevent the mask collapse
(F-2) and simultaneously rewards looking where nothing happens. `e_i = v_i/count_i^2` is the fix at
the per-channel level (`count^2` normalization makes it independent of mask SIZE, so it measures
"are the selected cells energetic AND mutually coherent"). `L_recon` is the fix at the collective
level, and additionally penalizes REDUNDANCY: a duplicate channel adds no explanatory power, so its
gradient points at the unexplained residual.

**The one rule that matters here (D-017):** every normalization in the loss must be
DIFFERENTIABLE. Autograd treats a `.detach()`-ed variance as a constant, so a "scale-free" loss
written with a detached std is *not* scale-free to the gradient, and the optimizer will shrink
every mask to zero. Three separate collapses (findings F-2) all trace back to this. No `.detach()`
in the loss path, no matrix inverse, and the ratio `var(Δs)/var(s)` written explicitly.

## What the observer IS, and how channels are classified (D-026, user ruling 2026-09-01)

**The observer is not a mode-recovery device.** Hidden modes are just hidden modes; in a real ocean
or atmosphere, capturing all of them is barely possible. What the kernel learns is WHERE TO LOOK, and
a family is a property of **the observable the channel constructs** — not of some injected mode. A
stationary channel may be a single static mode, a persistent phenomenon, or a COMBINATION of varying
signals whose sum barely moves. All three are legitimate.

This changes the SCORING, not just the code:

| quantity | status |
|---|---|
| `population` vs 1/3/6 | a **DESIGN** target — the mix of observables we ask the bank to build (the rung ladder). NOT a mode-recovery score. |
| `role_obedience` | the real check on the ladder: did each channel become what its rung asked for? |
| `amp_ratio < 0.05` | the ONLY criterion for a stationary channel. No persistence check (user's call: flatness over the record is enough). |
| `mode_recovery` (max\|corr\|, ensemble R²) | DIAGNOSTIC. |
| `mask_align_with_pattern` | DIAGNOSTIC. Judging a flat channel by its alignment with the injected pattern was the wrong frame. |
| `cancellation_vs_independent` | DIAGNOSTIC: HOW a flat channel got flat (see below). |

**Cancellation is allowed but not encouraged** — no loss term rewards it. `L_level` rewards flatness
however the kernel achieves it. To tell the mechanisms apart, the probe projects every hidden mode
onto the flat channel's footprint and compares the net fluctuation to the independent-addition
baseline `sqrt(Σ sd_k²)`:

```
< 0.7   destructive interference beyond chance   -> a cancelling COMBINATION
~ 1.0   incoherent addition                      -> a genuinely quiet footprint / persistent phenomenon
> 1.3   the contributions reinforce
```

Measured on the ladder run `20260901_080506`, ch1: 3 hidden modes reach the footprint, index
**1.01** — so that stationary observer is a quiet footprint, not a cancellation. (Normalize against
INDEPENDENCE, not against the plain sum of the parts: 3 equal independent parts already give 0.58 of
the sum, so a "net/sum = 0.63" reading looks like cancellation while being exactly what chance
predicts. The wrong baseline invents findings.)

### Classification follows GLOBAL STRUCTURE, not the residue

A channel that visibly oscillates while carrying fast noise on a wandering baseline is CYCLIC. My
earlier ACF-recurrence test read the RAW series, where the noise and the drift dominate the
autocorrelation, so it answered a question about the residue and called such channels non-cyclic.
`family.decompose` now splits a channel four ways:

```
level    = mean(x)                        -> gates the stationary label via amp_ratio
trend    = periodogram bins 1..3          -> fewer than 4 cycles in the record: at that resolution
                                             a drift and a "cycle" are indistinguishable, so this
                                             band IS the drift band. Counts toward CYCLIC (D-025).
osc      = the dominant peak's HALF-POWER band (bins >= 4), min +-2 bins. Adapts to the peak: ~1 bin
           for a pure tone, wide for a quasi-periodic hump -- a fixed window would have scored a
           broad hump as residual, i.e. called a real oscillation chaotic.
residual = the rest, above bin 4          -> genuinely broadband
```

`label = flat gate (amp_ratio) first, then trend+osc vs residual`.

**Why the level GATES instead of joining the argmax.** A plain argmax over all four shares is
tidier but wrong: `level² > var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its
level would score "stationary". Counter-example on the record: ch14 of run `20260901_080655`,
amp_ratio 0.526, plainly broadband chaotic.

**Measured (post-hoc — a label is a readout, so no retraining is needed to change one):**

| | truth accuracy | population |
|---|---|---|
| rev3 truth modes | **10/10** | 1/3/6 |
| rev2 truth modes | 9/10 — the "miss" is the old OU mode 0 now labelled **cyclic** (trend share 0.63), which is D-025 as ruled | — |
| rev3 ladder `20260901_080506` | — | 2/5/9, role obedience **1.00** |
| rev3 baseline `20260901_080438` | — | 0/11/5 |

Channel-level: ch2-6 of the ladder run are cyclic at osc share 0.70-0.93. ch9/ch11 stay chaotic at
osc share 0.41/0.32 against residual 0.59/0.67 — real oscillatory character with broadband still
dominant, which is the honest reading of "has cyclic characteristics but is not 100% cyclic".

**Known limit:** a non-sinusoidal periodic signal puts power in harmonics, which land in `residual`
and bias the vote toward chaotic. Harmonic folding is a deferred refinement.

## STATIONARY = CONSTANT (D-025, user ruling 2026-09-01) — read this before the ladder section

The section below was written with "stationary" meaning a SLOW, long-memory channel. That is not
this project's definition. **A stationary signal is a CONSTANT one — flat in time.** Consequences,
all measured (finding F-13):

- **The testbed had no flat mode at all.** `_ou_series` ends in `_standardize`, so the old mode 0
  was emitted at UNIT VARIANCE — as much temporal energy as the sinusoids. Generator rev3
  (`data.stationary_constant: true`) makes mode 0 a static pattern with a flat amplitude, living in
  the field's TIME MEAN. Verified: amp std 0.000, and the field's time-mean map correlates 1.000
  with the mode-0 pattern (rms 0.818 vs field std 1.0).
- **A slow red DRIFT is `cyclic`, not stationary** (user's call: it is not constant). The labeller
  therefore tests FLATNESS first, on `amp_ratio = std/|mean|` measured on the RAW series — a
  standardized series carries no amplitude information, so the test is skipped rather than guessed
  when the caller cannot supply it. 10/10 on the hidden truth over seeds 0-4.
- **The loss was blind to a constant, and partly hostile to one.** Every term uses the CENTERED
  channel (`sc = s_t - mu`), so a static signal is invisible to slowness, whitening, coverage and
  the energy floor; and `l_var = relu(1 - std)^2` plus `l_energy` actively PENALIZE a flat channel.
  So the stationary rungs now get `L_level` and are exempted from `l_var` / `l_energy` / `L_band` /
  `L_line`:

```
level_i = |mean_t s_i|      fluct_i = std_t s_i        r_i = level_i / (level_i + fluct_i)
L_level = mean over SLOW rungs of relu(flat_target - r_i)^2
```

  `flat_target` MUST match the labeller's cut or the hinge goes quiet before the channel qualifies:
  `r = 1/(1 + amp_ratio)`, and flat means `amp_ratio < 0.05`, so `r > 0.952` -> `flat_target=0.95`.
  A first attempt at 0.8 corresponded to `amp_ratio` 0.25, i.e. a channel that still visibly moves.
  `L_level` also replaces the protection `l_energy` was giving those rungs (F-6): it is what keeps a
  flat channel on the static STRUCTURE instead of in an empty corner.

**Measured, rev3 field, K=16, 5000 steps, seed 0:**

| | baseline (no ladder) | ladder + `L_level` |
|---|---|---|
| run | `20260901_080438` | `20260901_080506` |
| population (s/c/ch) | 0/14/2 | **2/5/9** (target proportion 1.6/4.8/9.6) |
| role obedience | n/a | **1.00** |
| flattest channel `amp_ratio` | 0.778 (*still moving*) | **0.023** |
| flat channels (< 0.05) | 0 | 2 |
| balanced recon R^2 | 0.7763 | 0.7765 |

The baseline provably cannot produce a stationary observer; nothing in it gets near flat. Note rev3
scores 0.78 balanced recon where rev2 scored 0.99 — the new field has a large static component that
anomaly-based decoding does not explain, and both runs are equally affected.

**Figure caveat (fixed, but know why).** `plots.py` used to standardize each channel before
plotting, dividing out the std — the very quantity that defines flatness. A channel fluctuating at
2% of its level was drawn as dynamic as a sinusoid. It now scales by RMS about zero and prints
`amp_ratio` per lane, so a constant channel reads flat. Second time a figure has misled us on this
project (cf. F-6): check what a plot normalizes by before trusting it.

**Tested and rejected:** de-aligning the size and timescale ladders (`slow_size_frac=0.08`) to
un-pin the flat rungs from the largest footprints. It halved the footprint (2590 -> 1318 cells) but
did not improve alignment with the injected pattern (|corr| 0.27 either way) and cost role obedience
(1.00 -> 0.94). Default 0.0.

**Still open:** the flat channels report a stable level but align with the injected pattern at only
|corr| 0.27 — they settle on a diffuse static average or the pattern's negative lobe. Open question
for the user: must the observer ISOLATE the static pattern, or only report a stable level?

## Spectral band ladder — the TIMESCALE assignment (D-024, added 2026-08-31)

**The problem it solves.** `L_slow` is ONE objective shared by all K channels, so the channels
compete for the same globally-slowest content. Raising K therefore does not redistribute the
dynamical families — it buys more near-duplicates of whatever regime dominates. Measured on the
K sweep (`.tmps/runs/20260831_0736*`), populations read 3/0/1 (K=4), 4/2/2 (K=8), 12/0/4 (K=16),
22/3/7 (K=32), 36/10/18 (K=64): slow-dominated at every K, almost no clean cyclic channel.
**K is not the lever on which families appear.** (These numbers only became visible after the
labeller was fixed — see F-11 and `src/probes/family.py`.)

**The mechanism** is the temporal twin of the D-020 size ladder: stop asking every channel for the
same thing, and ASSIGN each one a target band, penalizing only outside it.

```
roles     = rung_roles(K, pop_target)      # [1,3,6] scaled to K -> K=16: 2 slow / 5 cyclic / 9 fast
band      = band_plan(...)                 # [K,n_bins] which rFFT bins each rung should occupy
S_win     = n_win random CONTIGUOUS windows of length win_len, sliced from enc(field)
p_i(k)    = periodogram of channel i, averaged over windows, DC bin dropped
bandfrac_i= in-band power / total power
linefrac_i= power within +-2 bins of the peak / total power
rho_i(L)  = lag-L autocorrelation, L = mem_lag

L_band = mean_i relu(band_target - bandfrac_i)^2
L_line = mean_i { relu(line_target - linefrac_i)^2   for CYCLIC rungs   (be a line)
                { relu(linefrac_i  - line_cap  )^2   otherwise          (be a hump, not a line)
L_mem  = mean over SLOW rungs of relu(mem_target - rho_i(mem_lag))^2
L     += λ_band·L_band + λ_line·L_line + λ_mem·L_mem
```

**Why three terms and not one.**
- The families OVERLAP in frequency on this testbed (cycles at 60/140/300, OU at tau=200), so a
  frequency band cannot separate a slow drift from a slow cycle. `L_line` does that, via the same
  line-vs-hump statistic the labeller uses (0.94 for the truth's cycles, 0.67 for its OU).
- `L_band` alone left the slow rungs on the WRONG content: tau_e 36/27 against the truth's 106,
  best-matching hidden mode a *chaotic* one at corr 0.50 (run `20260831_091525`). "Power at periods
  >= 128" is too weak — any large red-ish footprint satisfies it. `L_mem` is the sharp version:
  at lag 64, OU(tau=200) gives rho=+0.73 while the period-300 and period-140 cycles give +0.23 and
  -0.96. With `L_mem` on, that rung reached tau_e 87 and corr **0.92** with the true stationary mode.
- A cycle whose period nearly divides `mem_lag` aliases back to rho~1, so `L_mem` alone would admit
  a metronome; `line_cap` on the same rung rejects it. The two terms are complementary.

**Rejected alternative (do not re-try):** matching a target lag-L autocorrelation `exp(-L/tau_i)`
per channel. It is wrong for a cyclic channel, whose ACF oscillates and whose long-lag gap reaches
4 — twice the exponential maximum — so it would actively suppress the cyclic family it was meant
to create. Frequency-domain band + line shape has no such bias.

**Why windows, not pairs.** A lag-1 pair minibatch says nothing about a period-300 cycle. The
periodogram needs contiguous time, so the ladder slices `n_win` random windows out of `enc(field)`
(one einsum over T x 8192 cells — cheap enough to redo every step). The stochasticity that D-011
asks for now comes from the window starts rather than the pair index.

**Measured (K=16, 5000 steps, seed 0, same config otherwise):**

| | baseline `spectral.enable=false` | + band/line ladder | + `L_mem` (default) |
|---|---|---|---|
| run | `20260831_091454` | `20260831_091525` | `20260831_091908` |
| population (s/c/ch) | 10/1/5 | 6/1/9 | 5/1/10 |
| role obedience | n/a | 0.75 | 0.69 |
| slow rung -> true stationary | (unassigned) | corr 0.50 to a *chaotic* mode | **corr 0.92, tau_e 87** |
| worst hidden-mode maxcorr | 0.25 (m6, m9) | 0.65 | 0.61 |
| effective rank / 16 | 9.23 | 10.08 | 9.56 |
| balanced recon R^2 | 0.9891 | 0.9891 | 0.9891 |

Reading: the fast rungs obey almost perfectly (9-10 of 9-10 land chaotic, gap1 ~0.10-0.13 against
the truth's 0.07-0.20, where the baseline's "chaotic" channels sat at 0.02-0.03), every hidden mode
is now seen by some channel at |corr| >= 0.61 (baseline left two chaotic modes at 0.25), and
reconstruction is unchanged — the ladder redistributes the channels without costing coverage.
`figs/features.png` shows the bank stratified slow -> cyclic -> fast down the channel index, where
the baseline's families are interleaved with no order.

**Known open gap (NOT fixed by this ladder).** The cyclic rungs still mostly fail: 1 of 5 reaches a
clean line (`linefrac` 0.92; the rest sit at 0.52-0.60 and get labelled stationary). A non-negative
regional mask sums everything under its footprint, and the basin-scale slow content leaks into every
region, so a channel cannot cancel its own red background — the same non-negativity limit as F-4,
now showing up in the time domain. Candidate levers, in order of cost: (a) a temporal high-pass in
the definition of `s_i` for cyclic/fast rungs, which changes the observer's semantics (D-005) and
needs a user decision; (b) more sign structure in the generator's mid-scale patterns; (c) accept it
and let module 2 handle mixed channels. The second slow rung also fails, and it is the LARGEST
footprint on the size ladder — the D-021 conflict between `L_energy` and the biggest rung. Not
aligning the two ladders is the obvious next test.

## Fast-rung shape term `L_struct` (D-027, added 2026-09-10 → 2026-09-27)

**The problem.** Channels were "ambiguous cyclic or chaotic" (user, 2026-09-10). Measured on run
`20260901_080506`: 7 of 16 channels sat within the ambiguity band, **all of them fast rungs**, at
structure share 0.33–0.44 against the labeller's cut at 0.5. The cyclic rungs were decisive
(0.74–0.93). The hidden chaotic modes score 0.17–0.35, so the fast channels really were
contaminated by oscillatory content — an encoder problem, not a labeller problem.

**Root cause: the objective and the readout asked different questions.** The labeller votes on
`(trend + osc) / total` vs `residual / total` (D-026). The training hinge `line_cap` scored power
within ±2 bins of the peak, capped at 0.75, on 512-step windows. It was measured **silent on all nine
fast rungs** (they sat at 0.15–0.45), so nothing ever pushed them to be broadband. Same lesson as
`flat_target`/`FLAT_THR` (D-025), in a harder form: there the two disagreed on the *threshold*, here
on the *quantity*.

**The term** (`structure_term` in `src/train/spectral.py`):

```
S_all    = enc(field)                          # [T,K] full record -- the same measurement the probe makes
p_i(k)   = |rFFT(s_i - mean)|^2, DC dropped
trend_i  = power in bins 1..3                  # < 4 cycles in the record: the drift band
osc_i    = power in the dominant peak's half-power band above bin 4, at least ±2 bins
struct_i = (trend_i + osc_i) / total_i         # = the labeller's `cyclic_share`
L_struct = mean over FAST rungs of relu(struct_i - struct_cap)^2      struct_cap = 0.30
```

- It asks a fast rung for at least 70% broadband power. The cap is calibrated on the physics: the
  hidden chaotic modes measure 0.17–0.35.
- It never filters the signal. `s_i = <mask_i, field>` is unchanged (D-005); the gradient moves the
  mask toward cells whose signal is broadband.
- Peak location and band edges are detached (integer set memberships); the power summed inside them
  is differentiable. A ratio of powers → scale-free in value and gradient (D-017).
- `lambda_struct = 2.3`, gradient-matched to `lambda_white` at init, K=16 seed 0:
  `|g_white| = 5.1e-3`, `|g_struct| = 2.3e-3`.

**Why fast rungs ONLY (`struct_rungs: fast`, default).** The first variant also asked the cyclic
rungs for `struct ≥ 0.85`. `trend + osc` credits ANY clean peak and never checks the rung's assigned
octave, so on seed 0 all five cyclic rungs moved to the period-60 cycle, the cleanest line in the
field (baseline: 286/143/61/61/61). The cap on the fast side favours no mode. The cyclic rungs keep
`L_line` (`line_target = 0.85`); `L_line` is switched off on the fast rungs, whose shape `L_struct`
now owns. `struct_rungs: all` reproduces the first variant; `shape_objective: line` reproduces D-024.

**Measured (K=16, 5000 steps, 5 seeds each, runs `.tmps/runs/<arm>_seed{0..4}`):**

| arm | ambiguous ch / seed | fast struct max | distinct cyclic periods (of 3) | flat ch / seed | role obedience |
|---|---|---|---|---|---|
| `base` — `shape_objective: line` | 6.8 | 0.44 | **2.4** | 0.8 | 0.91 |
| `struct` — L_struct on all rungs, λ 3.5 | 3.0 | 0.31 | 1.0 | 0.6 | 0.91 |
| **`fastonly` — default** | **1.2** | 0.31 | 1.4 | 0.8 | 0.93 |

"Ambiguous" = structure share in 0.35–0.65 (|margin| < 0.30), flat channels excluded. The 0.30 is a
convenience, not a calibration: hidden chaotic mode m6 sits exactly on its edge (margin −0.30).
Coverage is unchanged (balanced recon R² 0.7765 vs 0.7765, seed 0).

**Known open gap — the cyclic rungs collapse onto one cycle per seed.** Fast-only fixes the
ambiguity, but it does NOT restore the slow cycles: seed 0 keeps all five cyclic rungs on period 61,
where the baseline on the same seed keeps 286 and 143. The collapse also exists in the baseline
(seed 3: all five on 143), and fast-only makes it more frequent (2.4 → 1.4 distinct). The cause is
not yet known — it is not the cyclic side of `L_struct`, since fast-only removed it. Two built-in
contributors: `L_band` holds the cyclic rungs weakly (on seed 0 the rung assigned 191–320 sits on
61), and 5 cyclic rungs share 3 cycles, with no cycle at all in the shortest octave (24–40).
**Next step: diagnose on seed 0** (period and in-band power of each cyclic rung over training,
baseline vs fast-only) before choosing between in-octave credit, a stronger band, or a new layout.

**Always check timescale coverage, not just ambiguity.** The first variant scored well on the
ambiguity count while every cyclic rung sat on one cycle. `.tmps/score_runs.py` now prints each cyclic
rung's dominant period and the number of distinct periods; the kernel/signal figures
(`src/probes/plots.py`) are what exposed it.

## v0 defaults (D-015 — all overridable via ./config/)

| knob | default | alt (deferred) |
|---|---|---|
| mask parameterization | soft-binary `sigmoid(w/temp)` | straight-through binary; signed `tanh` (ablation) |
| mask init | **multiscale** (own length scale per channel, D-016) | random / flat (flat is degenerate) |
| temp anneal | **OFF** (`temp0 = temp1 = 1.0`, D-016) | anneal → 0.1 AFTER masks organize; STE |
| mask domain | full-volume `(V,H,W)` | per-layer `(H,W)` |
| `s_i` normalization | **none** (whitening owns the scale) | soft-count / sqrt-count (ablation) |
| whitening | **`corr`** (differentiable off-diag decorrelation, D-017) | `hard`/`soft` (both collapse) |
| energy floor | `lambda_energy=0.2` (gradient-matched) | off (kernels flee to dead corners) |
| coverage | `lambda_recon=0.3` (gradient-matched) | off (chaotic modes get discarded) |
| scale ladder | `lambda_size` on a geometric footprint ladder (D-020) | off (all masks shrink to ~0.3% of domain) |
| band ladder | `spectral.enable=true`, weights gradient-matched (D-024) | off (families do not redistribute at ANY K) |
| stationary rungs | `spectral.slow_objective=level` + `L_level` (D-025: flat = constant) | `memory` (reads stationary as a slow DRIFT -- ablation) |
| shape term | `spectral.shape_objective=structure`, `struct_rungs=fast`: `L_struct` caps fast rungs, `L_line` on cyclic rungs (D-027) | `struct_rungs=all` (cyclic rungs collapse onto one cycle); `line` (D-024, fast rungs drift to the boundary) |
| stationary mode | `data.stationary_constant=true` (rev3) | false = legacy OU drift at unit variance |
| slowness gap | L2 | L1 |
| optimizer / regime | Adam, pair-minibatch (B random consecutive pairs) | pure single-pair online |

## Reproducibility & I/O (D-014)

- All hyperparameters in `./config/config.yaml` (Hydra). A run = its composed config.
- Run dir under `./.tmps/runs/<timestamp>/`; artifacts + figures written there.
- **ONE TensorBoard dir per run:** both `train` and `evaluate` write into `<run>/tb/`,
  separated by tag namespace (train `loss/*`,`var/*`,`energy/*`; eval `masks/*`,`S/*`,
  `collective/*`). Never a second event dir -- two per run is unreadable for a human.
- Seeds fixed (numpy generator for data, `torch.manual_seed` for model).
- Artifacts (masks, final S, metrics.json) → `.tmps/`. Promotion to `./results/` needs user OK.

## TensorBoard log schema

- train scalars: `loss/total`, `loss/slow`, `loss/white`, `loss/var_hinge`, `loss/energy`,
  `loss/recon`, `loss/size`, `loss/band`, `loss/line`, `loss/mem`, `loss/level`,
  `band/flat_ratio_max`, `band/frac_{min,mean}`,
  `band/line_{min,max}`, `band/rho_lag_max`, `energy/density_{min,max}`, `mask/count_{mean,min,max}`,
  `var/{mean,min,max}`, `slow_per_ch/{min,max}` (emergent slow↔fast spread), `grad_norm`, `temp`.
- eval (SAME `tb/` dir): `masks/trained`, `masks/init` (images), `S/ch*` (the K series),
  `collective/*`.
- Figures (`src/probes/plots.py` → `<run>/figs/`) are the PRIMARY readout, not the scalars:
  `masks.png`, `masks_on_energy.png`, `features.png`, `feature_var.png`. Always look at them.

## Edge cases / failure modes

- **Collapse (M→0, s→0):** the FAILURE MODE OF `whitening: soft` — observed 2026-08-26: masks go
  to 0, `var→2.5e-6`, `white→K=16` (i.e. `‖0−I‖_F²`), `slow→3e-7`. The gap term is quadratic in the
  mask scale while the penalty is bounded by `K`, so shrinking always wins. `whitening: hard` is
  the fix: the loss is measured on the whitened `z`, which is scale-free, so collapse buys nothing.
- **Ill-conditioned whitening:** if `cond` explodes, two channels have become near-duplicates;
  `white_eps` floors the eigenvalue and `grad_clip` caps the spike. Persistent high `cond` means K
  exceeds the usable slow subspace.
- **Batch < K:** covariance rank-deficient → require `B ≥ 2K` (default B=128 ≫ K=16).
- **Anneal too fast:** masks freeze before organizing → keep temp1 not too small; it's tunable.
- **NaN:** guard denom `clamp_min(1e-6)`; assert finite S each log step.

## Verification

```
conda run -n oceanai python -m src.train.train            # ~5000 steps, seconds on an H100
conda run -n oceanai python -m src.probes.evaluate        # reads the newest .tmps/runs/*
```

Healthy run (measured 2026-08-26, K=16): `offcorr2` falls 0.99 → ~0.27 and flattens by ~2500
steps; `var` stays O(1e2..1e5) (NOT falling toward 0); `hinge` stays 0.0000; `slow_ch` spreads by
>20× between the slowest and fastest channel. The probe should then report effective rank ≳ 4,
mask IoU ≲ 0.2, and several channels with |corr| > 0.8 against a hidden mode.

Known-bad signatures: `var` decaying monotonically past ~1e0 → a scale-invariance leak in the
loss; `offcorr2 → 0` with `var → 0` → total mask death; effective rank ≈ 1 → the constraint is not
differentiable.

## Known structural limits (NOT bugs — see D-018, findings F-4/F-5)

- Decorrelation saturates at off-diag corr² ≈ 0.244 / effective rank ≈ 4 of 16 because the masks
  are NON-NEGATIVE and cannot cancel a shared large-scale mode. λ_white is irrelevant (identical
  at 1, 20, 100, 500). Largely a GENERATOR artifact: every `φ_k` is a positive bump, so 100% of
  cell pairs are positively correlated (findings F-4 caveat).
- **The family population is NOT a function of K** (D-024, measured 3/0/1 -> 36/10/18 over
  K=4..64, slow-dominated throughout). A shared objective gives K copies of one optimum; the
  timescale must be ASSIGNED per channel. Same shape of problem as the mask-size collapse (D-020).
- Minimizing MEAN slowness discards the chaotic modes — FIXED by `lambda_recon` (D-019): chaotic
  ensemble R² 0.01–0.03 → 0.79–1.00. The energy floor alone does NOT fix it (0.01–0.06).
- `lambda_slow` should stay SMALL. At 10 it collapses diversity (eff rank 2.95, spread 31× → 2.3×).
- Masks shrink from the ~50%-of-domain init to a median of ~28 cells (0.3%) unless `lambda_size`
  holds them on the ladder; `L_energy` accelerates the shrinkage because mean pairwise covariance
  is maximized by a tiny coherent patch (D-020).
