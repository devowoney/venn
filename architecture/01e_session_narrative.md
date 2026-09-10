# Session narrative, 2026-08-31 → 2026-09-01 — the labeller was the bug; the ladder is the lever

> Chronological record including dead ends and my own corrections. Three parts, in order: the
> timescale ladder (D-024), the user's "stationary = constant" ruling (D-025), and the
> global-structure / observables reframe (D-026). Later parts CORRECT earlier ones — read to the end
> before acting on a number from part 1.
>
> Tracked copy of `memory/sessions/2026-08-31_stationary-observer-and-timescale-ladder.md`
> (`memory/` is gitignored, so the original does not travel with the branch).
> Code: git worktree `.claude/worktrees/stationary-observer`, branch `worktree-stationary-observer`.
> Nothing committed — git is the user's.

## The ask

> "From built encoder (observer) I want to develope to capture stationary mode too. Deepen the
> kernel increase some visibility of slow mode, but still not enough to say that the kernel number
> is the lever to detect every dynamical mode."

Two claims to test: (1) the observer does not capture the stationary mode; (2) K is not the lever.
Claim 2 held. Claim 1 turned out to be false — and the reason it looked true is the interesting part.

## What I did first: read the K sweep instead of trusting the summary

The user had left five fresh runs, `.tmps/runs/20260831_0736{32,50,40,32,23}` = K 4/8/16/32/64.
Every one reported `population.stationary = 0`. But in the SAME json,
`mode_recovery.m0(stationary).max_abs_corr` read 0.92 / 0.62 / 0.93 / 0.82 with ensemble R² ≈ 1.00,
and at K=16 channel 0 had `best_corr` 0.928 against the stationary mode with `tau_e` 120.

A readout claiming zero stationary channels while also reporting a channel that tracks the
stationary mode at 0.93 is self-contradictory. So I checked the labeller against the answer key
instead of the model — and the smoking gun was already printed in every eval.json we had ever
written: `truth[0].family = 'stationary'`, `truth[0].label = 'cyclic'`, and
`labeller_truth_accuracy = 0.7`. The probe misclassified the GENERATOR'S OWN stationary series.

Cause: `label_family` returned `cyclic` whenever the single largest FFT bin held >10% of the power.
A red spectrum piles its power in the lowest resolvable bin, so an OU process always presents as "a
clean oscillation of period T/2". The label `stationary` was unreachable for anything red — which is
every stationary series there is. **Every `population.stationary = 0` in this project's history
(D-018, F-5, the module-2 entry) measured the probe.** F-11.

Note for next time: `labeller_truth_accuracy` was 0.7 in the output for five sessions and neither
the user nor I treated it as a blocker. A readout should be validated against ground truth before
it is allowed to judge a model. Third instance of this lesson after F-6 (figures caught the dead
corners) and F-9 (var_ratio caught the mean-collapse ranking).

## The fix: `src/probes/family.py`

Calibrated descriptors on the ten hidden truth modes rather than guessing thresholds. The separator
is line WIDTH, not peak height:

| | line_frac (power within ±2 bins of peak) | tau_e |
|---|---|---|
| 3 cyclic | 0.94 / 0.95 / 0.94 | 12-57 |
| 1 stationary | 0.67 | 106 |
| 6 chaotic | 0.10-0.46 | 3-7 |

Rule: cyclic if `line_frac > 0.80` and `k_pk >= 4`; else stationary if `tau_e >= 20`; else chaotic.
Both thresholds mid-gap with >3x margin. 10/10 on seeds 0-4 (was 7/10). One module, imported by
`evaluate.py`, so the two copies cannot drift again.

**Re-scoring the archived K sweep with it inverted the sign of the problem:** 3/0/1, 4/2/2, 12/0/4,
22/3/7, 36/10/18 for K=4..64. The encoder does not under-produce slow channels — it produces
almost nothing else (12 of 16 red at K=16, ZERO clean cyclic). F-5's mean-slowness pathology had
been hiding behind a labeller that shouted "cyclic" at every red spectrum.

## Why K cannot be the lever (F-12)

`L_slow = mean_i var(Δs_i)/var(s_i)` is ONE objective shared by all K channels. They compete for
the same globally-slowest content, so channel K+1 buys a near-duplicate of the dominant regime, not
a new timescale. The measured proportions barely move from K=4 to K=64. The user's intuition was
right, and now there is a mechanism and a number behind it.

## The build: D-024 spectral band ladder

Same move D-020 made for mask size: stop asking every channel for the same thing, ASSIGN each a
target and penalize only outside it. `src/train/spectral.py`:

- `rung_roles(K, pop_target)` — 1/3/6 scaled to K (K=16 -> 2 slow / 5 cyclic / 9 fast);
- `L_band` — in-band power fraction per assigned rFFT band;
- `L_line` — line-vs-hump: cyclic rungs must BE a line, slow/fast rungs must NOT;
- `L_mem` — slow rungs only, lag-64 autocorrelation ≥ 0.6.

Computed on random contiguous WINDOWS sliced from `enc(field)`, because a lag-1 pair minibatch says
nothing about a period-300 cycle. Encoding the whole series each step is cheap (one einsum over
T×8192), and the stochasticity D-011 wants now comes from the window starts.

### Dead ends and corrections along the way

1. **Per-channel exponential-memory target (`rho_i(L) = exp(-L/tau_i)`), abandoned before coding.**
   Wrong for a cyclic channel: its ACF oscillates and its long-lag gap reaches 4, twice the
   exponential maximum, so the term would have actively suppressed the cyclic family it existed to
   create. Recorded in the module docstring so it does not get re-tried.
2. **Partitioned frequency bands, caught by the gradient-norm measurement.** My first `band_plan`
   tiled the cyclic rungs over [24, slow_period_min=128], which puts the testbed's period-140 and
   period-300 cycles inside the SLOW band and out of reach of every cyclic rung. The families are
   interleaved in frequency (cycles 60/140/300 vs OU tau=200), so a partition is wrong in principle.
   Added `cyclic_period_max=320` and let the bands OVERLAP — masks were never required to be
   disjoint (D-013), and it is `L_line` that separates a slow line from a slow hump.
3. **Band alone left the slow rungs on the wrong content.** Run `20260831_091525`: the two slow
   rungs took the `stationary` LABEL but had tau_e 36/27 against the truth's 106, and their
   best-matching hidden mode was a CHAOTIC one at corr 0.50/0.33 — while two cyclic rungs picked up
   the actual OU mode. "Power at periods ≥128" is too weak; any large red-ish footprint satisfies
   it. Added `L_mem`: at lag 64, OU(tau=200) gives rho=+0.73 while cycles of period 300/140 give
   +0.23/-0.96, so it is a sharp stationary selector. Its one blind spot — a cycle whose period
   divides the lag aliases back to rho≈1 — is covered by `line_cap` on the same rung.
4. **`lambda_slow=0` ablation: slightly worse** (obedience 0.69, population 7/0/9). Mean slowness
   stays at 1.0 as a weak tiebreak, consistent with F-7.

Weights gradient-matched to `lambda_white` at init per the D-019 convention, not value-matched:
measured |g_white|=3.9e-3, |g_band|=7.8e-4, |g_line|=1.0e-3, |g_mem|=9.5e-3 -> `lambda_band=5.0`,
`lambda_line=4.0`, `lambda_mem=0.4`.

## Result (K=16, 5000 steps, seed 0, all else equal)

| | baseline `091454` | +band/line `091525` | +L_mem `091908` |
|---|---|---|---|
| population (s/c/ch) | 10/1/5 | 6/1/9 | 5/1/10 |
| role obedience | n/a | 0.75 | 0.69 |
| slow rung vs true stationary | (unassigned) | corr 0.50 to a chaotic mode | **corr 0.92, tau_e 87** |
| worst hidden-mode maxcorr | 0.25 | 0.65 | 0.61 |
| effective rank /16 | 9.23 | 10.08 | 9.56 |
| balanced recon R² | 0.9891 | 0.9891 | 0.9891 |

The deliverable on the user's ask: a channel DEDICATED to the stationary mode, holding it at corr
0.92 with honest memory (tau_e 87 vs the truth's 106). Worth noting the baseline's ch0 also reached
corr 0.93 — but at gap1 0.0007 against the truth's 0.0112, i.e. a 16x over-smoothed basin average
that happens to correlate, not the OU's own dynamics. The ladder channel is the better observer even
where the correlation is a hair lower.

Also: the fast rungs obey nearly perfectly (gap1 0.10-0.13 vs the truth's chaotic 0.07-0.20; the
baseline's "chaotic" channels sat at 0.02-0.03 and were not really chaotic), and no hidden mode is
abandoned any more (baseline left two chaotic modes at |corr| 0.25).

## What the figures showed that the metrics did not

`figs/features.png`, both runs. The ladder bank is visibly stratified — smooth wandering at ch0-6, a
textbook sinusoid at ch4, broadband hash at ch7-15 — while the baseline's families are interleaved
down the channel index in no order. The same figure is what shows the failure honestly: cyclic rungs
2/3/5/6 wander instead of oscillating. No scalar in the run said that as clearly.

## Open, in priority order

1. **The cyclic rungs are the real remaining gap** — 1 of 5 reaches a clean line (line_frac 0.92;
   the rest 0.52-0.60 and get labelled stationary). A non-negative regional mask sums everything
   under its footprint and the basin-scale red content leaks into every region, so a channel cannot
   cancel its own slow background. This is F-4's non-negativity limit resurfacing in the time
   domain. Levers: (a) a temporal high-pass in the definition of `s_i` for cyclic/fast rungs —
   NEEDS A USER DECISION, it changes D-005's observer semantics; (b) more sign structure in the
   generator's mid-scale patterns; (c) accept mixed channels and let module 2 absorb it.
2. **De-align the size and timescale ladders.** The one slow rung that failed is the
   largest-footprint one (2275 cells) — the D-021 `L_energy`-vs-biggest-rung conflict. Cheap test.
3. **A learned classifier is now trainable** against a labeller that actually works.
4. **Re-run modules 2/3** on a ladder encoder; the per-family channels are what module 2's
   per-family interpretation was blocked on.

---

# 2026-09-01 (same session, continued) — the user corrected the DEFINITION, and it invalidated my taxonomy

## The correction

> "Definitely we have misunderstanding, population baseline is not 10/1/5. The reference of
> stationary is not clear enough. for me, at least ch 2, 4, 5, 6, 10 S modes are clearly cyclic.
> Return to begining, when we decided 'stationary signal', I refered that some constant signal.
> Actually, your injected mode 0 in pseudo SSH-SST field, it is not stationary for me. it should be
> more flat. Large means maybe the basin-sacle slow signal, but the mode is not really constant."

Two claims. I checked both instead of assuming either.

**Claim A — mode 0 is not stationary: CORRECT, and structurally so.** `_ou_series` ends with
`_standardize`, so the "stationary" mode was emitted at UNIT VARIANCE — as much temporal energy as
the three sinusoids and the six Lorenz channels. It was a tau=200 wanderer. The testbed contained
no flat mode at all, so every "the encoder cannot capture the stationary mode" conclusion in this
project's history (mine from yesterday included) was measured against a mode that was not
stationary. My F-11 labeller had then *defined* stationary as "long memory, no spectral line" —
i.e. slow red noise — which is a different taxonomy from the project's, and that is where 10/1/5
came from. Those counts are slow/cyclic/fast, not family counts. F-13 records the correction.

**Claim B — ch2/4/5/6/10 are clearly cyclic: partly.** Measured the ACF recurrence (a cycle's
autocorrelation comes back UP after its trough; red noise does not):

| ch | ACF trough | rebound | verdict |
|---|---|---|---|
| 2 | −0.83 | **+0.74** | genuinely oscillatory — user is right |
| 4 | −0.15 | −0.04 | no recurrence |
| 5 | −0.31 | −0.05 | no recurrence |
| 6 | −0.40 | −0.31 | no recurrence |
| 10 | −0.61 | −0.47 | no rebound |

The three true cyclic modes sit at trough −0.92…−0.98 with rebound +0.85…+0.97. So ch4/5/6/10 are
slow wandering with visible wiggles — red noise reads as "waves" to the eye. I said so rather than
agreeing wholesale, while conceding ch2 and, more importantly, claim A.

Then I asked, rather than guessed, because the taxonomy is the user's to define. Their rulings:
mode 0 becomes exactly constant; slow drift is classed **cyclic**; and the encoder gets a
mean/level term for the stationary rungs.

## What changed (three places, because the definition touches all of them)

1. **Generator rev3** (`stationary_constant: true`). `amp_0[t] = stationary_amp`, un-standardized.
   Two traps found while building it:
   - `_ar1_response` would have destroyed the mode in SST: a constant has zero variance, and that
     function ends in `_standardize`, so SST would have received exactly 0. Also it would have added
     a startup transient to something that is meant to be transient-free. Now bypassed for constant
     modes — an AR1's steady state under constant forcing IS that constant.
   - Checked (rather than assumed) that field standardization preserves it: `(fv - fv.mean())`
     removes a GLOBAL SCALAR mean, not a per-cell one. Verified the field's time-mean map correlates
     **1.000** with `phi_0`, rms 0.818 against field std 1.0.
2. **Labeller.** Flatness tested FIRST, via `amp_ratio = std/|mean|` on the RAW series. Critically,
   flatness is *undecidable* from a standardized series — standardizing a constant divides by ~0 —
   so `amp_ratio=None` makes the flat test SKIP rather than guess. 10/10 on the hidden truth over
   seeds 0–4, population 1/3/6.
3. **Encoder — `L_level`.** The real structural finding: a constant lives entirely in the channel
   MEAN, and every term in the v0 loss uses the centered channel `sc = s_t - mu`. Slowness,
   whitening, coverage and the energy floor are all anomaly-only, so a static signal is invisible to
   them; and `l_var = relu(1 - std)^2` plus `l_energy` actively PENALIZE a flat channel. A stationary
   channel was not merely unrewarded, it was forbidden. So the slow rungs now get
   `relu(flat_target - level/(level+fluct))^2` and are exempted from `l_var` / `l_energy` / `L_band` /
   `L_line` — every term that presumes fluctuation. `L_level` also replaces the on-signal protection
   `l_energy` was providing (F-6).

## Mistake caught before it cost a run

I first set `flat_target = 0.8`. That corresponds to `amp_ratio = 0.25` — a channel that still
visibly moves — so the hinge would have gone quiet *before* the channel could pass the labeller's
0.05 cut. The objective and the readout have to agree on where the bar is: `r = 1/(1 + amp_ratio)`,
so flat (`< 0.05`) means `r > 0.952`, hence 0.95.

## Result (rev3 field, K=16, 5000 steps, seed 0)

| | baseline `20260901_080438` | ladder + `L_level` `20260901_080506` |
|---|---|---|
| population (s/c/ch) | 0/14/2 | **2/5/9** (target proportion 1.6/4.8/9.6) |
| role obedience | n/a | **1.00** |
| flattest channel `amp_ratio` | 0.778 — *still moving* | **0.023** (second at 0.049) |
| flat channels (< 0.05) | 0 | 2 |
| cyclic rungs at line% ≥ 70 | — | 5 of 5 |
| balanced recon R² | 0.7763 | 0.7765 |

The baseline provably cannot produce a stationary observer — nothing in it gets near flat. rev3
scores 0.78 balanced recon where rev2 scored 0.99; that is the new field being harder (a large
static component anomaly-decoding cannot explain), and it hits both runs equally.

## A figure was lying again (second time on this project, cf. F-6)

`plots.py` standardized every channel before plotting — dividing out the std, the very quantity that
defines flatness. A channel fluctuating at 2% of its level was drawn as dynamic as a sinusoid, which
is exactly how the ladder run first read to me as "ch0/ch1 are oscillating". Now scaled by RMS about
zero with `amp_ratio` printed per lane: ch0/ch1 draw as flat lines, ch2–6 as clean cycles, ch7–15 as
broadband hash. **Check what a plot normalizes by before trusting it.**

## Null result

De-aligning the size and timescale ladders (`slow_size_frac=0.08`, run `20260901_080655`) to un-pin
the flat rungs from the largest footprints: footprint halved 2590 → 1318 cells, alignment with the
injected pattern UNCHANGED (|corr| 0.27 either way), role obedience fell 1.00 → 0.94. So the flat
channels' diffuseness is not the size ladder's fault. Default back to 0.0, reasoning kept in the
config comment so it is not re-tried blind.

## Open

1. **Does the observer need to ISOLATE the static pattern, or just report a stable level?** The flat
   channels are flat (amp_ratio 0.023) but align with `phi_0` at only |corr| 0.27 — settling on the
   pattern's negative lobe or a diffuse static average. Needs a user call.
2. **Module 2 has never seen a constant channel** and normalizes per channel — it will divide by
   ~0. Must be handled before re-running the predictor on a rev3 encoder.
3. `slow_period_min` / `cyclic_period_max` were tuned when the slow band was meant to hold the OU
   drift. The slow band now holds no stationary mode at all, so those knobs want revisiting.
4. A learned classifier is now worth building — the labeller is finally trustworthy.

---

# 2026-09-01 (cont.) — "let's slow down": two frame corrections

The user stopped the build to fix the frame, and both corrections landed on things I had just built.

## Correction 1 — classify by GLOBAL STRUCTURE, not the residue

> "for me ch02, 03, 04, 05 and 06 are cyclic. Even 09 and 11 have cyclic characteristics. Ok I am
> agree with that there are not 100 percent cyclic like sin cosin signal. But we should classifier
> with there global strcutre not their small fluctuation and drifts."

My ACF-recurrence test read the RAW series, where fast noise and a wandering baseline dominate the
autocorrelation — so it answered a question about the *residue*. That is why it called visibly
oscillating channels non-cyclic, and it is a better explanation of the earlier disagreement than
"the user is reading waves into red noise".

Replaced by an explicit 4-way decomposition (D-026): `level` + `trend` (bins 1–3: fewer than 4
cycles in the record, where a drift and a "cycle" are indistinguishable — so this IS the drift band)
+ `osc` (the dominant peak's HALF-POWER band, which adapts — ~1 bin for a tone, wide for a
quasi-periodic hump; a fixed ±2 window would have scored a broad hump as residual and called a real
oscillation chaotic) + `residual`. Label = flat gate, then `trend + osc` vs `residual`.

**A tidier variant I rejected:** plain argmax over all four shares including the level. `level² >
var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its level would score
"stationary". Counter-example already on the record: ch14 of run 20260901_080655, amp_ratio 0.526,
plainly broadband. So the level GATES and the structure votes.

**Answering the specific channels** (post-hoc — a label is a readout, no retraining needed):
ch2–6 come out cyclic at osc share 0.70–0.93, agreeing with the user. ch9/ch11 stay chaotic but at
osc share 0.41/0.32 against residual 0.59/0.67 — genuine oscillatory character, broadband still
dominant. That is the honest version of "has cyclic characteristics but is not 100% cyclic"; forcing
them to cyclic would contradict the Lorenz-derived mode they track and cost truth accuracy.

Re-scored: rev3 truth **10/10**; rev2 truth 9/10 where the single "miss" is the old OU mode 0 now
labelled **cyclic** (trend share 0.63) — D-025 working as ruled, not a bug; rev3 ladder 2/5/9 with
role obedience 1.00; rev3 baseline 0/11/5.

## Correction 2 — the observer is not a mode-recovery device

> "the stationary mode can live in the ouside of the signal. Hidden mode is just hidden mode. […] the
> Observer should decide where we should see where we should focus on. That becomes learned kernel.
> if the kernel captures stationary signal, it could be a single mode […] But, it also can be a
> persist phenomen or the combination of multi signals which shows barely varying value."

This retires a metric I had introduced the same day. I had been reading the flat channel's |corr|
0.27 with the injected φ₀ as a weakness and had even run an experiment to "fix" it. Under the
correct frame that number is a diagnostic: the family belongs to the OBSERVABLE the kernel builds,
and a flat channel may be a single mode, a persistent phenomenon, or a cancelling combination.

Changed in the probe: `mask_align_with_pattern` and `mode_recovery` are marked DIAGNOSTIC;
`population` is scored against a DESIGN target (the rung ladder's mix), with
`target_is_design_not_recovery: true` written into eval.json so the intent survives.

**New diagnostic for the "or a combination" case.** Project every hidden mode onto the flat
channel's footprint, compare the net fluctuation against the independent-addition baseline
`sqrt(Σ sd_k²)`: < 0.7 = destructive interference beyond chance (a cancelling combination), ~1.0 =
incoherent addition (a quiet footprint / persistent phenomenon), > 1.3 = reinforcement.

Ladder run ch1: 3 hidden modes reach the footprint, index **1.01** — so this stationary observer is
a quiet footprint, not a cancellation.

**Mistake caught inside this metric.** My first version normalized by the plain SUM of the parts and
read 0.634, which looks like partial cancellation — but three independent parts already give 0.577
of the sum, so 0.634 is almost exactly what chance predicts. The wrong baseline would have
manufactured a finding. Normalizing against independence is the fix.

## Asked before building

Four questions, all answered by the user: explicit 4-part decomposition; NO persistence check
(flatness over the record is enough); cancellation allowed but not encouraged (no term rewards it);
1/3/6 is a design target for the bank, not a recovery score. Worth keeping that habit — three of
the four answers ruled out work I would otherwise have done.

## Open

1. Harmonic folding in `decompose`: a non-sinusoidal periodic signal leaks into `residual` and biases
   the vote toward chaotic.
2. Module 2 normalizes per channel and will divide by ~0 on a constant channel.
3. A learned classifier is now worth building — the labeller finally matches the project's own
   definitions.
4. `slow_period_min` / `cyclic_period_max` were tuned when the slow band was meant to hold the OU
   drift; the slow band now holds no stationary mode at all.
