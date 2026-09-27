# Decisions — D-024, D-025, D-026

Numbering continues `memory/decisions.md`. D-025 and D-026 are **user rulings**; D-024 is mine,
derived from measurement.

---

## D-024 — Spectral band ladder: the TIMESCALE is assigned per channel, not emergent

**Date:** 2026-08-31. **Status:** BUILT and measured.

**Context.** CLAUDE.md's v0 core expects the stationary/cyclic/chaotic spread across kernel index to
EMERGE from slowness + decorrelation, with the classifier only labelling it (D-011). Two
measurements say that mechanism does not deliver: the family population is slow-dominated at every
K from 4 to 64 (F-12), and the emergent spread was never observable anyway because the labeller
could not emit "stationary" (F-11).

**Decision.** Keep everything in D-015/D-019/D-020 and ADD a per-channel timescale assignment — the
temporal twin of the D-020 size ladder. Channels get roles in the `pop_target` proportion (1/3/6
scaled to K → 2 slow / 5 cyclic / 9 fast at K=16) and three hinge terms:

- `L_band` — in-band power fraction, per-channel assigned rFFT band;
- `L_line` — line-vs-hump shape: cyclic rungs must BE a spectral line, slow and fast rungs must NOT;
- `L_mem` — slow rungs only, lag-`mem_lag` autocorrelation ≥ `mem_target`. *(Superseded as the
  default by D-025's `L_level`; retained as `slow_objective: memory`.)*

All are ratios of powers (scale-free in value AND gradient — the D-017 rule) and all are one-sided
hinges (silent once satisfied, so they guide rather than pin — D-020's philosophy).

**Weights are GRADIENT-matched to `lambda_white` at init**, per the D-019 convention, never
value-matched: measured `|g_white|=3.9e-3`, `|g_band|=7.8e-4`, `|g_line|=1.0e-3`, `|g_mem|=9.5e-3`
→ `lambda_band=5.0`, `lambda_line=4.0`, `lambda_mem=0.4`.

**Why band AND line, not just a band.** The families overlap in FREQUENCY on this testbed (cycles
60/140/300, OU τ=200), so a frequency band alone cannot separate a slow drift from a slow cycle. The
cyclic band range is therefore deliberately allowed to OVERLAP the slow band
(`cyclic_period_max=320` > `slow_period_min=128`); masks were never required to be disjoint (D-013).

**Why windows, not pairs.** A lag-1 pair minibatch says nothing about a period-300 cycle. The
periodogram needs contiguous time, so the ladder slices `n_win` random windows out of `enc(field)`
(one einsum over T×8192 cells — cheap enough to redo every step). The stochasticity D-011 asks for
now comes from the window starts rather than the pair index.

**Rejected alternative — do not re-try.** Matching a per-channel target autocorrelation
`exp(-L/tau_i)`. Wrong for a cyclic channel, whose ACF oscillates and whose long-lag gap reaches 4 —
twice the exponential maximum — so it would have actively suppressed the cyclic family it was meant
to create.

**Consequence for the framework.** The family spread is now DESIGNED (a rung layout), not emergent.
This narrows D-011's "emergent diversity" claim: decorrelation diversifies WHERE channels look, but
not WHICH dynamics they hold. The labeller stays post-hoc and reversible (D-004) — the ladder shapes
where a channel looks in the spectrum, it never asserts a family. `role_obedience` is the honest
check on whether the assignment took.

---

## D-025 — STATIONARY = CONSTANT (supersedes the "slow" reading in D-024/F-11)

**Date:** 2026-09-01. **Status:** BUILT and measured. **Source:** direct user ruling.

**The ruling.** A stationary signal is a CONSTANT one — flat in time. The generator's mode 0 (an OU
drift at τ=200, standardized to unit variance) was never stationary, and a slow red DRIFT belongs to
the `cyclic` family, because it is not constant. This supersedes the definition I used in D-024 and
F-11, where "stationary" meant long-memory red noise. **Every population figure reported under that
reading is a slow/cyclic/fast count, not a family count.**

**Three changes, because the definition touches three layers:**

1. **Generator rev3** (`data.stationary_constant: true`): mode 0 is a static pattern with a flat,
   un-standardized amplitude, living in the field's time mean. Verified std `0.000e+00`, and the
   field's time-mean map correlates **1.000** with the mode-0 pattern. `GenConfig` default stays
   `False` so old runs reproduce.
   - *Detail that matters:* a constant must BYPASS `_ar1_response` for SST. An AR1's steady state
     under constant forcing IS that constant, so routing it through would add a startup transient —
     and that function's closing `_standardize` divides a zero-variance series by ~0, which would
     have deleted the mode from SST entirely.
   - *Checked, not assumed:* field standardization `(fv - fv.mean())/fv.std()` removes a GLOBAL
     SCALAR mean, not a per-cell one, so a spatially structured static pattern survives.
2. **Labeller** (`src/probes/family.py`): flatness is tested FIRST, via `amp_ratio = std/|mean|`
   measured on the RAW series. Flatness is undecidable from a standardized series (standardizing a
   constant divides by ~0), so with `amp_ratio=None` the test is SKIPPED rather than guessed.
3. **Encoder** (`level_term` in `src/train/spectral.py`): the stationary rungs are asked for
   `level/(level+fluct) ≥ flat_target` and are EXEMPTED from `l_var`, `l_energy`, `L_band` and
   `L_line` — every term that presumes a channel fluctuates. `lambda_level=0.13`, gradient-matched.

**`flat_target` must agree with the labeller's cut.** `r = 1/(1 + amp_ratio)`, and flat means
`amp_ratio < 0.05`, so `r > 0.952` → `flat_target = 0.95`. A first attempt at 0.8 corresponded to
`amp_ratio` 0.25 — the hinge would have gone quiet before the channel could qualify. **The objective
and the readout have to agree on where the bar is.**

**Why the encoder could not have found a constant before.** A constant lives ENTIRELY in the channel
mean, and every v0 loss term uses the CENTERED channel `sc = s_t - mu`: slowness, whitening,
coverage and the energy floor are all anomaly-only. Two terms went further and actively forbade a
flat channel — `l_var = relu(1 - std)²` demands unit normalized std, and `l_energy` demands the
selected cells be energetic in TIME. A stationary channel was not merely unrewarded, it was
penalized. `L_level` also replaces the on-signal protection `l_energy` was providing (F-6).

**Consequence for the architecture.** The observer must report two kinds of thing: a level (static
family) and an anomaly (everything else). D-005 selection semantics are unchanged —
`s_i = <mask_i, field>` already carries both — but the LOSS had to stop being anomaly-only. This is
the minimal version: per-rung exemptions rather than a full mean/anomaly split of `s_i`, which
remains available if modules 2/3 need the level separately.

---

## D-026 — Classify by global structure; families belong to the OBSERVABLES

**Date:** 2026-09-01. **Status:** BUILT (labeller + probe; no retraining needed). **Source:** user.

**Ruling 1 — classification follows a channel's GLOBAL STRUCTURE, not its small fluctuation and
drift.** A channel that oscillates under fast noise on a wandering baseline is CYCLIC. Implemented as
a 4-part decomposition (`family.decompose`): `level` + `trend` (bins 1–3) + `osc` (the dominant
peak's half-power band) + `residual`. Label = flat gate, then `trend + osc` vs `residual`. Drift
counts as structure, not noise, per D-025.

**Ruling 2 — the observer is not a mode-recovery device.** Hidden modes are just hidden modes, and in
a real ocean/atmosphere capturing them all is barely possible. What the kernel learns is WHERE TO
LOOK, and a family is a property of the OBSERVABLE it constructs. A stationary channel may be a
single static mode, a persistent phenomenon, or a combination of varying signals that barely moves.

**Consequences for scoring — this changes conclusions, not just code:**

- `population` is measured against a **DESIGN** target (the rung ladder's 1/3/6 proportion = the mix
  of observables we ask for), NOT the hidden mode counts. `target_is_design_not_recovery: true` is
  written into `eval.json` so the intent survives.
- `mode_recovery` (max |corr|, ensemble R²) and `mask_align_with_pattern` are **DIAGNOSTICS**.
  Judging a flat channel by its alignment with the injected pattern was the wrong frame; reported,
  not optimized.
- The success criterion for a stationary channel is **FLATNESS ALONE** (`amp_ratio < 0.05`). A
  persistence check across sub-windows was offered and the user chose against it.
- **Cancellation is allowed but NOT encouraged** — no term rewards it. `L_level` rewards flatness
  however the kernel achieves it. The new `cancellation_vs_independent` diagnostic reports which
  mechanism a given flat channel actually used.

**Design note — the level GATES, it does not join the argmax.** A plain argmax over all four shares
is tidier but wrong: `level² > var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its
level would score "stationary". Counter-example on the record: ch14 of run `20260901_080655`,
`amp_ratio` 0.526, plainly broadband chaotic.

**What D-026 does NOT change:** D-005 selection semantics, D-013 (masks are sensors, overlap is
fine), or the ladder mechanics of D-024/D-025. It changes the labeller and what counts as success.

---

## D-027 — `L_struct`: the fast rungs' shape is scored in the readout's own quantity

**Date:** 2026-09-10 → 2026-09-27. **Status:** BUILT and measured (5 seeds). **Source:** user asked to
refine the observer ("ambiguous signal cyclic or chaotic"); fast-rungs-only chosen by the user after
the first variant collapsed the cyclic rungs.

**Decision.** Add `L_struct = mean over FAST rungs of relu((trend+osc)/total − 0.30)²`, computed on the
full series exactly as `family.decompose` does. It replaces `L_line` on the fast rungs; the cyclic
rungs keep `L_line`. Config: `shape_objective: structure`, `struct_rungs: fast`, `struct_cap: 0.30`,
`lambda_struct: 2.3` (gradient-matched).

**Why.** `line_cap` measured silent on all nine fast rungs while six sat near the labeller's 0.5 cut
(F-15). The cap 0.30 is the physics' own bar: the hidden chaotic modes measure 0.17–0.35.

**Superseded variant — `struct_rungs: all`** (also `struct_target ≥ 0.85` on cyclic rungs). Rejected
after the figures showed every cyclic rung on one cycle (F-16). Kept as an ablation.

**`flat_target 0.95 → 0.98` — DECIDED by the user, 2026-09-27.** 0.95 sat exactly on the labeller's
cut, so the hinge faded just before the line. On the fixed default, 5 seeds: flat channels 0.8 → 1.6
per seed, obedience 0.93 → 0.97, ambiguous 2.2 → 1.2, cyclic coverage and recon unchanged.

**Not decided:** `lambda_struct = 10` (measured only with the rejected all-rungs variant; not
re-tested on the fast-only default).
