# Open questions, and alternatives already rejected

> Moved verbatim from `architecture/01d_open_and_rejected.md` on 2026-09-28 (architecture/ retired; path
> references updated). Encoder track.

## Open — in the order I would tackle them

0. **RESOLVED 2026-09-27 (F-17): averaging bug; distinct periods 1.4 → 2.6.** Original note: **The cyclic rungs collapse onto one cycle per seed (F-16).** With D-027
   fast-only, seed 0 keeps all five cyclic rungs on period 61; the baseline keeps 286 and 143. Cause
   unknown. Diagnose on seed 0 first: dominant period and in-band power of each cyclic rung over
   training, baseline vs fast-only, and whether the baseline's fast rungs carried slow content. Then
   choose: credit only the peak inside the rung's own octave; a stronger band for cyclic rungs; or a
   new rung layout (5 cyclic rungs for 3 cycles, and no cycle in the 24–40 octave — a user decision:
   fit the testbed, or stay generic for a real ocean).

0a. **(NEXT candidates) Remaining ambiguity = stationary rungs that miss the flat cut (5 cases) and
    contaminated slow cyclic rungs (6 cases); `L_band` loses on ch4/ch6 by layout.** See SOP 02 (`decisions.md`) D-027.

0b. **RESOLVED 2026-09-27: `flat_target 0.98` is the default** (flat 0.8 → 1.6 per seed).

0c. **Fast-rung redundancy.** Five of nine fast channels track hidden mode m6 on seed 0, with near-
    identical traces. Revisit after 0.

1. **The cyclic rungs are the biggest remaining gap (rev2 measurement; recheck on rev3).**
   On the rev2 field only 1 of 5 cyclic rungs reached a clean spectral line (`line_frac` 0.92; the
   rest 0.52–0.60). A non-negative regional mask sums everything under its footprint, and
   basin-scale slow content leaks into every region, so a channel cannot cancel its own red
   background — F-4's non-negativity limit resurfacing in the time domain. On the rev3 field the
   cyclic rungs do much better (osc share 0.70–0.93), so this may already be resolved by the testbed
   change; **verify before spending effort on it.**
   Levers if it persists: (a) a temporal high-pass in the definition of `s_i` for cyclic/fast rungs
   — NEEDS A USER DECISION, it changes D-005's observer semantics; (b) more sign structure in the
   generator's mid-scale patterns; (c) accept mixed channels and let module 2 absorb it.

2. **Module 2 will divide by ~0 on a constant channel.** `src/models/predictor.py` normalizes per
   channel; a flat channel has ≈zero variance. Must be handled before re-running the latent
   predictor on a rev3 ladder encoder. This also raises a design question: should the predictor
   forecast a constant channel at all, or should the level be passed through untouched?

3. **Harmonic folding in `family.decompose`.** A non-sinusoidal periodic signal puts power in
   harmonics, which currently land in `residual` and bias the vote toward chaotic. Folding harmonics
   of the dominant peak into `osc` would sharpen the cyclic/chaotic boundary. This is the most likely
   reason ch9/ch11 sit at osc share 0.41/0.32 rather than higher.

4. **A learned classifier is now worth building.** The labeller finally matches the project's own
   definitions and is validated at 10/10 against the truth, so it can supervise or be compared
   against a learned one. Still post-hoc and reversible (D-004).

5. **The ladder's period knobs were tuned for the wrong reading.** `slow_period_min=128` and
   `cyclic_period_max=320` were chosen when the slow band was meant to hold an OU drift. Under D-025
   the slow band holds no stationary mode at all (a constant is DC, and the DC bin is deliberately
   zeroed), so the slow rungs are driven entirely by `L_level`. Those two knobs, and whether slow
   rungs need a band at all, want revisiting.

6. **The flat channels are spatially diffuse.** They align with the injected pattern at only
   |corr| 0.27, settling on a diffuse static average or the pattern's negative lobe. Per D-026 this
   is NOT a failure — the observer chooses where to look, and a flat channel is a valid observable
   however it is composed. Listed here only because if we ever *do* want the observer to isolate a
   named structure, that is a new objective, not a bug fix.

7. **`.gitignore` carries unresolved merge-conflict markers** (`<<<<<<< HEAD`, `=======`,
   `>>>>>>> 7afcabf`). Pre-existing and harmless, worth cleaning during the merge.

## Rejected — measured, and recorded so they are not re-tried blind

| alternative | why rejected |
|---|---|
| **`L_struct` on the cyclic rungs too** (`struct_rungs: all`, `struct_target 0.85`) | `trend+osc` credits any clean peak, so every cyclic rung moved to the period-60 cycle (seed 0: 61×5 vs baseline 286/143/61/61/61); distinct periods 1.0 vs 2.4. The ambiguity count (3.0) looked good and hid it (F-16). |
| **Per-channel exponential-memory target** `rho_i(L) = exp(-L/tau_i)` as the ladder's mechanism | Wrong for a cyclic channel: its ACF oscillates and its long-lag gap reaches 4, twice the exponential maximum, so the term would actively suppress the cyclic family it was meant to create. Abandoned before coding, on the arithmetic. |
| **Partitioned frequency bands** (cyclic rungs tile only below `slow_period_min`) | The families are interleaved in frequency on this testbed (cycles 60/140/300 vs OU τ=200), so a partition puts periods 140 and 300 out of reach of every cyclic rung. Bands now deliberately OVERLAP; masks were never required to be disjoint (D-013). |
| **`lambda_slow = 0`** once the ladder exists | Slightly worse: role obedience 0.69 vs 0.75, population 7/0/9 vs 6/1/9 (runs `20260831_091605` vs `091525`). Mean slowness stays at 1.0 as a weak tiebreak. |
| **De-aligning the size and timescale ladders** (`slow_size_frac=0.08`, so flat rungs are not pinned to the largest footprints) | Footprint did halve (2590 → 1318 cells) but alignment with the constant mode's pattern did NOT improve (\|corr\| 0.27 either way) and role obedience fell 1.00 → 0.94, with one flat channel instead of two. The diffuseness is not caused by the size ladder. Knob kept at 0.0; the reasoning still applies to a testbed whose static pattern is small. |
| **`flat_target = 0.8`** | Corresponds to `amp_ratio` 0.25 — the hinge would go quiet before the channel passed the labeller's 0.05 cut. The objective and the readout must agree on where the bar is. Then 0.95, now 0.98 (past the cut, D-027). |
| **Plain argmax over all four structure shares, including the level** | `level² > var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its level would score "stationary". Counter-example: ch14 of `20260901_080655`, `amp_ratio` 0.526, plainly broadband. The level gates; the structure votes. |
| **Normalizing the cancellation diagnostic by the plain SUM of the parts** | Reads 0.634 for the flat channel, which looks like partial cancellation — but three independent parts already give 0.577 of the sum, so it is what chance predicts. Normalize against independence (`sqrt(Σ sd²)`) instead; the wrong baseline manufactures findings. |
| **A persistence check for stationarity** (level stable across sub-windows / held-out time) | Offered to the user, who chose against it: flatness over the record is enough. |
| **A term rewarding cancellation** (so the observer learns to build flat channels from cancelling parts) | Offered to the user, who chose "allowed, not encouraged": `L_level` rewards flatness however achieved, and the mechanism is reported as a diagnostic. |
| **Scoring the bank on hidden-mode recovery** | Rejected on principle by the user (D-026): families belong to the observables the kernels build. Recovery metrics are diagnostics. |

## Lessons that generalize beyond this branch

1. **Validate a readout against ground truth before it is allowed to judge a model** (F-11). A
   metric that cannot emit an outcome reads as evidence against the model.
   `src/probes/validate_labeller.py` makes this a one-command habit.
2. **Check what a figure normalizes by** (F-13). Standardizing hid exactly the property being
   measured. Second time on this project after F-6.
3. **A diagnostic needs a null model.** "Net/sum-of-parts = 0.63" means nothing without knowing that
   chance gives 0.58 (F-14).
4. **The objective and the readout must agree on thresholds**, or the loss goes quiet before the
   metric is satisfied (`flat_target` vs `FLAT_THR`).
5. **Match GRADIENTS, not loss values,** when weighting a new term (D-019 convention, held
   throughout: the values differ by ~300× and misrank the terms).
