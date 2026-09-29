# 2026-08-29 — Module 2: the latent predictor (observer → forecaster in the compressed domain)

## Where this session started

Modules 1 (selection-mask encoder) and its probes existed and worked; the generator was at rev2
(D-021). Modules 2 and 3 were unbuilt. The user's ask: *"build latent forecaster who predicts next
feature in compressed domain"* — i.e. D-012 module 2, on the frozen channels, nothing spatial.

## What got built

Following the Golden Rule, the SOP went first: `memory/sop/03_latent_predictor.md`, then

- `src/models/predictor.py` — three per-channel architectures behind one
  `init_state / step / warmup / rollout` interface, so training, evaluation and (later) module 3 are
  written once. Plus the closed-form AR(p) fit + rollout used as the baseline.
- `src/probes/pred_metrics.py` — shared metric code, imported by BOTH the trainer and the probe, so
  the number printed during training is literally the number the probe reports. (Deliberate: in the
  previous sessions a metric drift between train and eval would have been invisible.)
- `src/train/train_predictor.py` — Hydra job reading a frozen module-1 run.
- `src/probes/evaluate_predictor.py` — 4 figures + `predictor_eval.json`.
- `config/predictor.yaml` — composes `config.yaml` so the generator/encoder settings stay in one
  place; the predictor's own hydra run dir is `.tmps/runs_pred/` so it never creates stub dirs in
  `.tmps/runs/` that the "newest run" globs would pick up.

## The design argument I want to keep

The whole module rests on one claim: **a memoryless map `s(t) → s(t+1)` is structurally incapable
here.** `s_i` is a 1D projection of a higher-dimensional field, so a memoryless map is a *function*
and must return the same future every time the channel revisits a value — but the true trajectory
passes through that value at many phases. A circle needs a 2D state; a chaotic attractor needs a
Takens embedding. Hence state, either learned (GRU) or explicit (delay window). I did NOT offer a
memoryless variant, and I made the delay-window MLP an ablation so the claim gets measured.

Two more choices that turned out to matter more than expected:

- **Residual parameterization** (`z(t+1) = z(t) + dz`): persistence becomes the zero-output
  solution. Without it a slow channel burns its capacity relearning "copy the input".
- **Horizon curriculum** 1 → 16 with free-running rollout in the loss. See the measurement below —
  this one is load-bearing, not a nicety.

## What the numbers said (and where they lied)

| predictor | params | skill h1 | skill h64 | var ratio @h64 |
|---|---|---|---|---|
| GRU (default) | 14 864 | +0.420 | +0.246 | **0.89** |
| MLP delay window | 4 624 | +0.398 | +0.393 | 0.65 |
| learned linear AR(16) | 272 | +0.356 | **+0.425** | **0.18** |
| GRU, 1-step loss only | 14 864 | +0.475 | **−33.07** | **16.12** |
| closed-form AR(8) | — | **+0.556** | +0.300 | — |

**The finding I did not expect and would have got wrong from metrics alone:** the linear model has
the BEST long-lead MSE skill and is the WORST forecaster. Its rollout amplitude decays to ~1e-5 of
truth by lead 64 — it wins on MSE by becoming the climatological mean. The GRU has the worst h64
MSE and the only honest amplitude. `pred_rollout_stats.png` shows this in one glance; no scalar in
the metrics table except `var_ratio` does. This is the exact reason the standing "always visualize"
rule exists, and it is now also encoded as a metric.

The 1-step-only ablation is the mirror image: better at h1, and the rollout *explodes* (16x
amplitude). So the two failure directions — collapse and divergence — sit either side of the
curriculum, and the curriculum is what keeps the model between them.

**Closed-form AR(8) beats every trained model at 1 step.** Honest reading: the trained models pay
1-step accuracy for rollout stability (the H=1 GRU recovers most of the gap, +0.475 vs +0.556), and
these channels are largely linear-predictable at 1 step *because the encoder was optimized for
slowness*. The baseline stays in every report; it is a real competitor on this testbed.

**Where the hidden state does earn its keep:** the two chaotic-labelled channels, skill h1 +0.875 /
+0.909 for the GRU vs +0.135 / +0.160 for closed-form AR — a 5x gap. On the smooth cyclic channels
AR wins. The aggregate hides this because 14 of 16 channels carry the "cyclic" label.

## Caveat that limits every per-family number here

The family labels come from the module-1 heuristic labeller, which is known to be wrong
(population 0/14/2 against a 1/3/6 target — it calls almost everything cyclic). The per-family
aggregates inherit that error. A real classifier is still unbuilt, and it now blocks the
interpretation of module 2's results, not just module 1's.

## Dead ends / things I checked and discarded

- Considered scoring the model by MSE skill alone — abandoned once the linear ablation exposed the
  mean-collapse. `var_ratio` is now a first-class metric, and the SOP says rank by it FIRST.
- Considered letting the predictor see all K channels (a joint latent model). Rejected for now: it
  contradicts D-012's "per-channel 1D" and would make the family labels meaningless. Left in as the
  `per_channel: false` ablation switch rather than as a silent default.

## Environment / process notes

- The background-job harness required isolation, so all code was written in the git worktree
  `.claude/worktrees/latent-predictor` (branch `worktree-latent-predictor`) instead of the main
  checkout. Git remains the user's (standing rule) — nothing was committed to `main`.
- Ran in `oceanai` (CUDA available). Encoder run reused: `.tmps/runs/20260827_151413`, copied into
  the worktree's own `.tmps/` so nothing wrote into the user's checkout.
- `.gitignore` in the repo is committed WITH unresolved merge-conflict markers
  (`<<<<<<< HEAD` / `=======` / `>>>>>>> 7afcabf`). Harmless today (both blocks' patterns are read
  as literal patterns, so `.tmps/` is still ignored) but it should be cleaned up.

## Next

1. A real classifier — it now blocks module 2's interpretation as well as module 1's.
2. Module 3 (forecaster by gradient inversion, `x(t+1) = x(t) + M⁺(ŝ − M·x)`). The interface is
   ready: `predictor.rollout` returns exactly the `ŝ` it needs.
3. Re-check whether an encoder trained with the predictor in the loop (rather than frozen) changes
   the channels — currently module 1 optimizes slowness, which is *why* linear AR does so well.

## Late in the session — two standing rules restated by the user, and one decision handed back

The user restated two rules:

1. **Coding rule:** human-readable code, with comments that explain what a function does and why the
   logic is the way it is. The module-2 code was written this way; it is now recorded as a standing
   rule rather than a habit.
2. **Visualization rule:** a figure exists to **make a decision WITH the user**. Looking at the PNG
   and then reporting a settled conclusion is only half the job.

Applying rule 2 retroactively: I had set `arch: gru` as module 2's default on my own reading of
`pred_rollout_stats.png`. I took that fork back to the user with the figure and the numbers, and
**the user chose the delay-window MLP** (D-023) — tied at h1/h4 for 3x fewer parameters, with an
inspectable embedding dimension, and reading the MLP's long-lead MSE advantage as partly the same
mean-collapse artifact rather than as a real gain. `arch: gru` is retained as the choice when
rollout amplitude matters most for module 3.

Worth remembering for future sessions: my unilateral pick optimized for one metric I had just
promoted (`var_ratio`); the user weighted parsimony and interpretability against a tie. Neither
reading is available from the numbers alone, which is exactly the point of the rule.

## Final state after the D-023 switch

Ran the pipeline with NO overrides to prove the edited default works end to end
(`.tmps/runs/enc_default`): `arch=mlp`, 4 624 params, skill h1 +0.398 / h4 +0.561 / h64 +0.393,
var ratio 0.65 — reproduces the enc_mlp run exactly.

Also improved `fig_series`: the free run now defaults to the FULL validation window (536 steps
instead of 300), and truth is drawn first as a wide pale band so the thin prediction lines sit on
top of it — previously the 1-step line covered the truth completely and each panel looked like a
single curve.

**What the 16-channel figure actually shows (single free-running trajectory per channel):**
1-step prediction is excellent on all 16. The free run splits three ways —
sustained oscillation with slow phase drift (ch1, 2, 9, 11, 12, 14) and correct chaotic wander
(ch7, 8); decay to a flat line (ch0, 3, 5, 10, 13); and WRONG ATTRACTOR / instability (ch4 ramps
upward off the data range, ch6 and ch15 settle onto a small ripple at the wrong level ~1.7).
The ch4 divergence is a real defect: nothing in the residual MLP bounds the free-running state.

NOTE for reading the numbers: `var_ratio` in the table is the spread ACROSS 472 launches at a given
lead, not the amplitude within the single trajectory that is plotted. ch0 flatlines in the figure
yet scores 0.95 — the two are different quantities and both are worth keeping.

## A figure that was quietly lying, and the fix (user caught it)

The user asked why the 1-step line in `pred_series.png` is identical to ground truth at every point.
It is, and the reason is not that the model is good:

- mean lag-1 autocorrelation of the 16 channels: **0.985** (ch0: 0.99977)
- model 1-step RMSE: **0.108 = 2.5% of the plotted y-range**
- persistence 1-step RMSE (copy the previous value): **0.154 = 3.6% of the y-range**

Both are sub-pixel at full scale. A model that did nothing but repeat `z(t)` would have looked
*equally* perfect. The whole skill score (+0.398) lives inside that invisible 2.5%. So the solid
line was decorative and I had presented it as if it carried information.

**Fix (user chose the option): a zoomed inset per panel**, ~60 steps, auto-placed on the stretch
with the largest summed |dz| for that channel, containing truth / 1-step / **persistence**.
Persistence appears ONLY there — it is the thing the skill score is measured against, so if the
coloured line is not visibly closer to truth than the dotted one, the model earned nothing on that
channel. Insets are fully opaque (at alpha<1 the main trace ghosted through and read as a fourth
curve). The suptitle now states the autocorrelation caveat outright.

What the insets reveal, consistent with the per-channel skill: on ch7, ch8, ch15, ch6 the dotted
persistence line is visibly lagging the truth while the prediction sits on it (skill +0.84 / +0.86 /
+0.79 / +0.55); on ch0, ch11, ch12, ch13 the dotted line is nearly on top of the prediction —
persistence is already near-optimal and the model's edge is genuinely small (+0.41 / +0.43 / +0.27 /
+0.14).

**Generalizable lesson:** on a strongly autocorrelated series, a "prediction vs truth" time-series
plot is close to unfalsifiable — it looks perfect for any model, including a trivial one. Always
plot the prediction against the BASELINE at a scale where they can differ (inset, residual, or
increments), never the prediction against truth at full scale.

## 2026-08-31 (cont.) — the three-expert proposal, tested and rejected

User's design intuition: three dynamical families, so three predictor modules. Two things had to be
said before building it, and both mattered:

1. **Every channel already has its own parameters** (`per_channel: true`). So "3 modules" read as
   *3 weight sets shared across 16 channels* would be strictly LESS flexible than what exists. The
   idea only has content as three different STRUCTURES (inductive biases), not three weight sets.
2. **Hard routing conflicts with D-013.** A mask is a regional sensor reading several hidden modes
   at once, so channels are mixtures; "channel 7 IS chaotic" is an approximation. And the labeller
   is broken anyway (0/14/2), so routing would starve one expert and overload another.

I proposed the soft-mixture version (3 experts + learned per-channel gates, gates doubling as an
unsupervised classifier and yielding the 1/3/6 population histogram) — attractive because it would
retire the labeller blocker as a by-product. The user chose the disciplined path instead: **test
whether specialization is worth anything at all before building the gating machinery.** That was
the right call, because the answer is no (F-11), and the mixture would have been built on a false
premise.

Built `OUPredictor` and `OscillatorPredictor`, plus `src/probes/compare_predictors.py` (per-channel
comparison + figure). Result: the specialized experts win **0 of 16 channels**; specialization is
worth at most +0.009 at h1. Root cause is structural — AR(1) and polar AR(2) are linear functions
of the delay window, hence special cases of the MLP that already contains them.

The h64 panel initially *looked* like specialization helping (oracle +0.512 vs +0.393). It was the
F-9 trap for the third time: `osc` var ratio 0.00, `ou` 0.19 — collapsed to the mean. I added a
third panel to the comparison figure showing amplitude on a log axis so this is impossible to
misread. **Three separate times now, long-lead MSE has ranked a collapsed model first.** Any new
comparison in this project should ship the amplitude row from the start, not add it after being
fooled.

**Where this leaves module 2:** finished and not worth more architecture work. F-10b and F-11 agree
that architecture is not a lever here. The levers are module 1 (encoder objective — slowness is why
everything is near-linear and why persistence/AR are so strong) and the classifier.

## 2026-09-27 — module 2 re-run on the current encoder

Found on resuming: the user had committed the module-2 work (`9f2215a`) and rebased it onto four
encoder commits from the parallel `stationary-observer` session (`5e5c129` timescale ladder + family
observer; three today: F-15 fast rungs ambiguous, F-16 cyclic collapse onto period 60, F-17 that
collapse was an `L_line` averaging bug, plus `flat_target 0.95 → 0.98`). Every module-2 number had
therefore been measured on channels that no longer exist.

Re-ran the predictor unchanged on `flat98fix_seed0` (mlp, gru, linear) → F-18. The GRU now wins on
every axis, which removes D-023's basis; I left the default at `mlp` and put the decision to the user
with the figures, per the visualization rule.

Near-miss worth remembering: MLP's h64 lagged closed-form AR (+0.241 vs +0.386) and I had half-written
"AR beats the NNs at rollout" before checking AR's amplitude — 0.14, collapse, the fifth firing of
the same trap. The trainer never printed AR's amplitude. Any baseline shown next to a model needs
its amplitude reported too, not only its skill.

Same day, later: the user asked to SEE whether the forecast works better. Built
`src/probes/compare_over_lead.py` (old-encoder MLP, old-encoder GRU, new-encoder GRU on shared lead-time
axes). The first version plotted skill vs persistence and seemed to show a big win — but the
correlation panel showed identical curves, which didn't square. Cause: persistence is a much weaker
opponent on the new faster channels (1-step RMSE 0.154 → 0.269), so skill rose with the yardstick,
not the forecast. Switched panel (a) to absolute RMSE. Honest result: **more realistic, not more
accurate.** This also meant my earlier claim that day — "1-step skill jumped ~0.40 → ~0.70, channels
far more predictable" — was wrong, and F-18's heading had recorded it. Corrected in both places.
Lesson: a relative score is only comparable when the denominator is shared.
