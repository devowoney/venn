# SOP 03 — Latent Predictor (module 2)

> Golden Rule: update this SOP BEFORE changing `src/models/predictor.py`,
> `src/train/train_predictor.py` or `src/probes/evaluate_predictor.py`.
> Realizes **module 2 of D-012**: evolve each scalar channel forward in the compressed domain,
> `s(t) → ŝ(t+1)`. Defaults recorded in D-022; all are Hydra-tunable.

## Goal

Given the K scalar channels produced by a **frozen** trained encoder (module 1), learn a
**per-channel 1D dynamical system with hidden state** that predicts the next value, and that keeps
predicting when run free (its own output fed back). Nothing spatial enters here — this module only
ever sees `S[T,K]`.

Two things this must deliver, and they are different:

1. **1-step skill** — beats persistence `ŝ(t+1) = s(t)`. Weak evidence: on a slow channel
   persistence is already near-perfect, so raw R² is meaningless. Only the **skill score against
   persistence** is informative.
2. **Rollout skill** — free-running over a horizon. This is what module 3 actually consumes, and
   it is where a stationary channel (should stay flat), a cyclic channel (should keep its phase for
   many periods) and a chaotic channel (should track for ~1 Lyapunov time, then decorrelate but
   stay on the attractor) separate from each other.

## Why the predictor needs a HIDDEN STATE (the design crux)

`s_i(t)` is a **1D projection of a higher-dimensional field**. A memoryless map `s(t) → s(t+1)`
is a function, so it must give the same next value every time the channel passes through the same
value — but the true trajectory passes through the same scalar value at many different phases of
the underlying dynamics, with different futures. For a cyclic mode a 1D memoryless map cannot even
represent a circle (it needs at least a 2D state); for a chaotic mode Takens says a delay embedding
of dimension `> 2·d_attractor` is required. So:

- a memoryless map is **structurally incapable** of this, no matter how big the network;
- the fix is state: either an explicit **delay window** `[s(t−p+1)…s(t)]` (Takens embedding), or a
  **learned recurrent hidden state** `h(t)` that accumulates the history (a learned, adaptive
  embedding). Both are implemented, and the claim "state is required" is therefore measured rather
  than asserted — the linear/AR variants share the same window machinery.

**Default = the learned recurrent state (`arch: gru`), user decision 2026-09-27 (D-028).** On the
current encoder the GRU wins on every axis — h1 +0.713 vs the MLP's +0.667, rollout amplitude 0.80
vs 0.57, 15/16 channels with honest amplitude vs 9/16, and the longest accuracy horizon (F-18). It
supersedes D-023, which chose the explicit delay window (`arch: mlp`) because the two *tied* at
short lead on the old Aug-27 encoder; that tie no longer exists. The MLP remains available, and is
the choice when an inspectable embedding dimension matters more than rollout fidelity.
See "Measured" below.

## Channel independence (D-012 "per-channel 1D")

Every channel gets its **own parameters** (`per_channel: true`, default). Channel `i`'s predictor
never sees channel `j` — that is what makes the emulator a bank of independent 1D systems and what
lets the classifier's family labels mean something per channel. Implemented as **batched**
per-channel weight tensors `[K, in, out]` with `einsum`, so all K systems train in one pass.
`per_channel: false` (one shared predictor, channels folded into the batch) is available as an
ablation — it tests whether the channels share a common dynamics.

## Tensors & call order

```
artifacts.npz (frozen encoder run)  ->  S[T,K]
   split in TIME (never shuffled across the boundary):
     train = S[:t_tr]        val = S[t_tr+gap:]          # gap avoids leakage through the window
   standardize per channel with TRAIN-ONLY stats:        # S std spans 9.6 .. 1250 across channels
     z = (s - mu_tr) / sd_tr                             # [T,K], each channel ~ N(0,1)

   per training step:
     idx ~ U[0, t_tr - (warmup+horizon) - 1], batch B
     w  = z[idx : idx+warmup]                            # [B,warmup,K]  burn-in (teacher forced)
     y  = z[idx+warmup : idx+warmup+horizon]             # [B,horizon,K] targets
     h  = predictor.warmup(w)                            # hidden state built from real data
     yhat, _ = predictor.rollout(w[:,-1], h, horizon)    # FREE-RUNNING: feeds its own output back
     L  = mean over (B,horizon,K) of (yhat - y)^2        # in standardized units
```

**Residual (increment) parameterization — default ON.** The network outputs `dz` and the state
update is `z(t+1) = z(t) + dz`. Persistence is then the *zero-output* solution, so the model starts
at persistence skill and has to learn only the departure from it. Without this, a slow channel
wastes all its capacity re-learning "output roughly the input".

**Horizon curriculum — default ON.** `horizon` ramps `1 → horizon_max` over the first
`curriculum_frac` of training. Free-running from step 0 at a long horizon diverges before the
1-step map is any good; teacher forcing only (horizon=1 forever) trains a model that has never seen
its own error and blows up at rollout. The ramp is the standard middle ground.

## Baselines (computed in the same script, on the same split — non-negotiable)

| baseline | definition | why |
|---|---|---|
| `persistence` | `ẑ(t+1) = z(t)` | the honest zero, and a *strong* one for slow channels |
| `linear AR(p)` | per-channel least squares on the train split, closed form | separates "the channel is linear-predictable" from "the network learned something" |

Headline metric = **skill score** `1 − MSE_model / MSE_persistence` at each horizon, per channel and
aggregated. Positive = better than persistence. Reported per family label so we can see, e.g., that
the win is concentrated in the cyclic channels.

## Metrics written (all per channel and aggregated, TRAIN and VAL)

- `mse_h`, `r2_h`, `skill_h` for `h = 1 … horizon_eval` (free-running rollout).
- `acc_horizon`: first horizon at which rollout correlation with truth drops below 0.5 — the
  channel's practical predictability time. Expect: stationary ≫ cyclic ≫ chaotic.
- rollout **variance ratio** `var(ŝ)/var(s)` at long horizon: catches the classic failure where the
  model minimizes MSE by decaying to the mean. A model that flatlines scores *well* on MSE and is
  useless to module 3 — this ratio is the flag.

## Figures (mandatory — the standing rule is never to judge a run from metrics alone)

Written to `<run>/figs_predictor/`:

- `pred_series.png` — per channel: truth, the 1-step prediction, a free-running rollout over the
  whole val window, the **flat rollout-persistence baseline** the rollout is scored against, and a
  **zoomed inset** (~60 steps, auto-placed on the most active stretch) holding truth / 1-step /
  1-step persistence.
  **Two different persistences appear, do not confuse them.** The main panel's black DASHED
  horizontal line is *rollout* persistence: hold `z(launch−1)` flat for the whole horizon — the
  literal denominator of the dashed curve's skill score. The inset's black DOTTED line is *1-step*
  persistence, re-anchored every step, which looks like truth shifted right by one, not like a flat
  line. Likewise the vertical dotted launch marker applies ONLY to the free run; the 1-step line is
  teacher-forced across the entire panel and keeps consuming real data to the right of it.
  The inset is not decoration. These channels have lag-1 autocorrelation **≈0.985**, so at full
  scale the 1-step prediction (RMSE 2.5% of the y-range) AND plain persistence (3.6%) both sit
  invisibly on the truth — the panel looks like a single curve and says nothing about the model.
  All of the 1-step information lives in that 2.5%, which is what the inset magnifies and what the
  skill score measures. Read the SOLID line only in the inset; at full scale only the DASHED
  free run is informative.
- `pred_horizon.png` — skill-vs-horizon curves, one line per channel, coloured by family label,
  with the persistence line at 0 and the AR(p) baseline dashed.
- `pred_scatter.png` — predicted vs true increment `dz`, per channel; a model stuck at persistence
  shows as a flat cloud at `dz_hat ≈ 0`.
- `pred_rollout_stats.png` — rollout variance ratio and correlation vs horizon per channel.

## Failure modes to watch (write findings when hit)

1. **Mean-collapse.** Long-horizon MSE is minimized by predicting the climatological mean. Detect
   with the variance ratio, not with MSE.
2. **Persistence mimicry.** With the residual parameterization the model can sit at `dz = 0` and
   look fine on 1-step R². Detect with the skill score (it will be ≈ 0) and `pred_scatter.png`.
3. **Scale leakage.** Standardizing with full-series statistics leaks val information into train.
   Stats are TRAIN-ONLY; the split has a gap of at least `warmup` steps.
4. **Chaotic channels look "unlearnable".** Expected: skill decays to 0 within a few steps. That is
   the physics, not a bug — judge them by whether the rollout stays on the attractor (variance
   ratio ≈ 1, right spectral peak), not by MSE.
5. **A dead channel** (encoder gave it near-zero variance) makes every ratio metric explode. Such
   channels are flagged and excluded from the aggregates, never silently dropped.

## Call order

```
1. python -m src.train.train                      # module 1, produces .tmps/runs/<ts>/artifacts.npz
2. python -m src.train.train_predictor  predictor.encoder_run=.tmps/runs/<ts>
                                                  # -> <ts>/predictor.pt, predictor_metrics.json, tb/
3. python -m src.probes.evaluate_predictor --run .tmps/runs/<ts>
                                                  # -> predictor_eval.json + figs_predictor/*.png
```

The predictor run logs into the **same** `<run>/tb/` directory as the encoder (D-020 convention:
one TensorBoard dir per run), under `pred/` tags.

## Measured on the CURRENT encoder — 2026-09-27 (`flat98fix_seed0`)

| predictor | h1 | h4 | h64 | amplitude @h64 | honest ch. | acc horizon |
|---|---|---|---|---|---|---|
| **GRU** | **+0.713** | **+0.746** | +0.264 | **0.80** | **15/16** | **40.0** |
| MLP (current default) | +0.667 | +0.692 | +0.241 | 0.57 | 9/16 | 35.3 |
| learned linear AR(16) | +0.588 | +0.560 | +0.443 | 0.14 | 2/16 | 32.4 |
| closed-form AR(8) | +0.684 | — | +0.386 | 0.14 | 3/16 | — |

On the current channels the GRU wins on every axis, and is the first trained model to beat the
closed-form baseline at 1 step. Both linear forms' high h64 skill is mean-collapse (amplitude 0.14).
Per family (GRU): cyclic horizon 62.6 steps, chaotic 26.8, stationary h1 ≈ 0 (expected — near-flat
channels, `amp_ratio` 0.02–0.04). Full account: F-18 in `memory/findings.md`.
**The user switched the default to `gru` on this evidence (D-028).**

**Old vs new encoder** (`compare_over_lead.py`, `.tmps/forecast_old_vs_new.png`): the forecast is
more **realistic** (94% of channels keep honest amplitude at h64, vs ~50–56%) but **not more
accurate** — absolute error and correlation curves overlap, usefulness ends ~40 steps either way.
Skill vs persistence is not comparable across encoders; see the F-18 correction.

## Measured on the OLD encoder (Aug 27, superseded by the table above) — 2026-08-29 / 2026-08-31 (K=16 encoder run, generator rev2)

Same frozen encoder for all rows; train `[0,1400)`, val `[1432,2000)`, 472 rollout launches,
warmup 32, `horizon_max` 16 in the loss, scored free-running to lead 64.

> **All AR numbers here are the CORRECTED ones (2026-08-31).** The AR baseline was leaking its own
> target until then — see finding F-10 in `memory/findings.md`. Anything quoting AR(8) at
> +0.556 predates the fix and is wrong.

| predictor | params/ch | skill h1 | skill h4 | skill h64 | var ratio @h64 | channels won |
|---|---|---|---|---|---|---|
| **MLP, 16-step delay window (DEFAULT)** | 289 | +0.398 | +0.561 | **+0.393** | 0.65 | 7 |
| GRU, learned hidden state | 929 | **+0.420** | **+0.564** | +0.246 | **0.89** | 9 |
| AR-anchored residual (`ar_mlp`) | 289 | +0.431 | +0.542 | +0.185 | 0.79 | — |
| `ar_mlp`, 1-step residual only | 289 | +0.443 | +0.544 | −8.89 | 17.7 | — |
| learned linear AR(16) (ablation) | 17 | +0.356 | +0.485 | +0.425 | **0.18** | — |
| GRU, 1-step loss only (no curriculum) | 929 | +0.475 | +0.593 | **−33.07** | **16.12** | — |
| `ou` — stationary expert, AR(1) bounded | 2 | **−0.007** | +0.062 | +0.301 | **0.19** | **0** |
| `osc` — cyclic expert, polar AR(2) | 3 | **−0.065** | −0.248 | +0.245 | **0.00** | **0** |
| closed-form AR(8), least squares | — | +0.438 | — | +0.295 | — | — |

Read this table with the amplitude column, not without it. Five conclusions:

1. **Long-lead MSE ranks the forecasters BACKWARDS, and `var_ratio` is the only thing that exposes
   it.** The linear model, `ou` and `osc` all score well at h64 *because* their rollouts decay to
   the climatological mean (amplitude 0.18 / 0.19 / 0.00). They would hand module 3 a flat field.
   This trap fired three separate times in this work. **Rank by `var_ratio` first, then by skill**,
   and ship the amplitude panel with every comparison from the start.
2. **The rollout curriculum is load-bearing.** Training on 1-step loss only buys ~+0.05 at h1 and
   destroys the rollout (skill −33, amplitude 16x). Never train this module teacher-forced-only.
3. **Architecture is NOT a lever here.** Every usable form lands within 0.045 at h1, and they win
   and lose on the SAME channels (`corr(AR, GRU)` across channels = **+0.964**). Per-channel oracle
   selection over four forms buys **+0.009**. What varies is channel predictability, which module 1
   decides — the encoder is optimized for SLOWNESS, which is exactly why every channel is
   near-linear and persistence/AR are so hard to beat.
4. **Family-specialized experts win 0 of 16 channels** (F-11). `ou` and `osc` are linear functions
   of the delay window, hence strict special cases of the MLP that already contains them; the
   specialization can only act as regularization, and there is no data shortage to regularize.
   Do not build the mixture-of-experts gate — its premise is measured false.
5. **The AR anchor is worth keeping anyway, for its floor, not its score.** `ar_mlp` provably cannot
   start below the closed-form baseline and the trainer asserts exactly that every run. That
   assertion is what caught F-10; nothing else cross-checked the baseline.

**Caveat on the family labels used in the figures:** they come from the module-1 heuristic
labeller, which calls almost everything cyclic (population 0/14/2 vs the 1/3/6 target). The
per-family aggregates inherit that error. Independently, "which form wins each channel" gives
`chaotic(2) -> gru x2`, `cyclic(14) -> mlp x7, gru x7` — it does not recover a three-family
structure either, which is consistent with D-013: masks are regional sensors, so channels are
MIXTURES of modes rather than pure family members.
