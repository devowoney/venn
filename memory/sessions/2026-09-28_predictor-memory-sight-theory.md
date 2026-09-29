# 2026-09-28 — Module 2 reframing: observer trajectory as memory, "sight", two configurations

Discussion-only sessions (no code changed, no runs; nothing committed — git is the user's). First
session transcript:
`~/.claude/projects/-home-sysadmin-jlee-venn--claude-worktrees-latent-predictor/c70264ed-4168-4a0c-a933-ac431eb71e76.jsonl`
(worktree `latent-predictor`). The user closed it to continue the discussion in a new session, recorded
in the last section ("continued, new session").

> Provenance: first written in the worktree as `memory/sessions/2026-09-28_predictor-memory-sight-theory.md`
> (that session could not write to the main checkout), then extended with the continuation session as
> the tracked copy `architecture/03e_module2_session_narrative.md`. Merged here 2026-09-28: this file in
> the main checkout's `memory/sessions/` is now the home; `architecture/` is retired.

## 1. Starting point — current predictor (recap given to the user)

- `config/predictor.yaml`: `arch: gru` (D-028), hidden 16, per-channel, residual `dz`, warmup 32
  teacher-forced from a ZERO hidden state, rollout curriculum 1→16, lambda_tf 1, 4000 steps.
- Data: frozen encoder `flat98fix_seed0`, S[2000,16], train [0,1400), gap 32, val [1432,2000).
- Performance (F-18): skill h1 +0.713 / h4 +0.746 / h64 +0.264; RMSE h1 0.091, h16 0.570,
  h64 1.029; amplitude 0.80 @h64, 15/16 honest; accuracy horizon ~36–40 steps
  (cyclic 62.6, chaotic 26.8, stationary skill ≈ 0). More realistic than the old encoder, NOT more
  accurate (F-18 correction).
- Stale label spotted: SOP 03 and F-18 tables still say "MLP (current default)". Not fixed.
- Code fact: every launch (train + eval) starts from `init_state` = zeros and sees only 32 steps;
  only the 1-step teacher-forced line in `evaluate_predictor.py:89` runs over the whole val split.

## 2. The user's argument (evolved over three turns — read in order)

1. Turn 1: the predictor has its own memory, so a 32-step warmup is not intrinsically needed.
   Warmup should be named **"predictor's sight"** (how much captured trajectory it sees). The
   past is "reanalysed" too, so training and inference are not fully separate — the predictor is
   continuously fine-tuned by appended data. Claimed benefit: extremes and long-period dynamics.
2. **I misread this** as the GRU hidden state (answered: sight is bounded by the training-window
   length; hidden state = filter not reanalysis; state update ≠ weight update).
3. Turn 2 correction: memory = **the trajectory extracted by the observer (encoder)**, not the GRU.
4. Turn 3 precise version — **two configurations**:
   - **(1) Retrieval:** a learned/stored trajectory memory; a new series of observed states ->
     find the probable trajectory in the past -> predict the next step. ≈ analog forecasting.
   - **(2) Continuous online:** the observer-captured trajectory IS a property of the dynamical
     system (definition of a dynamical system = evolution of its observables). The model does
     built-in online training + prediction; **observer sight is the key parameter**; attention
     looks promising. Architecture is still being developed.
   The user asked: find the weak parts of this theory.

## 3. Weaknesses raised (the open list to discuss next)

Both configurations:
- **W1 — the trajectory belongs to system + observer, not the system alone.** The observer is
  LEARNED: a retrained encoder changes the trajectory completely (old→new encoder: timescales,
  family mix 0/14/2 → 2/5/9, skill yardstick). The memory is only valid for one observer version;
  the slowness objective decides which dynamics are visible. The theory needs a condition on the
  observer (e.g. it must be an embedding of the attractor).
- **W2 — per-channel independence (D-012) weakens Takens.** Delay embedding from one scalar
  holds only for low-dimensional, noise-free, autonomous systems; the channels are mixtures
  (D-013); the joint 16-channel record carries far more of the state. Hurts extremes most
  (precursors often in another channel).
- **W3 — autonomy / stationarity assumed.** Forced systems (seasons, wind, trends) break "the past
  determines the future". The synthetic generator is probably autonomous, so it cannot test this.
- **W4 — record length.** T=2000 (1400 for training): periods beyond ~300–500 steps can't be
  learned; extremes are too few.

Configuration (1):
- **W5 — the number of possible analogs explodes with dimension** (Van den Dool). Sight has an
  OPTIMUM, it is not "longer = better".
- **W6 — extremes averaged away or capped:** averaging k analogs = mean-collapse; can't exceed the
  recorded maximum. Reframe as "reproduce recorded types of extremes".

Configuration (2):
- **W7 — "online training" = in-context learning;** the weights are still trained offline, and it
  only covers regimes seen in that training. That needs many trajectories/regimes; we have one
  series. Undercuts the extremes claim.
- **W8 — (1) and (2) may collapse into one:** attention over past states ≈ soft learned
  nearest-neighbour matching. The only difference may be sight length vs archive length — the
  theory needs a sharper dividing line.
- **W9 — sight has no principled value yet:** as a hyperparameter it is just `window`/`warmup`
  renamed. Proposal: per-channel sight derived from the encoder's timescale rung.
- **W10 — online protocol risks:** feeding forecasts back → errors compound; feeding observations
  back → filtering, so the lead time must be stated; train/test boundary disappears → strictly
  causal "predict, reveal, update" scoring; attention cost O(L²); on univariate series simple
  linear models often match transformers (Zeng et al. 2023) → must beat analog/AR on
  amplitude-aware metrics.

My ranking of the weakest points: W1, W2, then W4/W7.

## 4. Proposed next measurements (NOT started; await the user's decision)

1. Analog forecaster: skill + amplitude vs sight (tests W5; the theory predicts a peak).
2. Single-channel vs joint 16-channel input, as a function of sight (tests W2).
3. (From turn 1, still valid) streaming state vs zero-start launches; training-window sweep
   8/16/32/64/128.

## Files touched

- Auto-memory: `~/.claude/projects/-home-sysadmin-jlee-venn/memory/predictor-two-configs-theory.md`
  (+ line in MEMORY.md).
- This record. No repo code or docs changed.

# 2026-09-28 (continued, new session) — the collapse of (1) and (2) is the GOAL

## The user's ruling on W8

> "W8: (1) and (2) may collapse. It is good news. I am not looking forward two different model.
> If it collapse, it means that my model is good to forecast."

W8 is therefore **not a weakness but a design goal**: ONE model, not two. Working form proposed:
attention over the observer-captured trajectory, with sight = the accessible archive, doing analog
retrieval (config 1) and within-window adaptation (config 2) in the same forward pass. A testable
statement of the collapse: attention weights land on past segments that are true dynamical analogs,
AND skill rises with sight up to an optimum (W5).

## Caveats recorded with the ruling

1. **A collapse shows consistency, not skill.** The unified model must still beat, on amplitude-aware
   metrics (rank by `var_ratio` first, F-9): a non-learned analog forecaster, AR/linear baselines
   (cf. Zeng et al. 2023) and the current GRU (F-18: skill h1 +0.713, h64 +0.264, horizon ~36–40).
2. **The collapse gives up part of the original claim.** If (2) is retrieval, W6 applies to the one
   model: it reproduces recorded TYPES of extremes but cannot exceed the recorded maximum. W7
   (in-context learning of unseen regimes) disappears; W4 sharpens (analog pool = 1400 steps).
3. **Still open before any architecture:** W1 (memory valid for one observer version only,
   `flat98fix_seed0`) and W2 (per-channel vs joint 16-channel input) — both decide the model's input.

## Proposed next measurement (NOT started; awaits the user's go)

Non-learned analog forecaster on the frozen encoder output, sight swept 8/16/32/64/128, per-channel
AND joint, scored on the val split `[1432,2000)` (skill + amplitude). It is the zero-parameter version
of the unified model, the baseline the attention model must beat, and a direct test of W5 and W2.
Outputs to `.tmps/` only.

## Housekeeping noted, not done

- The main checkout (`1812638`) is older than this branch; `memory/` ≙ `architecture/` for the
  tracked record (user, 2026-09-28).
- Stale label still unfixed: SOP 03 and F-18 say "MLP (current default)" (D-028 made it the GRU).
  *(At the merge the `findings.md` F-18 row was relabelled "default at the time; D-023"; the SOP 03
  table, copied verbatim, still carries the stale label.)*
- `architecture/README.md` is still the stationary-observer handoff and does not list the 03* files.
  *(Resolved by the 2026-09-28 merge: `architecture/` retired; README → `2026-09-01_stationary-observer-
  merge-handoff.md`, 03/03a/03b/03e → `memory/sop/03_latent_predictor.md`, `decisions.md`, `findings.md`,
  this file.)*

# 2026-09-28 (continued) — W1 retired into A1; W1′ = sufficient observer

User: "I don't think that W1 is a problem. because the observer will be trained by a training data set.
Therefore I suppose that a dynamical system has a stable observer." Then: "I agree with sufficient observer."

- Agreed: a frozen observer is stable by construction; retraining = design change. Recorded as **A1** (D-029).
- What survives of W1 is **sufficiency**, not stability: the per-channel predictor (D-012) can only forecast
  dynamics some channel sees. Evidence: `fixavg` seeds 2 and 3 have no period-60 channel. Recorded as
  **W1′**, merged with W2 (D-029).
- Ranking of open weak points is now: **W1′(+W2)**, then W4 / W6 (W7 dropped with the W8 collapse).
- Test launched: analog forecaster × encoder seeds `flat98fix_seed{0..4}` (outputs in `.tmps/`).

## Seed test result (F-19) — and a correction of my own argument

- My W1′ evidence ("seeds 2/3 blind to the period-60 cycle", from dominant FFT periods) was WRONG: their best
  single channel forecasts that mode at +0.99. Every seed's observer is sufficient for all three cyclic modes.
- Joint 16-channel analogs lose to one channel (dimension explosion, W5) — naive retrieval can't use joint info.
- Sight optimum is family-dependent (cyclic ↑ to 128, chaotic peak 16) — supports per-channel sight (W9).
- k=10 analogs: higher skill, amplitude 0.52 at h64 (W6 mean-collapse). k=1: honest amplitude, skill ≈ GRU at h64,
  far below it at h1.
- Figure: `.tmps/analog_seed/analog_seed_test.png`.

# 2026-09-29 (same discussion) — out of the train/inference scheme: streaming emulator (D-030)

- User: the input can be a single time step (observer trained on [0:2000], predictor fed 2001 to predict the
  next); the emulator is trained for one dynamic and keeps its own captured memory; after 10 years of
  training it is used from the next day by APPENDING states. Ruling: **weights frozen, only memory grows.**
- This answers my "input must be a window" refinement: the history is in the memory, not the input.
- Agreed consequence: evaluation must stream. Three departures found in the current tests: GRU zero-state
  restarts (F-18), frozen analog library (F-19), encoder trained on the full record incl. the scored period.
- Recorded as **D-030** with the streaming protocol (observer + predictor trained on [0,1400), stream
  1400→1999, predict-reveal-append-score). Not yet run.
