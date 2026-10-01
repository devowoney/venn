# task_plan.md — Phases, Goals, Checklists

> Living project memory (per LLMAIProjectInstruction.md → FILE STRUCTURE / AGENT BEHAVIOR SUMMARY).
> Purpose: current and upcoming tasks, phase-by-phase goals, and checklists.
> Phase order is fixed: ARCHITECT → BLUEPRINT → LINK → STYLIZE → TRIGGER.

## Status

- **Frontal-cortex track (2026-10-01, branch `memory-intepreter`, D-033, SOP 05):** module 3 = frontal decoder maps the
  hippocampus's latent forecast to the field. v0 built (linear ridge, lags [0]) and scored on M=10/K=16: field R²/C
  .99 (h1) / .96 (h8) / .84 (h64); decoding loss ~0.007, the rest is the hippocampus.
  - [ ] User sign-off on the result -> F-26
  - [ ] Field-space uncertainty: decode the hippocampus ensemble members (spread map, field CRPS)
  - [ ] Hole regime (eyes < signals): `dec.lags` history window, when a testbed needs it
- **Observer track (2026-09-30, branch `feature-observer`, D-032):** North Star = a stable eye (over training time)
  with sufficient signal to reconstruct the state. Measured (F-24/F-25): eye still drifts at 8000 steps, no gain from
  8000 vs 2000 training steps; reconstruction already at the noise ceiling (R²/C 0.97-1.00) because the M=10 testbed
  has only 14 independent signals for K=16 eyes. Score definition: `memory/reconstruction_score.md`.
  - [ ] User decision: harder testbed (e.g. M=40 → 74 signals, K sized to it) so reconstruction can rank observers
  - [ ] Family ladder at many modes (0 stationary channels at M=40)
  - [ ] Longer budget + stop rule for "stable" (e.g. < 0.5% active pixels change per 1000 steps)
- *(Entries below predate the observer / history-forecaster tracks; kept as recorded.)*

- **Current phase:** Phase L done → early Phase S (STYLIZE). The v0 encoder trains stably and
  produces real structure, but two structural limits (decisions **D-018**) BLOCK the next design
  step and need a user call. See progress.md 2026-08-26 and findings F-1…F-5.
- **Awaiting user (D-018):** (a) allow a sign in the selection mask, (b) replace mean-slowness
  with an explicit population objective, and/or (c) add a reconstruction/coverage term.
- **Discovery status:** North Star, data shape, testbed, curriculum captured (see CLAUDE.md +
  findings/decisions). Remaining opens are design-level (mask domain, s_i normalization,
  per-scalar predictor + history window, distribution objective, success metric).

### Blueprint Discovery Questions (PENDING USER INPUT — ask one at a time)

Reframed for a PyTorch signal-processing NN project (see decisions.md D-002):

1. **North Star** — "What is the singular outcome that means we've won?" (a measurable target metric on a held-out set, e.g., RMSE / accuracy / SNR / F1 threshold)
2. **Data & Signals** — "What is the input signal and the target output?" (modality, sampling rate, length, channels, dtype, label/target)
3. **Source of Truth** — "Where does the primary data live?" (folder, dataset, zarr/netCDF store, DB; note size/format)
4. **Deliverable** — "What is the final artifact and how is success reported?" (checkpoint + metrics + plots; inference script)
5. **Constraints & Rules** — "What must the system do / not do?" (compute budget, reproducibility, coding conventions, data-handling rules)

Note: user said details will come step by step, so Discovery may be answered incrementally rather than in one pass.

## Implementation milestones (user-driven sequencing)

- **Milestone 0 — Synthetic SSH/SST testbed (FIRST):** generate spatio-temporally correlated
  dynamics with 2 pseudo-variables (SSH, SST), `field[time,2,lat,lon]`, with KNOWN ground-truth
  spatial patterns + dynamical families. Test the FULL pipeline (mask-encoder → K scalars →
  1D dynamics → classifier → self-organization) and validate encoder + classifier against
  ground truth. (decisions.md D-009; supersedes Lorenz D-006)
- **Milestone 1 — Ocean emulator:** apply the full pipeline (global selection-mask encoder →
  K scalar channels → independent 1D dynamics → classifier → kernel self-organization) to
  `field[time, vardepth, lat, lon]`, with the 0/0/100 → natural curriculum. (D-007, D-008)

## Phase Checklist

- [x] Phase A – ARCHITECT: memory scaffolding initialized (task_plan / findings / progress / decisions).
- [x] Phase A – ARCHITECT: A.N.T. SOPs written (SOP 00 (`decisions.md`), SOP 01). 2026-08-26.
- [~] Phase B – BLUEPRINT: Discovery largely done; Data-First schema in CLAUDE.md. Research pass (findings.md) STILL OPEN.
- [x] Phase L – LINK: env verified (oceanai, torch 2.13+cu126, H100); generator + K=16 smoke probe PASS. 2026-08-26.
      (encoder is PROVISIONAL plumbing; real encoder design still to confirm.)
- [x] Phase L – LINK: v0 training loop runs end-to-end and learns (D-017); eval probe reports the
      collective readout. 2026-08-26.
- [~] Phase S – STYLIZE: metrics + readout exist (`src/probes/evaluate.py` → `eval.json`, TB).
      BLOCKED on D-018 before results are worth promoting to `./results/`. Still to build:
      learned classifier (module 1½), latent predictor (module 2), forecaster (module 3).
- [ ] Phase S – STYLIZE: refine results (metrics tables, plots, named checkpoints); verify (pytest/plot/one-liner); get user sign-off.
- [ ] Phase T – TRIGGER: transfer to production; set up firing mechanism; finalize CLAUDE.md stability section; wire self-annealing repair loop.
