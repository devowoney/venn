# progress.md — What Was Done, Errors, Tests, Results

> Living project memory (per .claude/LLMAI_venn_init.md → FILE STRUCTURE / AGENT BEHAVIOR SUMMARY).
> Purpose: status updates — what's done / pending, errors hit, tests run, and results.
> Long-form per-session narratives (incl. dead ends) live in `memory/sessions/` — see its README.

## Log

### 2026-09-01 (later) — Frame corrections: structural classifier + observables, not modes (D-026)

- **User slowed the work down to fix two framings, and both landed on things I had built.**
  (1) Classify by GLOBAL STRUCTURE, not small fluctuation and drift — my ACF-recurrence test read
  the raw series, where noise and drift dominate, so it mislabelled visibly cyclic channels.
  (2) The observer is not a mode-recovery device: the kernel decides WHERE TO LOOK, and a family is
  a property of the observable, so a stationary channel may be a single mode, a persistent
  phenomenon, or a combination of signals that barely varies.
- **Asked before building** (4 questions, all answered): explicit 4-part decomposition; no
  persistence check (flatness over the record is enough); cancellation allowed but not encouraged;
  1/3/6 is a design target for the bank, not a recovery score.
- **Built:** `family.decompose` (level / trend / osc / residual, with a HALF-POWER band for `osc` so
  a quasi-periodic hump counts as oscillation instead of falling into the residual); label = flat
  gate then `trend+osc` vs `residual`; probe now prints the three shares instead of line%/period.
- **Rejected a tidier variant:** argmax over all four shares including the level. `level^2 > var` is
  just `amp_ratio < 1`, so a channel fluctuating at 50% of its level would read "stationary"
  (counter-example: ch14 of 20260901_080655, amp_ratio 0.526, plainly broadband). Level gates,
  structure votes.
- **Re-scored everything post-hoc (a label is a readout — no retraining):** rev3 truth 10/10;
  rev2 truth 9/10 where the "miss" is the old OU mode 0 now labelled cyclic, i.e. D-025 working;
  rev3 ladder 2/5/9 with role obedience 1.00; rev3 baseline 0/11/5.
- **Answered the specific channels:** ch2-6 are cyclic (osc share 0.70-0.93), agreeing with the
  user. ch9/ch11 stay chaotic but at osc share 0.41/0.32 vs residual 0.59/0.67 — genuine cyclic
  character, broadband still dominant; forcing them to cyclic would contradict the Lorenz mode they
  track and cost truth accuracy.
- **Demoted, per the ruling:** `mask_align_with_pattern` and `mode_recovery` are now labelled
  diagnostics; `population` is scored against a DESIGN target. I had been reading the flat channel's
  |corr| 0.27 with the injected pattern as a failure — wrong frame, now recorded as such.
- **New diagnostic:** `cancellation_vs_independent` — projects every hidden mode onto the flat
  channel's footprint and compares the net fluctuation to the independent-addition baseline
  `sqrt(sum sd^2)`. First measurement (ladder ch1): 3 modes reach the footprint, index **1.01** =
  incoherent addition, so this stationary observer is a QUIET FOOTPRINT, not a cancelling
  combination. My first version normalized by the plain sum of parts and read 0.634 — which looks
  like cancellation but is what chance predicts for 3 parts (0.577); the wrong baseline would have
  manufactured a finding.
- **Errors:** none outstanding. **Tests run:** re-evaluation of 3 archived runs under the new rule,
  truth validation, figure regeneration.
- **Pending / next:** (1) harmonic folding in `decompose` — a non-sinusoidal periodic signal leaks
  into `residual` and biases toward chaotic; (2) module 2 still divides by ~0 on a constant channel;
  (3) a learned classifier, now that the labeller matches the project's definitions; (4) whether the
  ladder's `slow_period_min` / `cyclic_period_max` still make sense now that the slow band holds no
  stationary mode.

### 2026-09-01 — User correction: stationary = CONSTANT (D-025); generator rev3 + L_level

- **User's correction, and it was right:** the injected mode 0 was not stationary — `_ou_series`
  standardizes, so the "stationary" mode carried UNIT VARIANCE and was just a slow wanderer. My
  F-11 labeller had defined stationary as "slow red noise", which is a different taxonomy from the
  project's, so the populations I reported yesterday (10/1/5 baseline) are slow/cyclic/fast counts,
  not family counts. Recorded as F-13; D-025 supersedes the definition.
- **Also checked the user's second claim honestly:** of the channels called out as "clearly cyclic",
  ch2 is genuinely oscillatory (ACF trough -0.83 with a +0.74 rebound; the true cyclic modes run
  -0.92..-0.98 / +0.85..+0.97), but ch4/5/6/10 show no ACF recurrence — they are slow wandering
  with visible wiggles. Reported the split rather than agreeing wholesale.
- **Built:** generator rev3 (`stationary_constant`, flat un-standardized mode 0, plus an SST
  bypass — a constant's AR1 steady state IS the constant, and `_ar1_response`'s closing
  `_standardize` would have divided by ~0 and wiped it out of SST); labeller rewritten around
  `amp_ratio` with flatness tested first; `level_term` in `src/train/spectral.py` with per-rung
  exemptions from `l_var`/`l_energy`/`L_band`/`L_line`; `stationary_capture` readout + constant-mode
  guards in the probe (`mode_r2` is degenerate for a zero-variance mode, now reports `--`).
- **Measured (rev3 field, K=16, 5000 steps, seed 0):** baseline population 0/14/2 with flattest
  channel `amp_ratio` **0.778 — no flat channel at all**; ladder + `L_level` gives **2/5/9**,
  **role obedience 1.00**, flattest channel `amp_ratio` **0.023**, 5 of 5 cyclic rungs at
  line% >= 70. Balanced recon R^2 0.7765 vs the baseline's 0.7763 (rev3 is a harder field than
  rev2's 0.99 — not a regression).
- **Fixed a lying figure (2nd time, cf. F-6):** `plots.py` standardized each channel, dividing out
  the std — the very quantity that defines flatness — so a 2%-of-level wiggle was drawn at full
  amplitude and the flat channels looked oscillatory. Now scaled by RMS about zero, with
  `amp_ratio` printed per lane. The corrected figure shows ch0/ch1 as flat lines, ch2-6 as clean
  cycles, ch7-15 as broadband.
- **Null result:** de-aligning the size and timescale ladders (`slow_size_frac=0.08`) halved the
  flat footprints but did not improve pattern alignment and cost role obedience (1.00 -> 0.94).
  Default reverted to 0.0, reasoning kept in the config comment.
- **Errors:** none outstanding. One self-inflicted: I first set `flat_target=0.8`, which corresponds
  to `amp_ratio` 0.25 — the hinge would have gone quiet before the channel passed the labeller's
  0.05 cut. Caught before running; now 0.95.
- **Tests run:** labeller validation on 5 seeds under rev3; 4 trainings (rev3 baseline, ladder,
  ladder with de-aligned sizes, plus a 150-step smoke) with eval on each; figures on two;
  a gradient-norm measurement for `lambda_level`.
- **Pending / next:** (1) the flat channels align with the injected pattern at only |corr| 0.27 —
  decide whether the observer must ISOLATE the static pattern or only report a stable level;
  (2) re-run modules 2/3 on a rev3 ladder encoder — module 2 has never seen a constant channel and
  its per-channel normalization may divide by ~0; (3) a learned classifier, now that the labeller
  is trustworthy; (4) `slow_period_min` / `cyclic_period_max` were tuned for the OU-drift reading
  and may want revisiting now that the slow band holds no stationary mode.

### 2026-08-31 — The labeller was the bug (F-11); timescale ladder built (D-024)

- **User's ask:** develop the observer to capture the stationary mode too, having found that
  deepening the kernel bank raises slow-mode visibility but is not the lever for every family.
- **Found first (F-11):** the encoder ALREADY captured the stationary mode (max |corr| 0.92-0.93,
  ensemble R² ~1.00 across the K sweep). What could not happen was the READOUT: the old labeller
  returned `cyclic` for any red spectrum — including the generator's own stationary series
  (accuracy 0.7/1.0, sitting in plain sight in every eval.json). Every `population.stationary = 0`
  in this project's history was a probe artifact.
- **Built:** `src/probes/family.py` — line-width (`line_frac`) + memory (`tau_e`) labeller,
  calibrated on the hidden truth: **10/10 on seeds 0-4** (was 7/10). `evaluate.py` now imports it;
  the local copy is gone so the two cannot drift.
- **Re-scored the K sweep with it:** populations 3/0/1 (K=4), 4/2/2, 12/0/4 (K=16), 22/3/7,
  36/10/18 (K=64) — slow-dominated at every K, and the sign of the problem is the OPPOSITE of what
  we believed: too much red, near-zero clean cyclic. F-12 records why K cannot fix it.
- **Built (D-024):** `src/train/spectral.py` — per-channel spectral band ladder (`L_band`,
  `L_line`) + a long-lag memory hinge on the slow rungs (`L_mem`), on random contiguous windows
  (a lag-1 pair minibatch cannot see a period-300 cycle). Wired into `train.py` behind
  `train.spectral.*`; weights gradient-matched; `roles` saved into artifacts.npz and checked by the
  probe as `role_obedience`.
- **Measured (K=16, 5000 steps, seed 0):** baseline 10/1/5 -> ladder 5/1/10 population; the slow
  rung goes from corr 0.50-with-a-chaotic-mode to **corr 0.92 with the true stationary mode**
  (tau_e 87 vs the truth's 106); fast rungs obey ~perfectly with honest gap1 0.10-0.13; worst
  hidden-mode |corr| 0.25 -> 0.61; balanced recon R² unchanged at 0.9891.
- **Figures looked at (standing rule):** `figs/features.png` for both runs. The ladder bank is
  stratified slow -> cyclic -> fast down the channel index; the baseline's is interleaved. The
  figure is also what shows the remaining failure honestly — cyclic rungs 2/3/5/6 wander instead of
  oscillating.
- **Errors:** none outstanding.
- **Tests run:** labeller validation on 5 seeds + re-scoring of 5 archived runs; 4 trainings
  (baseline / ladder / ladder+lambda_slow=0 / ladder+L_mem) with eval on each and figures on two;
  two per-term gradient-norm measurements.
- **Process:** background job, so the work is in the git worktree
  `.claude/worktrees/stationary-observer` (branch `worktree-stationary-observer`), NOT the main
  checkout. Nothing committed — git stays the user's (standing rule).
- **Pending / next:** (1) the CYCLIC rungs are the open gap — 1 of 5 reaches a clean line, because a
  non-negative mask cannot cancel the basin-scale red background it sits in (F-4 in the time
  domain); candidate levers in SOP 02, and (a) a temporal high-pass in `s_i` needs a user decision
  since it changes D-005's observer semantics. (2) De-align the size and timescale ladders — the
  failing slow rung is the largest-footprint one (D-021 conflict). (3) A LEARNED classifier can now
  be trained against a labeller that actually works. (4) Modules 2/3 re-run on a ladder encoder.

### 2026-08-29 — Module 2 BUILT: latent predictor with hidden state (D-022); F-9 inverts the ranking

- **Built (SOP first, per the Golden Rule):** `memory/sop/03_latent_predictor.md`, then
  `src/models/predictor.py` (per-channel GRU / delay-window MLP / learned-linear behind one
  `init_state / step / warmup / rollout` interface, + closed-form AR(p) baseline),
  `src/probes/pred_metrics.py` (shared by trainer AND probe so the two can't drift),
  `src/train/train_predictor.py`, `src/probes/evaluate_predictor.py` (4 figures),
  `config/predictor.yaml` (composes `config.yaml`; own hydra dir `.tmps/runs_pred/`).
- **Design crux:** a memoryless `s(t)->s(t+1)` is structurally impossible for a 1D projection of a
  higher-dim field, so state is mandatory (learned GRU state, or explicit Takens window as the
  ablation). Residual parameterization + free-running rollout loss with a 1->16 horizon curriculum.
- **Measured (4 trainings, same frozen encoder run 20260827_151413):** GRU skill h1 +0.420 /
  h64 +0.246, var ratio 0.89 | MLP +0.398 / +0.393, 0.65 | linear +0.356 / **+0.425**, **0.18** |
  GRU 1-step-only +0.475 / **-33.07**, **16.12** | closed-form AR(8) **+0.556** / +0.300.
- **F-9, the finding that matters:** long-lead MSE skill ranks the forecasters BACKWARDS. The
  linear model scores best at h64 *because* it decays to the climatological mean (amplitude 1e-5 of
  truth); the GRU scores worst at h64 and is the only one with honest amplitude. `var_ratio` is now
  a first-class metric ranked BEFORE MSE skill (D-022). Same lesson as F-6 in a new place.
- **Also measured:** the curriculum is load-bearing (1-step-only training makes the rollout
  diverge, 16x amplitude); closed-form AR(8) still wins at 1 step, because module 1 optimizes
  SLOWNESS and therefore hands the predictor near-linear channels; the hidden state wins 5x over AR
  exactly on the two chaotic-labelled channels (+0.875/+0.909 vs +0.135/+0.160).
- **Errors:** one, fixed — torch>=2.6 `weights_only=True` default broke reloading our own checkpoint
  (it carries numpy norm stats + AR coefficients).
- **Tests run:** smoke run (201 steps), 4 full 4000-step trainings, eval probe + figures on each.
- **Process note:** the background-job harness forced isolation, so the code lives in the git
  worktree `.claude/worktrees/latent-predictor` (branch `worktree-latent-predictor`), NOT in the
  main checkout. Git remains the user's; nothing was committed to `main`.
- **Repo issue spotted (pre-existing):** `.gitignore` is committed WITH unresolved merge-conflict
  markers (`<<<<<<< HEAD` / `=======` / `>>>>>>> 7afcabf`). Harmless today, worth cleaning.
- **Pending / next:** (1) a real classifier — it now blocks module 2's per-family interpretation as
  well as module 1's (population still 0/14/2 vs the 1/3/6 target); (2) module 3, the forecaster by
  gradient inversion — `predictor.rollout` already returns exactly the `s_hat` it needs;
  (3) consider training the encoder jointly with the predictor instead of freezing it.

### 2026-08-27 — Generator rev2 (D-021): the decorrelation ceiling is gone

- **Built:** `src/probes/plots_field.py` (field snapshots, hidden-mode answer key,
  kernel-activation map, `field.gif` with outlines pulsing at their instantaneous activation), then
  generator rev2 fixing all three F-8 defects. Back-compatible: `GenConfig` defaults = legacy.
- **Measured on the field:** dead cells 26.3% → 3.4%; corr(SSH,SST) 0.981 → 0.400; negative
  cell-pair correlations 0.0% → 49.0%.
- **Measured on training:** off-diag corr² 0.259 → 0.063; effective rank 3.89 → **9.23/16**; mask
  IoU 0.111 → 0.024; balanced R² 0.880 → 0.989; all ten hidden modes at ensemble R² 0.80–1.00.
- **Conclusion that reverses my earlier one:** the F-4 non-negativity "structural limit" was a
  TESTBED artifact. rev2 with plain non-negative masks (9.23) beats the signed-mask ablation on the
  old field (8.57), so D-005's selection semantics need no change and D-018 is resolved.
- **The figures caught a bad intermediate:** domain-filling waves scored BETTER on every metric
  (eff rank 9.56) while turning the field into a global plaid. Only `field_snapshots.png` showed
  it. Fixed with regional wave packets (`wave_env_sigma=15`, `wave_kmax=2`).
- **Also fixed:** `plots_field --no-run` was falling back to `GenConfig()` — i.e. showing the
  LEGACY field — instead of the live `config/config.yaml`.
- **Errors:** none outstanding. **Tests run:** legacy-vs-rev2 field diagnostics, 2 full trainings
  + eval + figures, wave-envelope iteration.
- **Pending / next:** `L_energy` now fights the size ladder at the largest rung (see D-021). The
  heuristic labeller still calls almost everything cyclic (population 0/14/2) — a real classifier
  is unbuilt, as are module 2 (latent predictor) and module 3 (forecaster).

### 2026-08-27 — Directory/git rearrangement + scale ladder (D-020)

- **Git (user directive):** removed the single root commit `8c7f3c9` with
  `git update-ref -d HEAD` + `git read-tree --empty` + gc. **All files kept on disk**; the repo now
  has NO commits and everything is untracked/ignored. Work lives locally, per the user.
  (Consistent with the standing rule that the user owns git in this repo.)
- **`.gitignore` (user wrote it):** now ignores `.claude/` and `memory/` under "AI artifacts", plus
  `.venv/`. The AI-maintained files are therefore untracked — I still maintain them as before.
  A tools venv `.venv/` now exists (D-003's second layer); it lacks matplotlib, so the plot probe
  runs from `oceanai`, which already had it.
- **User also moved:** `LLMAIProjectInstruction.md` -> `.claude/LLMAI_venn_init.md`, and
  `venn_idea.txt` -> `.claude/venn_idea.txt`. Updated CLAUDE.md's pointer to the new path.
- **TensorBoard (user directive):** merged train + eval into ONE dir per run. `evaluate.py` now
  writes into `<run>/tb/` instead of `<run>/tb_eval/`; verified 45 scalar tags in one dir
  (7 `loss/`, 16 `S/`, 9 `collective/`) + 2 image tags.
- **Bug found while verifying:** `evaluate.py` referenced a stale `r2_init` (renamed to `r2_i`
  when the balanced metrics went in) -> NameError AFTER the images/series were written, so the
  `collective/*` scalars were silently missing and my greps hid the traceback. Fixed and expanded
  to 9 collective tags.
- **Answered the user's question with data:** is the energy term enough on its own? NO. Energy-only
  leaves the chaotic modes at ensemble R² 0.01-0.06 (same as baseline); only energy+recon restores
  them to 0.79-1.00. Both terms stay.
- **Built (D-020):** per-channel scale ladder. Footprints now span 8..1143 cells in a monotone
  ladder (was 7..186 all-small), mask IoU 0.111 -> 0.061, other metrics unchanged.
- **Errors:** the `r2_init` NameError above. **Tests run:** λ_size sweep (0 / 0.03 / 0.1 / 0.3),
  per-term gradient-norm measurement incl. size, eval + figures on each.
- **Pending / next:** effective rank is still ~3.6 of 16 — the figures show the ladder diversifies
  SCALE but the footprints nest on the same few energetic structures, so LOCATION diversity is the
  open lever, alongside the still-unapproved F-4 generator fix (sign-varying patterns).

### 2026-08-27 — Visualization probe + energy/coverage terms (D-019); F-6 and F-5 fixed

- **User directive (standing rule):** *do not judge a result from metrics — always visualize it.*
  Built `src/probes/plots.py` → `<run>/figs/`: `masks.png`, `masks_on_energy.png` (kernel outlines
  over the field's temporal-std map), `features.png` (the K scalar series), `feature_var.png`
  (rolling variance + slowness ranking). matplotlib 3.11.1 was already in `oceanai`; nothing
  installed (D-003 respected — this probe only READS artifacts).
- **What the figures caught that no metric did (F-6):** the learned kernels had migrated to the
  DEAD CORNERS of the domain (lowest field energy), because the scale-free `var(Δs)/var(s)` ratio
  is indifferent to amplitude — a tiny pure-slow tail in an empty corner scores near-perfectly.
  That single mechanism explained the duplication, the lost chaotic modes, and the sub-random
  reconstruction all at once. My earlier metric-based diagnosis had the priority wrong.
- **Built (D-019):** energy floor `relu(1 − e_i/e_ref)²` with `e_i = var(s_i)/count_i²`, and a
  differentiable coverage term `1 − R²` from the best per-batch linear decode. Weights set by
  matching GRADIENT norms (values misrank the terms by 300×).
- **Result (F-7):** min energy density 0.05 → 0.82; chaotic-mode ensemble R² 0.01–0.03 → 0.79–1.00;
  balanced reconstruction 0.861 → 0.880 (untrained 0.881, i.e. no longer degrading); mask IoU
  0.170 → 0.111; slowness spread 31× preserved. `lambda_slow=10` is harmful — slowness must stay a
  weak tiebreak.
- **Also corrected:** the F-4 non-negativity limit is largely a GENERATOR artifact — every `φ_k` in
  `synthetic.py` is a strictly positive Gaussian bump, so 100.0% of cell pairs are positively
  correlated (mean +0.543) and no non-negative mask combination can decorrelate. Real ocean anomaly
  patterns have +/− lobes. CLAUDE.md's schema already says "smooth bumps / **low-wavenumber**" —
  the low-wavenumber (sign-varying) half was never implemented.
- **Errors:** two implementation traps, both fixed: Hydra `job.chdir`; and the recon decode built
  from raw (variance ~1e6) channels made the normal equations near-singular, whose garbage gradient
  froze training at ANY lambda.
- **Tests run:** ~12 more runs — energy/recon 2x2 ablation, λ_recon sweep (0.003…1), λ_slow sweep,
  per-term gradient-norm measurement, plus eval + figures on each.
- **Pending / next:** duplication (effective rank ~4 of 16) is the remaining gap, and it is the F-4
  ceiling. Proposed next step, NEEDS USER OK since it changes the testbed: add sign-varying
  low-wavenumber spatial patterns to the generator (dipoles / wave modes), per CLAUDE.md's own
  schema. Then modules 2 (latent predictor) and 3 (forecaster), and a learned classifier.

### 2026-08-26 — v0 encoder TRAINS; two structural limits found (D-016/017/018)

- **Built:** `src/probes/evaluate.py` (post-hoc readout: per-channel family labels + mask
  descriptors, population vs 1/3/6, effective rank, mask overlap, balanced reconstruction R²,
  per-hidden-mode ensemble R², TB images/series). Reworked the loss in `src/train/train.py`.
- **Fixed:** Hydra `job.chdir` (defaults to false in ≥1.2, so `artifacts.npz` / `metrics.json` /
  `tb/` had been landing in the repo ROOT, violating D-014). Stale root outputs moved to
  `.tmps/stale_root_outputs_20260826/`.
- **Collapse modes found and fixed (findings F-2, D-017):** soft `‖Cov−I‖` penalty → masks → 0;
  detached ZCA → all 16 channels merge into ONE (eff. rank 1.00/16); detached per-channel std →
  masks → 0 again. Root cause of all three: the constraint must be differentiable and
  scale-invariant *in the gradient*. New default `whitening: corr` trains stably.
- **Works now (findings F-3):** masks become small separated regional sensors (IoU 0.17);
  channels latch onto hidden modes unsupervised (|corr| up to 1.00, 0.99, 0.94); per-channel
  slowness spreads 31× across channels.
- **Two structural limits (findings F-4/F-5 → D-018, BLOCKING):**
  1. Non-negative masks cap decorrelation at off-diag corr² 0.244 / eff. rank 4.08 of 16,
     unchanged across λ_white 1→500; signed `tanh` masks reach 0.070 / 8.57. `Cov[s]=I` (D-011)
     is unreachable under D-005 selection semantics.
  2. Slowness training DESTROYS chaotic content: chaotic modes go from ensemble R² ≈ 1.0
     (untrained masks) to 0.01–0.03; population 0/14/2 vs the 1/3/6 target; balanced
     reconstruction R² 0.863 trained vs 0.881 untrained.
- **Errors:** none outstanding. **Tests run:** ~12 training runs (see `.tmps/runs/`), each with
  the eval probe; λ_white sweep 1/20/100/500; signed-mask ablation.
- **Pending / next:** USER DECISION on D-018 (a) signed selection, (b) explicit population
  objective instead of mean-slowness, (c) reconstruction/coverage term. Then rebuild the
  objective. Classifier is still the heuristic labeller in the probe (6/10 on known families —
  it mislabels the slow OU mode as cyclic and 2 chaotic modes as cyclic); a learned classifier
  is unbuilt, as is the latent predictor (module 2) and forecaster (module 3).

### 2026-08-26 — Phase L smoke test PASS (generator + K=16 encoder handshake)

- **Env (Phase L.1):** `oceanai` active; torch 2.13.0+cu126; CUDA True; NVIDIA H100 PCIe.
- **Built:** SOPs `memory/sop/00_pipeline_overview.md` + `01_synthetic_generator.md`;
  `src/data/synthetic.py` (multi-scale generator), `src/models/encoder.py` (PROVISIONAL
  soft-binary K=16 encoder), `src/probes/smoke.py`.
- **Smoke result (Phase L.2 handshake), seed=0:**
  - field (2000, 2, 64, 64) float32, finite; per-var standardized (SSH std=1.001, SST std=1.001).
  - 10 latent modes = 1 stationary / 3 cyclic / 6 chaotic (hidden; scale-organized large/med/small).
  - encoder K=16 on cuda → S (2000, 16), finite, std≈0.637.
  - slowness loss=0.0493; mask-grad norm=1.31e-4, finite & nonzero → gradients reach masks.
  - **SMOKE TEST: PASS.** Scalars saved to .tmps/smoke_S.npz (dir renamed per D-014).
- **Errors:** none.
- **Pending / next:** confirm the REAL encoder design (soft-binary anneal vs STE; domain;
  s_i normalization; whitening hard vs soft) → build the actual training loop (online pairwise
  SGD, slowness + whitening). Then classifier + collective metric. Then deeper K.
- **Still open:** concrete collective-encoding metric; Phase B Research pass.

### 2026-08-26 — Kernel-semantics reframe (D-013); resuming toward Phase L

- **Context:** Resumed after server shutdown; recovered latest point from living memory (state
  consistent, no loss). Discussed generator design ahead of Phase L.
- **Done:** Recorded **D-013** — masks are multi-scale regional "sensors" (overlap OK, NOT
  orthogonal unmixers); "deep" = many channels; success = collective encoding quality, not
  per-kernel isolation. Corrects an earlier agent unmixer framing. Generator drops
  spatial-separability requirement → spatially heterogeneous, multi-scale dynamics; latent modes
  kept as a HIDDEN answer key only.
- **Errors:** none. **Tests run:** none (no code yet).
- **Pending / next:** confirm the revised generator recipe (multi-scale heterogeneous field),
  then Phase L: write generator SOP in ./architecture/ (now `memory/sop/`) → generator + smoke-test
  probe in ./src/.
  Still open: concrete collective metric, K.

### 2026-08-24 — Memory scaffolding initialized (Phase A)

- **Done:** Read LLMAIProjectInstruction.md (authoritative spec). Surveyed project root.
- **Existing before this session:** `LLMAIProjectInstruction.md`, `environment.yaml`,
  `.claude/` (agent config).
- **Created this session:** `./memory/` with `task_plan.md`, `findings.md`, `progress.md`,
  `decisions.md` (this scaffolding only).
- **Not created (deliberately, per task + spec):** `./src/`, `./architecture/`, `CLAUDE.md`,
  `.env`, `./tmp/` (ephemeral, create when needed). No code written.
- **Errors:** none.
- **Tests run:** none (no code yet).
- **Pending / next:** Phase B Blueprint Discovery — ask the five discovery questions one at a
  time (see ./memory/task_plan.md), then define the Data-First JSON schema.

### 2026-08-24 — Framework adapted for PyTorch signal-processing NN project

- **User directive:** project is signal processing via neural networks (Python/PyTorch);
  "adjust some useless part for coding in the instruction"; details to follow step by step.
- **Done:** Rewrote `LLMAIProjectInstruction.md` in place — reframed SaaS-integration parts
  (Discovery Q2/Q4/Q5, Phase L handshakes, Phase S payload) toward data/tensors/training;
  added Reproducibility principle; standardized ephemeral dir on `./tmp/`. See decisions.md D-002.
- **Memory synced:** task_plan.md (reframed discovery questions + checklist), decisions.md
  (D-002 + pending list), this log.
- **Errors:** none. **Tests run:** none (no code yet).
- **Pending / next:** begin Discovery — awaiting user's step-by-step details, starting with
  North Star (target metric).

### 2026-08-25 — Testbed pivot + Data-First schema + CLAUDE.md created

- **Done:** Dropped Lorenz (no spatial field); adopted synthetic SSH/SST spatio-temporal
  testbed (D-009). Confirmed params: `field[2000,2,64,64]`, 10 ground-truth modes = 1/3/6
  (stationary/cyclic/chaotic), K=16 kernels. Created **CLAUDE.md** with the Data-First schema
  (Milestone 0 testbed + Milestone 1 ocean) and open design points.
- **Decisions:** D-005 (kernel=selection), D-006 superseded, D-007 (curriculum), D-008 (ocean
  shape), D-009 (synthetic testbed). Kernel parameterization leaning soft-binary anneal / STE.
- **Errors:** none. **Tests run:** none (no code yet).
- **Pending / next:** Phase L — build synthetic generator + smoke-test probe in ./src/. Still
  open: mask domain (full-volume vs per-layer), s_i normalization, per-scalar predictor +
  history window, distribution-matching objective, concrete success metric.

### 2026-08-24 — Emulator concept captured from venn_idea.txt

- **Done:** Read `venn_idea.txt`; recorded the authoritative emulator architecture in
  findings.md and decisions.md (D-004). Supersedes the earlier verbal "retrieval vs decoder"
  framing. Core: global binary-mask kernels → independent 1D scalar channels → learned
  temporal classifier (stationary/cyclic/chaotic, post-hoc & reversible) → kernel
  self-organization toward a target family distribution.
- **Errors:** none. **Tests run:** none (no code yet).
- **Pending / next:** resolve open questions (binary-mask learning method, target-distribution
  loss, per-scalar predictor, spatial reconstruction) and the Data-First tensor schema
  (K kernels, grid, variables, time sampling). Awaiting user's next step-by-step detail.
