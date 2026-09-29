# progress.md — What Was Done, Errors, Tests, Results

> Living project memory (per .claude/LLMAI_venn_init.md → FILE STRUCTURE / AGENT BEHAVIOR SUMMARY).
> Purpose: status updates — what's done / pending, errors hit, tests run, and results.
> Long-form per-session narratives (incl. dead ends) live in the "Session narratives" section at the end of this file.

## Log

### 2026-09-29 — Memory made flat: `memory/sop/` and `memory/sessions/` folded into the core files

- **User directive:** remove `memory/sop/` and `memory/sessions/` on every branch and integrate their content
  into the existing memory markdown files, with no new files; edit so later merges/rebases do not conflict.
- **Done:** the SOPs (00, 01, 02, 02a, 03, 04) are now sections at the end of `decisions.md`; the session
  narratives and their convention are the "Session narratives" section at the end of this file. Content moved
  verbatim (headings demoted; path references repointed). Union across branches: SOP 02's `train.t_train`
  section and SOP 04 came from `history-forecastor`, F-20 and the 2026-09-29 session addendum from `main`.
- **Conflict avoidance:** every `memory/*.md` is byte-identical on `main`, `stationary-observer` and
  `history-forecastor`, so the memory directory no longer differs between branches.
- **Open:** SOP 02 cites "D-030/D-031", but no D-031 entry exists yet.

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

- **Built (SOP first, per the Golden Rule):** SOP 03 (`decisions.md`), then
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
- **Built:** SOP 00 + SOP 01 (`decisions.md`);
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
  then Phase L: write generator SOP in ./architecture/ (now the SOP sections of `decisions.md`) → generator + smoke-test
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

---

## Session narratives (long-form; formerly the Session narratives section of `progress.md`)

Convention (user request, 2026-08-28; home moved here 2026-09-29): each working session gets a dated
long-form record, so the trail survives a shutdown and a new session can resume without re-deriving
anything. The `## Log` above stays the short chronological entry; `decisions.md` holds the D-xxx
rationale and the SOPs; `findings.md` holds the F-x evidence. These narratives tie them together —
including the dead ends, which the other files deliberately do not record.

- **Heading:** `### YYYY-MM-DD — short topic` (start date of the session), oldest first, appended at the
  END of this section.
- **Contents:** what was asked, what was built, what was measured, what broke, decisions/findings produced
  (cross-referenced to D-xxx / F-x), files touched, commands to reproduce, and the open thread to pick up.
- Raw Claude Code transcripts (JSONL) live outside the repo in
  `~/.claude/projects/-home-sysadmin-jlee-venn/<session-uuid>.jsonl`; each record names its own transcript.

Note: `memory/` was originally git-ignored, which is why a tracked `architecture/` copy once existed. As of
2026-09-28 `memory/` is the tracked home of the project record and `architecture/` is retired. As of
2026-09-29 `memory/` is flat: the SOPs live in `decisions.md`, the session narratives here.

| session | topic |
|---|---|
| 2026-08-26 | v0 encoder objective made to work (D-016…D-020) + generator rev2 (D-021) |
| 2026-08-29 | module 2 latent predictor (D-022, D-023, F-9…F-12 module-2 track, F-18) |
| 2026-08-31 | stationary observer, timescale ladder (D-024…D-026) |
| 2026-09-01 | stationary-observer merge handoff (was `architecture/README.md`) |
| 2026-09-28 | module 2: trajectory as memory, "sight", W8 → one model (was `03e`) |

### Session 2026-08-26 → 08-28 — v0 encoder objective made to work, then generator rev2

**Transcript:** `~/.claude/projects/-home-sysadmin-jlee-venn/2f0d3a1d-a1b6-48bc-89e5-7e9017ed5e43.jsonl`
**Produced:** D-016 … D-021, findings F-1 … F-8.
**Phase:** L complete → early S (STYLIZE). Modules 2 (latent predictor) and 3 (forecaster) still unbuilt.

---

#### 1. Where the session started

`src/train/train.py` + `config/config.yaml` existed from the previous session. The hard-whitening
fix had been *written into config but never run*. The last actual run had collapsed
(`var≈2.5e-6`, `white=16`). No evaluation existed beyond training-loss scalars, and no figures.

#### 2. The arc, in order

##### 2a. Three collapses, one root cause (D-017, F-2)

Ran the untested config and chased three separate failures:

| variant | what happened |
|---|---|
| `whitening: soft` (`‖Cov−I‖_F²`) | masks → 0; `var→2.5e-6`, `white→16 = ‖0−I‖_F²` |
| `whitening: hard` (detached ZCA) | all 16 channels merged into ONE; **effective rank 1.00/16** |
| `corr` with a DETACHED per-channel std | masks → 0 again; `var` 1e6 → 1e-40 in 150 steps |

**Root cause of all three: the constraint must be differentiable and scale-invariant *in the
gradient*.** Autograd treats a `.detach()`-ed variance as a constant, so a "scale-free" loss
written with a detached std is not scale-free to the optimizer, and a detached whitening matrix
imposes no cost on redundancy at all. Fix = SFA's Rayleigh quotient plus an explicit off-diagonal
correlation penalty, everything differentiable, no matrix inverse:

```
L = λ_slow·mean_i var(Δs_i)/var(s_i) + λ_white·mean_{i≠j} corr_ij² + λ_var·hinge
```

Also fixed here: Hydra `job.chdir` (defaults false in ≥1.2, so artifacts had been landing in the
repo ROOT, violating D-014), and a bad init (D-016) — the old `0.01*randn` logits made every mask
≈0.5 everywhere, so all channels read the same spatial mean and the correlation matrix was rank-1
from step 0 (`cond = K/eps = 16000`). Replaced with a multi-scale smooth random init. The
temperature anneal was turned OFF (it saturated the sigmoids and froze the masks before they
organized: off-diag corr² 0.53 annealed vs 0.27 not).

##### 2b. First honest readout → two apparent structural limits (F-4, F-5)

Built `src/probes/evaluate.py` (per-channel descriptors + family labels, population, effective
rank, mask overlap, balanced reconstruction R², per-hidden-mode ensemble R²). It said:

- decorrelation saturated at off-diag corr² **0.244**, effective rank **4.08/16**, and λ_white
  from 1 → 500 changed *nothing*;
- slowness training **destroyed the chaotic modes**: ensemble R² 1.0 (untrained masks!) → 0.01–0.03.

A signed-`tanh` mask ablation reached 0.070 / 8.57, so I concluded non-negativity was the binding
constraint and proposed possibly revising D-005. **That conclusion was later shown wrong — see §2e.**

##### 2c. USER RULE: never judge from metrics, always visualize (F-6)

> *"don't judge the result your self. Always make visualization on it."*

Built `src/probes/plots.py`. The first figure overturned the metric-based diagnosis immediately:
**the kernels had migrated to the dead corners of the domain** — the lowest-energy cells — because
`var(Δs)/var(s)` is a *ratio*, blind to amplitude: a tiny, almost purely slow tail in an empty
corner scores near-perfectly, while an energetic region carries a mix of slow and fast content.
That one mechanism explained the duplication, the lost chaotic modes, and the sub-random
reconstruction simultaneously. No scalar in the eval had surfaced it.

##### 2d. Energy + coverage terms (D-019, F-7)

```
L_energy = mean_i relu(1 − e_i/e_ref)²,  e_i = var(s_i)/count_i²      # per-channel: be ON SIGNAL
L_recon  = 1 − R² of the best per-batch linear decode of the field    # collective: SPAN the field
```

`count²` makes `e_i` independent of mask size (it is the mean pairwise covariance of the selected
cells). `L_recon` also penalizes redundancy — a duplicate channel buys no reduction. Results: min
energy density 0.05 → 0.82, chaotic-mode ensemble R² 0.01–0.03 → **0.79–1.00**, reconstruction
stopped degrading.

**Two traps worth remembering:**
1. The decode must use *standardized* channels. With raw `s` (variance ~1e6) the normal-equation
   matrix has entries ~1e9, the ridge is negligible, and with channels correlated at 0.99 the solve
   is near-singular — its garbage gradient froze training at *any* λ (λ=0.003 already did).
2. **Weight multi-term losses by matching GRADIENT norms, not loss values.** Measured at init:
   `|g_slow|=6.7e-6`, `|g_white|=1.7e-4`, `|g_energy|=8.8e-4`, `|g_recon|=5.0e-4`, `|g_size|=5.9e-3`
   — the values ranked them in the *opposite* order (l_recon: smallest value, largest gradient).

Also learned: **`lambda_slow` must stay small.** At 10 it collapses diversity (eff rank 2.95,
spread 31× → 2.3×). Slowness is a weak tiebreak, not a driver — somewhat against the D-010/D-011
framing.

Answering the user's direct question — is the energy term enough alone? **No.** Energy-only leaves
the chaotic modes at ensemble R² 0.01–0.06, same as baseline. Both terms are needed.

##### 2e. Housekeeping + scale ladder (D-020)

User directives: remove the git commit (kept every file on disk; repo now has no commits), the
`.gitignore` now excludes `.claude/` and `memory/`, and train+eval TensorBoard merged into ONE
`<run>/tb/` per run. User also observed masks were shrinking — confirmed: from a ~4095-cell init to
a **median of 28 cells**, and *my energy term had made it worse* (median 104 → 28) because mean
pairwise covariance is maximized by a tiny coherent patch. Fix = per-channel geometric footprint
ladder (`L_size`), a tolerance-band hinge, not a pin. Footprints then spanned 8 → 1143 cells and
mask IoU halved (0.111 → 0.061) at no cost to any other metric.

##### 2f. Field visualization → three generator defects (F-8) → rev2 (D-021)

User asked to actually *see* the testbed. Built `src/probes/plots_field.py` (snapshots, hidden-mode
answer key, kernel-activation map, and `field.gif` with kernel outlines pulsing at their
instantaneous activation). It exposed three defects nobody had seen:

1. **Dead borders** — mode centres drawn inside a 12-cell margin + all-local patterns left ~25% of
   the domain at near-zero variance. *The F-6 trap was built into the testbed.*
2. **SSH ≈ SST** — cell corr 0.86, domain-mean **0.981**. A 3-step lag is nothing against periods
   of 60–300, so `V=2` was near-redundant.
3. **All-positive φ_k** — visual confirmation of F-4's arithmetic: 100% of cell pairs positively
   correlated, so decorrelation was *arithmetically* impossible for a non-negative mask.

Fixes (all back-compatible; `GenConfig` defaults = legacy, new values in `config/config.yaml`):
full-domain centres + periodic x; SST as an AR(1) response to SSH forcing + private modes per
variable; zero-mean band, dipoles, and low-wavenumber wave packets.

**Result — the "structural limit" dissolved:**

| metric | legacy field | rev2 field |
|---|---|---|
| off-diag corr² | 0.259 (saturated, λ-independent) | **0.063** |
| effective rank | 3.89 / 16 | **9.23 / 16** |
| mask overlap IoU | 0.111 | **0.024** |
| balanced recon R² | 0.880 | **0.989** |
| slowness spread | 30.7× | **170×** |
| hidden modes (ensemble R²) | chaotic 0.79–1.00 | **all ten 0.80–1.00** |

**⇒ D-005 stands; D-018 is resolved without changing it.** rev2 with plain non-negative masks
(9.23) beats the signed-`tanh` ablation on the old field (8.57). The binding constraint was the
TESTBED, not the mask semantics. My §2b conclusion was wrong and is corrected here.

**The visualization rule paid off a second time:** the first rev2 used domain-filling waves and
scored the best numbers of the whole session (eff rank 9.56) while turning the field into a global
interference plaid — an un-enveloped wave carries |φ|=1 over all 4096 cells and swamps every
localized structure. Only `field_snapshots.png` showed it. Fixed with regional wave packets, at a
cost of 0.3 effective rank.

---

#### 3. Files created / changed this session

| file | state |
|---|---|
| `src/train/train.py` | loss rewritten: `corr` whitening + energy + recon + size + hinge; grad clip |
| `src/models/encoder.py` | multiscale/random/flat init, `signed` ablation flag, `soft_count()` |
| `src/data/synthetic.py` | rev2 patterns (dipole, wave packet, zero-mean band), periodic x, AR1 SST, private modes |
| `src/probes/evaluate.py` | NEW — post-hoc readout → `eval.json` + TB |
| `src/probes/plots.py` | NEW — masks / masks-over-energy / features / feature-variance |
| `src/probes/plots_field.py` | NEW — field snapshots / hidden modes / kernel activation / `field.gif` |
| `config/config.yaml` | all of the above as Hydra knobs; `hydra.job.chdir: true` |
| the SOP sections of `decisions.md`01_…md`, `02_…md` | SOPs updated ahead of each code change (Golden Rule) |
| `memory/{decisions,findings,progress}.md` | D-016…D-021, F-1…F-8, log entries |

#### 4. Reproduce

```bash
conda run -n oceanai python -m src.train.train                    # ~5000 steps, seconds on the H100
conda run -n oceanai python -m src.probes.evaluate                # newest run → eval.json + TB
conda run -n oceanai python -m src.probes.plots                   # 4 figures → <run>/figs/
conda run -n oceanai python -m src.probes.plots_field             # field figures + field.gif
```
Best run of the session: `.tmps/runs/20260827_151413/` (figures in its `figs/`).
Runs are ephemeral under `.tmps/` (D-014); nothing has been promoted to `results/`.

#### 5. Open thread — pick up here

1. **`L_energy` now fights the size ladder** at the top rung (energy 0.34, min `e/e_ref` 0.02): the
   ladder asks for a 30%-of-domain footprint, but the rev2 field has sign structure *inside* any
   such footprint, so a large mask partly cancels itself. Options: exempt the top rungs, or score
   `e_i` against what a RANDOM mask of the same size would get.
2. **The family labeller is crude** — a spectral-peak heuristic in `evaluate.py` that reproduces
   only 6/10 known families and calls almost everything "cyclic" (population 0/14/2). D-004 wants a
   learned post-hoc classifier; unbuilt. Note the user has relaxed the 1/3/6 target: *"if all
   kernel can help to extract meaningful dynamics it is OK"* — so collective encoding quality is
   the criterion, not the proportion.
3. **Module 2 (latent predictor)** and **module 3 (forecaster)** of D-012 are not started. The
   encoder is now good enough to feed them (effective rank 9.23/16, balanced R² 0.989).
4. Never revisited: K > 16 ("deep = many channels", D-013), straight-through binary masks, and the
   Phase B research pass (`findings.md` still marks it PENDING).

#### 6. Standing rules confirmed this session

- **Always visualize before judging a result** — it overturned the diagnosis twice here.
- **The user owns git.** No commits; work stays local. `memory/` and `.claude/` are git-ignored but
  still maintained. (Update 2026-09-28: `memory/` is now the tracked home of the project record;
  `architecture/` was retired and merged into it.)
- Ephemeral output → `.tmps/`; `results/` only with explicit authorization; Hydra owns every
  hyperparameter; ONE TensorBoard dir per run.
- Update the SOP in the SOP sections of `decisions.md` *before* changing the code it describes.

### 2026-08-29 — Module 2: the latent predictor (observer → forecaster in the compressed domain)

#### Where this session started

Modules 1 (selection-mask encoder) and its probes existed and worked; the generator was at rev2
(D-021). Modules 2 and 3 were unbuilt. The user's ask: *"build latent forecaster who predicts next
feature in compressed domain"* — i.e. D-012 module 2, on the frozen channels, nothing spatial.

#### What got built

Following the Golden Rule, the SOP went first: SOP 03 (`decisions.md`), then

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

#### The design argument I want to keep

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

#### What the numbers said (and where they lied)

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

#### Caveat that limits every per-family number here

The family labels come from the module-1 heuristic labeller, which is known to be wrong
(population 0/14/2 against a 1/3/6 target — it calls almost everything cyclic). The per-family
aggregates inherit that error. A real classifier is still unbuilt, and it now blocks the
interpretation of module 2's results, not just module 1's.

#### Dead ends / things I checked and discarded

- Considered scoring the model by MSE skill alone — abandoned once the linear ablation exposed the
  mean-collapse. `var_ratio` is now a first-class metric, and the SOP says rank by it FIRST.
- Considered letting the predictor see all K channels (a joint latent model). Rejected for now: it
  contradicts D-012's "per-channel 1D" and would make the family labels meaningless. Left in as the
  `per_channel: false` ablation switch rather than as a silent default.

#### Environment / process notes

- The background-job harness required isolation, so all code was written in the git worktree
  `.claude/worktrees/latent-predictor` (branch `worktree-latent-predictor`) instead of the main
  checkout. Git remains the user's (standing rule) — nothing was committed to `main`.
- Ran in `oceanai` (CUDA available). Encoder run reused: `.tmps/runs/20260827_151413`, copied into
  the worktree's own `.tmps/` so nothing wrote into the user's checkout.
- `.gitignore` in the repo is committed WITH unresolved merge-conflict markers
  (`<<<<<<< HEAD` / `=======` / `>>>>>>> 7afcabf`). Harmless today (both blocks' patterns are read
  as literal patterns, so `.tmps/` is still ignored) but it should be cleaned up.

#### Next

1. A real classifier — it now blocks module 2's interpretation as well as module 1's.
2. Module 3 (forecaster by gradient inversion, `x(t+1) = x(t) + M⁺(ŝ − M·x)`). The interface is
   ready: `predictor.rollout` returns exactly the `ŝ` it needs.
3. Re-check whether an encoder trained with the predictor in the loop (rather than frozen) changes
   the channels — currently module 1 optimizes slowness, which is *why* linear AR does so well.

#### Late in the session — two standing rules restated by the user, and one decision handed back

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

#### Final state after the D-023 switch

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

#### A figure that was quietly lying, and the fix (user caught it)

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

#### 2026-08-31 (cont.) — the three-expert proposal, tested and rejected

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

#### 2026-09-27 — module 2 re-run on the current encoder

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

### 2026-08-31 — The labeller was the bug; the timescale ladder is the lever

> Long-form record (dead ends included), per the Session narratives convention in `progress.md`.
>
> Session span 2026-08-31 → 2026-09-01. Chronological record including dead ends and my own corrections.
> Three parts, in order: the timescale ladder (D-024), the user's "stationary = constant" ruling (D-025),
> and the global-structure / observables reframe (D-026). Later parts CORRECT earlier ones — read to the
> end before acting on a number from part 1.
> (A tracked copy of this record lived at `architecture/01e_session_narrative.md` until `architecture/`
> was retired on 2026-09-28.)
> Code: git worktree `.claude/worktrees/stationary-observer`, branch `worktree-stationary-observer`.
> Nothing committed — git is the user's.

#### The ask

> "From built encoder (observer) I want to develope to capture stationary mode too. Deepen the
> kernel increase some visibility of slow mode, but still not enough to say that the kernel number
> is the lever to detect every dynamical mode."

Two claims to test: (1) the observer does not capture the stationary mode; (2) K is not the lever.
Claim 2 held. Claim 1 turned out to be false — and the reason it looked true is the interesting part.

#### What I did first: read the K sweep instead of trusting the summary

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

#### The fix: `src/probes/family.py`

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

#### Why K cannot be the lever (F-12)

`L_slow = mean_i var(Δs_i)/var(s_i)` is ONE objective shared by all K channels. They compete for
the same globally-slowest content, so channel K+1 buys a near-duplicate of the dominant regime, not
a new timescale. The measured proportions barely move from K=4 to K=64. The user's intuition was
right, and now there is a mechanism and a number behind it.

#### The build: D-024 spectral band ladder

Same move D-020 made for mask size: stop asking every channel for the same thing, ASSIGN each a
target and penalize only outside it. `src/train/spectral.py`:

- `rung_roles(K, pop_target)` — 1/3/6 scaled to K (K=16 -> 2 slow / 5 cyclic / 9 fast);
- `L_band` — in-band power fraction per assigned rFFT band;
- `L_line` — line-vs-hump: cyclic rungs must BE a line, slow/fast rungs must NOT;
- `L_mem` — slow rungs only, lag-64 autocorrelation ≥ 0.6.

Computed on random contiguous WINDOWS sliced from `enc(field)`, because a lag-1 pair minibatch says
nothing about a period-300 cycle. Encoding the whole series each step is cheap (one einsum over
T×8192), and the stochasticity D-011 wants now comes from the window starts.

##### Dead ends and corrections along the way

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

#### Result (K=16, 5000 steps, seed 0, all else equal)

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

#### What the figures showed that the metrics did not

`figs/features.png`, both runs. The ladder bank is visibly stratified — smooth wandering at ch0-6, a
textbook sinusoid at ch4, broadband hash at ch7-15 — while the baseline's families are interleaved
down the channel index in no order. The same figure is what shows the failure honestly: cyclic rungs
2/3/5/6 wander instead of oscillating. No scalar in the run said that as clearly.

#### Open, in priority order

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

### 2026-09-01 (same session, continued) — the user corrected the DEFINITION, and it invalidated my taxonomy

#### The correction

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

#### What changed (three places, because the definition touches all of them)

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

#### Mistake caught before it cost a run

I first set `flat_target = 0.8`. That corresponds to `amp_ratio = 0.25` — a channel that still
visibly moves — so the hinge would have gone quiet *before* the channel could pass the labeller's
0.05 cut. The objective and the readout have to agree on where the bar is: `r = 1/(1 + amp_ratio)`,
so flat (`< 0.05`) means `r > 0.952`, hence 0.95.

#### Result (rev3 field, K=16, 5000 steps, seed 0)

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

#### A figure was lying again (second time on this project, cf. F-6)

`plots.py` standardized every channel before plotting — dividing out the std, the very quantity that
defines flatness. A channel fluctuating at 2% of its level was drawn as dynamic as a sinusoid, which
is exactly how the ladder run first read to me as "ch0/ch1 are oscillating". Now scaled by RMS about
zero with `amp_ratio` printed per lane: ch0/ch1 draw as flat lines, ch2–6 as clean cycles, ch7–15 as
broadband hash. **Check what a plot normalizes by before trusting it.**

#### Null result

De-aligning the size and timescale ladders (`slow_size_frac=0.08`, run `20260901_080655`) to un-pin
the flat rungs from the largest footprints: footprint halved 2590 → 1318 cells, alignment with the
injected pattern UNCHANGED (|corr| 0.27 either way), role obedience fell 1.00 → 0.94. So the flat
channels' diffuseness is not the size ladder's fault. Default back to 0.0, reasoning kept in the
config comment so it is not re-tried blind.

#### Open

1. **Does the observer need to ISOLATE the static pattern, or just report a stable level?** The flat
   channels are flat (amp_ratio 0.023) but align with `phi_0` at only |corr| 0.27 — settling on the
   pattern's negative lobe or a diffuse static average. Needs a user call.
2. **Module 2 has never seen a constant channel** and normalizes per channel — it will divide by
   ~0. Must be handled before re-running the predictor on a rev3 encoder.
3. `slow_period_min` / `cyclic_period_max` were tuned when the slow band was meant to hold the OU
   drift. The slow band now holds no stationary mode at all, so those knobs want revisiting.
4. A learned classifier is now worth building — the labeller is finally trustworthy.

---

### 2026-09-01 (cont.) — "let's slow down": two frame corrections

The user stopped the build to fix the frame, and both corrections landed on things I had just built.

#### Correction 1 — classify by GLOBAL STRUCTURE, not the residue

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

#### Correction 2 — the observer is not a mode-recovery device

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

#### Asked before building

Four questions, all answered by the user: explicit 4-part decomposition; NO persistence check
(flatness over the record is enough); cancellation allowed but not encouraged (no term rewards it);
1/3/6 is a design target for the bank, not a recovery score. Worth keeping that habit — three of
the four answers ruled out work I would otherwise have done.

#### Open

1. Harmonic folding in `decompose`: a non-sinusoidal periodic signal leaks into `residual` and biases
   the vote toward chaotic.
2. Module 2 normalizes per channel and will divide by ~0 on a constant channel.
3. A learned classifier is now worth building — the labeller finally matches the project's own
   definitions.
4. `slow_period_min` / `cyclic_period_max` were tuned when the slow band was meant to hold the OU
   drift; the slow band now holds no stationary mode at all.

### Handoff — `worktree-stationary-observer` (2026-08-31 → 2026-09-01)

> Saved here on 2026-09-28 from `architecture/README.md` when `architecture/` was retired and merged into
> `memory/`. The file table below is updated to the new `memory/` paths; the rest is as written at merge time.

Self-contained record of this session's decisions, findings and measurements, written for the
**merge back into `main`**.

#### Why these files exist here and not in `memory/`

*(Historical — as written 2026-09-01.)* `.gitignore` ignores both `memory/` and `.claude/`, so the living
memory does NOT travel with a branch. Everything needed to review and merge this work is therefore
duplicated here, in a tracked directory. `memory/findings.md`, `memory/decisions.md`, `memory/progress.md`
and `progress.md` → Session narratives, 2026-08-31 (stationary-observer-and-timescale-ladder) in the MAIN checkout carry
the same content (they are shared across worktrees on disk).

**Update 2026-09-28:** `memory/` is now the tracked home of the project record; `architecture/` is retired
and its contents live at the paths below.

| file (now) | was | contents |
|---|---|---|
| `memory/decisions.md` | `01_decisions.md` → `01a_decisions.md` | D-024, D-025, D-026 — full text, with what each supersedes |
| `memory/findings.md` | `02_findings.md` → `01b_findings.md` | F-11 … F-14 — what was measured and what it invalidated |
| `memory/measurements.md` | `03_measurements.md` → `01c_measurements.md` | every number, with the run ID that produced it |
| `memory/open_and_rejected.md` | `04_open_and_rejected.md` → `01d_…` | open questions, plus alternatives tested and rejected |
| `progress.md` → Session narratives, 2026-08-31 (stationary-observer-and-timescale-ladder) | `05_session_narrative.md` → `01e_…` | the story |

(The `04` row's original text read "open questions, plus alternatives already tested and rejected (do not
re-try)"; the `05` row's read "the chronological story, including dead ends and my own corrections".)

#### Code changed on this branch

| path | status | what |
|---|---|---|
| `src/probes/family.py` | **new** | the family labeller: `amp_ratio` flat gate + 4-part structural decomposition. Single source of truth for a family label. |
| `src/train/spectral.py` | **new** | spectral band ladder (`band_plan`, `spectral_terms`), `level_term` (flatness), `memory_term` (ablation) |
| `src/probes/validate_labeller.py` | **new** | validates the labeller against the hidden truth over several seeds, and re-scores archived runs. Run this FIRST after any labeller change (see F-11). |
| `src/data/synthetic.py` | modified | generator rev3: `stationary_constant` — a flat, un-standardized mode 0, plus the SST bypass for constant modes |
| `src/train/train.py` | modified | ladder wiring, per-rung exemptions from `l_var`/`l_energy`/`L_band`/`L_line`, `roles` saved to artifacts |
| `src/probes/evaluate.py` | modified | imports the labeller, prints structure shares, `stationary_observer` readout, constant-mode guards, design-vs-recovery wording |
| `src/probes/plots.py` | modified | features panel scaled by RMS (was standardized, which hid flatness) + `amp_ratio` per lane |
| `config/config.yaml` | modified | `data.stationary_constant`, the whole `train.spectral` block, all weights gradient-matched |
| `architecture/01_synthetic_generator.md` (now SOP 01 (`decisions.md`)) | modified | rev3 section |
| `architecture/02_encoder_training.md` (now SOP 02 (`decisions.md`)) | modified | D-024/D-025/D-026 sections |

#### Merge checklist

1. **Nothing is committed.** Git is the user's in this repo; the branch holds working-tree changes
   only. Commit or cherry-pick as preferred.
2. **Back-compatibility is preserved by dataclass defaults.** `GenConfig.stationary_constant`
   defaults to `False`, so every pre-2026-09-01 run regenerates its original field from its own
   saved config. The new value lives in `config/config.yaml`. `evaluate.py` depends on this.
3. **Re-scoring old runs changes their reported labels**, by design (D-025/D-026). Any figure or
   number quoted from before 2026-08-31 should be re-read with `validate_labeller.py`.
4. **`.gitignore` has unresolved merge-conflict markers** (`<<<<<<< HEAD`, `=======`,
   `>>>>>>> 7afcabf`). Pre-existing, harmless today, worth cleaning during the merge.
5. Run order to reproduce:
   ```
   conda run -n oceanai python -m src.probes.validate_labeller     # readout first
   conda run -n oceanai python -m src.train.train                  # ladder on by default
   conda run -n oceanai python -m src.probes.evaluate
   conda run -n oceanai python -m src.probes.plots
   ```

#### The one-paragraph summary

The encoder was never failing to see slow structure; the READOUT could not emit the label, and the
testbed's "stationary" mode was not stationary (it was standardized to unit variance, making it a
drift). Fixing the definition end-to-end — a constant mode in the generator, a flatness-gated
structural labeller, and an `L_level` term that makes a flat channel reachable instead of forbidden —
turns a bank that captured no stationary observable at all (baseline: flattest channel `amp_ratio`
0.778) into one where a dedicated channel sits flat at `amp_ratio` 0.023 and every channel lands in
its assigned family (`role_obedience` 1.00). The deeper reframing: **K is not the lever on which
dynamical families appear** (measured flat from K=4 to K=64) — the timescale must be ASSIGNED per
channel — and **families are properties of the observables the kernels build, not of the hidden
modes**, so mode recovery is a diagnostic, not the goal.

### 2026-09-28 — Module 2 reframing: observer trajectory as memory, "sight", two configurations

Discussion-only sessions (no code changed, no runs; nothing committed — git is the user's). First
session transcript:
`~/.claude/projects/-home-sysadmin-jlee-venn--claude-worktrees-latent-predictor/c70264ed-4168-4a0c-a933-ac431eb71e76.jsonl`
(worktree `latent-predictor`). The user closed it to continue the discussion in a new session, recorded
in the last section ("continued, new session").

> Provenance: first written in the worktree as `progress.md` → Session narratives, 2026-09-28 (predictor-memory-sight-theory)
> (that session could not write to the main checkout), then extended with the continuation session as
> the tracked copy `architecture/03e_module2_session_narrative.md`. Merged here 2026-09-28: this file in
> the main checkout's the Session narratives section of `progress.md` is now the home; `architecture/` is retired.

#### 1. Starting point — current predictor (recap given to the user)

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

#### 2. The user's argument (evolved over three turns — read in order)

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

#### 3. Weaknesses raised (the open list to discuss next)

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

#### 4. Proposed next measurements (NOT started; await the user's decision)

1. Analog forecaster: skill + amplitude vs sight (tests W5; the theory predicts a peak).
2. Single-channel vs joint 16-channel input, as a function of sight (tests W2).
3. (From turn 1, still valid) streaming state vs zero-start launches; training-window sweep
   8/16/32/64/128.

#### Files touched

- Auto-memory: `~/.claude/projects/-home-sysadmin-jlee-venn/memory/predictor-two-configs-theory.md`
  (+ line in MEMORY.md).
- This record. No repo code or docs changed.

### 2026-09-28 (continued, new session) — the collapse of (1) and (2) is the GOAL

#### The user's ruling on W8

> "W8: (1) and (2) may collapse. It is good news. I am not looking forward two different model.
> If it collapse, it means that my model is good to forecast."

W8 is therefore **not a weakness but a design goal**: ONE model, not two. Working form proposed:
attention over the observer-captured trajectory, with sight = the accessible archive, doing analog
retrieval (config 1) and within-window adaptation (config 2) in the same forward pass. A testable
statement of the collapse: attention weights land on past segments that are true dynamical analogs,
AND skill rises with sight up to an optimum (W5).

#### Caveats recorded with the ruling

1. **A collapse shows consistency, not skill.** The unified model must still beat, on amplitude-aware
   metrics (rank by `var_ratio` first, F-9): a non-learned analog forecaster, AR/linear baselines
   (cf. Zeng et al. 2023) and the current GRU (F-18: skill h1 +0.713, h64 +0.264, horizon ~36–40).
2. **The collapse gives up part of the original claim.** If (2) is retrieval, W6 applies to the one
   model: it reproduces recorded TYPES of extremes but cannot exceed the recorded maximum. W7
   (in-context learning of unseen regimes) disappears; W4 sharpens (analog pool = 1400 steps).
3. **Still open before any architecture:** W1 (memory valid for one observer version only,
   `flat98fix_seed0`) and W2 (per-channel vs joint 16-channel input) — both decide the model's input.

#### Proposed next measurement (NOT started; awaits the user's go)

Non-learned analog forecaster on the frozen encoder output, sight swept 8/16/32/64/128, per-channel
AND joint, scored on the val split `[1432,2000)` (skill + amplitude). It is the zero-parameter version
of the unified model, the baseline the attention model must beat, and a direct test of W5 and W2.
Outputs to `.tmps/` only.

#### Housekeeping noted, not done

- The main checkout (`1812638`) is older than this branch; `memory/` ≙ `architecture/` for the
  tracked record (user, 2026-09-28).
- Stale label still unfixed: SOP 03 and F-18 say "MLP (current default)" (D-028 made it the GRU).
  *(At the merge the `findings.md` F-18 row was relabelled "default at the time; D-023"; the SOP 03
  table, copied verbatim, still carries the stale label.)*
- `architecture/README.md` is still the stationary-observer handoff and does not list the 03* files.
  *(Resolved by the 2026-09-28 merge: `architecture/` retired; README → `2026-09-01_stationary-observer-
  merge-handoff.md`, 03/03a/03b/03e → SOP 03 (`decisions.md`), `decisions.md`, `findings.md`,
  this file.)*

### 2026-09-28 (continued) — W1 retired into A1; W1′ = sufficient observer

User: "I don't think that W1 is a problem. because the observer will be trained by a training data set.
Therefore I suppose that a dynamical system has a stable observer." Then: "I agree with sufficient observer."

- Agreed: a frozen observer is stable by construction; retraining = design change. Recorded as **A1** (D-029).
- What survives of W1 is **sufficiency**, not stability: the per-channel predictor (D-012) can only forecast
  dynamics some channel sees. Evidence: `fixavg` seeds 2 and 3 have no period-60 channel. Recorded as
  **W1′**, merged with W2 (D-029).
- Ranking of open weak points is now: **W1′(+W2)**, then W4 / W6 (W7 dropped with the W8 collapse).
- Test launched: analog forecaster × encoder seeds `flat98fix_seed{0..4}` (outputs in `.tmps/`).

#### Seed test result (F-19) — and a correction of my own argument

- My W1′ evidence ("seeds 2/3 blind to the period-60 cycle", from dominant FFT periods) was WRONG: their best
  single channel forecasts that mode at +0.99. Every seed's observer is sufficient for all three cyclic modes.
- Joint 16-channel analogs lose to one channel (dimension explosion, W5) — naive retrieval can't use joint info.
- Sight optimum is family-dependent (cyclic ↑ to 128, chaotic peak 16) — supports per-channel sight (W9).
- k=10 analogs: higher skill, amplitude 0.52 at h64 (W6 mean-collapse). k=1: honest amplitude, skill ≈ GRU at h64,
  far below it at h1.
- Figure: `.tmps/analog_seed/analog_seed_test.png`.

### 2026-09-29 (same discussion) — out of the train/inference scheme: streaming emulator (D-030)

- User: the input can be a single time step (observer trained on [0:2000], predictor fed 2001 to predict the
  next); the emulator is trained for one dynamic and keeps its own captured memory; after 10 years of
  training it is used from the next day by APPENDING states. Ruling: **weights frozen, only memory grows.**
- This answers my "input must be a window" refinement: the history is in the memory, not the input.
- Agreed consequence: evaluation must stream. Three departures found in the current tests: GRU zero-state
  restarts (F-18), frozen analog library (F-19), encoder trained on the full record incl. the scored period.
- Recorded as **D-030** with the streaming protocol (observer + predictor trained on [0,1400), stream
  1400→1999, predict-reveal-append-score). Not yet run.

#### First D-030 run (F-20), 2026-09-29

- Added feature flags `train.t_train` (encoder learns only from the past) and `predictor.select=last` (no
  val-peeking checkpoint); SOP 02/03 updated first; probe `src/probes/stream_eval.py`.
- Observer trained on the past generalizes to unseen days (GRU h1 +0.669 vs +0.672 full-record), but the
  longest cycle (P286) loses sufficiency on 3/5 seeds → W4.
- The never-reset GRU is worse than the zero-restart GRU: it was never trained to carry a long state →
  a D-030 predictor must be trained in streaming mode. Memory growth (analog) buys ~nothing over 600 steps.
- Noted: `src/probes/analog_seed_test.py` (F-19's probe) is missing from the worktree after the user's
  commits (never committed); its outputs in `.tmps/analog_seed/` remain.
