# decisions.md — Architectural Choices and Reasoning

> Living project memory (per LLMAIProjectInstruction.md → FILE STRUCTURE; also Phase S.4 Feedback
> and Phase T self-annealing loop feed decisions here).
> Purpose: key decisions and their rationale. One entry per decision.

## D-001 — Initialize memory scaffolding only (no src/architecture/code yet)

- **Date:** 2026-08-24
- **Decision:** Create only the four `./memory/` files. Do NOT create `./src/`,
  `./architecture/`, `CLAUDE.md`, or `.env` yet.
- **Rationale:** Per spec, phases run in order ARCHITECT → BLUEPRINT → LINK → STYLIZE → TRIGGER,
  and "when in doubt, update documentation first." Project scope (North Star, Integrations,
  Source of Truth, Delivery Payload, Behavioral Rules) is undecided — Phase B Discovery has not
  run. Follows the "Surgical Changes" and "Simplicity First" operating principles: no
  speculative structure, no fabricated project content.

## D-002 — Adapt the framework for a PyTorch signal-processing NN project

- **Date:** 2026-08-24
- **Decision:** Rewrote `LLMAIProjectInstruction.md` in place to tailor the generic
  LLM-agent framework to a Python/PyTorch signal-processing project (user directive:
  "adjust some useless part for coding"). Preserved the deterministic A.N.T. 3-layer
  build, the 5-phase order, and the operating principles. Removed/reframed the
  SaaS-integration machinery.
- **Changes:**
  - Discovery Q2 "Integrations (Slack/Notion/…)" → "Data & Signals" (input signal /
    target output). Q4 "Delivery Payload (Slack/Notion/email)" → "Deliverable"
    (checkpoint + metrics + plots). Q5 reframed to compute/reproducibility/coding rules.
  - Phase L "API credential handshakes" → environment/GPU/data connectivity + a
    forward/backward smoke test.
  - Phase S payload refinement (Slack/Notion/email) → metrics tables, plots, named
    checkpoints.
  - Data-First "JSON schema" → tensor schema (shape/dtype/units/sampling rate/channels),
    recorded in CLAUDE.md.
  - Added Operating Principle #6 Reproducibility (seeds, logged configs, re-creatable runs).
  - `.env` marked optional (only for W&B/HF tokens).
- **Rationale:** North Star is a target metric, source of truth is a dataset, delivery
  is a model artifact — no external SaaS. Keeps the framework useful without dead weight.
- **Ambiguity resolved:** the spec's `.tmp` vs `tmp` inconsistency → standardized on
  `./tmp/` everywhere in the rewrite.

## D-003 — Two-layer environment (base `oceanai` + tools venv)

- **Date:** 2026-08-24
- **Decision:** The project uses two separated environments. Documented in
  `LLMAIProjectInstruction.md` (new "ENVIRONMENT (TWO-LAYER)" section + file structure).
  - **Base (`oceanai`, conda / environment.yaml):** structural + necessary packages to
    DEVELOP the ML-based (ocean) signal emulator — PyTorch, scientific/data stack, CUDA.
    Kept lean and reproducible; source of truth for training/inference deps.
  - **Tools (separate venv, e.g. requirements-tools.txt):** auxiliary non-training tasks —
    visualization, TensorBoard, Jupyter notebook / localhost kernels, plotting.
    Isolated so viz deps never contaminate the base env; disposable/re-creatable.
- **Rule:** base env produces the science; tools env only observes/visualizes. When adding
  a package, decide its layer first. Never install tooling into base; never train from tools.
- **Rationale:** keeps the core training env minimal and reproducible; viz/notebook stacks
  are heavy and optional.

## D-004 — Emulator architecture (from venn_idea.txt)

- **Date:** 2026-08-24
- **Source:** `venn_idea.txt` (user's authoritative project description).
- **Decision (design chosen by user):**
  - Encode spatial fields via many **global binary-mask kernels**; each kernel → one
    **scalar time series** representing a global spatial pattern (`s_i(t)=<mask_i,field(t)>`).
  - Each scalar channel is an **independent 1D dynamical system — no inter-channel coupling**
    (deliberate: simplicity, interpretability, stability).
  - A small **learned temporal classifier** (1D CNN / tiny RNN) labels each series as
    **stationary / cyclic / chaotic**, **post hoc and reversibly**.
  - Kernels start as **random binary masks** (→ chaotic) and are modified so the **population
    distribution** of families matches a **target distribution** — population shaping, not
    per-kernel forcing. Yields a **self-organizing specialization loop**.
- **Full detail:** findings.md → "Project concept / emulator architecture (authoritative)".
- **Status:** architecture accepted as the project's direction. Implementation-level choices
  (binary-mask optimization method, target-distribution loss, per-scalar predictor, spatial
  reconstruction) are OPEN and listed as findings.md open questions — do not pre-decide.

## D-005 — Kernel = learned spatial SELECTION (not a CNN feature detector)

- **Date:** 2026-08-25
- **Decision (user):** The full-domain kernel's purpose is to make a **selection / "choice"
  ("vision") among the spatial data** — which spatial cells feed each scalar. This is the
  defining difference from a CNN (local, sliding, weight-shared feature detector).
- **Consequence:** weight parameterization MUST preserve selection semantics.
  - **Rejected:** free/unconstrained continuous kernel (plain CNN/linear) — it turns the
    choice into a weighted average and loses the selection meaning.
  - **Preferred (leaning, not locked):** soft-binary `sigmoid(w/T)` with temperature `T`
    annealed downward (soft selection early → hard {0,1} choice late); STE as the
    strict-binary alternative. Both keep true "learned selection" and fit the
    random-init → chaotic → self-organize dynamic.
- **Status:** direction agreed; exact scheme (anneal vs. STE vs. Gumbel) to confirm before
  coding the encoder.

## D-006 — Lorenz-first validation before ocean  [SUPERSEDED by D-009]

- **Date:** 2026-08-25
- **Decision (user):** Validate on the **Lorenz system** first. **Dropped same day** —
  Lorenz has no spatial field, so it would only test the temporal half, not the
  mask-encoder. Replaced by a synthetic spatio-temporal testbed (D-009).

## D-009 — Synthetic spatio-temporal testbed (2 pseudo-vars: SSH & SST)

- **Date:** 2026-08-25
- **Decision (user):** Milestone-0 testbed = **generated spatio-temporally correlated
  dynamics with 2 pseudo-variables imitating SSH and SST**, then test the model on it.
- **Rationale:** unlike Lorenz it has a spatial field → exercises the FULL pipeline; being
  synthetic, ground truth (spatial patterns + per-mode dynamical family) is known, so both
  the encoder and the classifier can be validated quantitatively.
- **Shape:** `field[time, 2, lat, lon]` (SSH, SST are surface → vardepth=2), same schema as ocean.
- **Proposed generator:** modal superposition `Σ_k a_k(t)·φ_k(x,y)` with each a_k assigned a
  known family (stationary/cyclic/chaotic); SSH & SST share modes for coupling. Confirm.
- **Open:** grid size, time length/sampling, mode/family mix, SSH↔SST coupling, K. Decide together.

## D-007 — Target family distribution is a curriculum (0/0/100 → natural)

- **Date:** 2026-08-25
- **Decision (user):** The stationary/cyclic/chaotic population target is not fixed.
  Start **0/0/100** (all chaotic, matching random binary init), hold until the emulator
  predicts a **reasonable ocean state**, then study/refine toward the **natural ocean
  signal proportion**.
- **Open:** concrete metric for the "reasonable ocean state" gate that triggers the shift.

## D-008 — Ocean input tensor shape

- **Date:** 2026-08-25
- **Decision (user):** Ocean input is `field[time, vardepth, lat, lon]`, `vardepth = var×depth`
  (channel axis = each variable at each depth as a 2D layer). `K` global selection kernels
  encode each state into `K` scalar time series.
- **Open:** mask over full `(vardepth,lat,lon)` volume vs per-layer; `s_i` normalization; `K`.

## D-010 — Learning objective: minimize temporal-neighbor gap of each mode

- **Date:** 2026-08-25
- **Decision (user):** Introduce a neural network trained to **minimize each scalar mode's gap
  between neighboring time steps** `L = Σ_i Σ_t d(s_i(t+1), s_i(t))`; gradients optimize the
  selection kernels, pulling modes from chaotic (random init) toward coherent/slow.
- **Agent-flagged, MUST resolve before coding (see findings.md):** pure gap-minimization is
  degenerate — collapses to constant/zero scalars (all stationary) and to K identical kernels.
  Requires **unit-variance + decorrelation/orthogonality** constraints (≈ Slow Feature Analysis).
  A governor (classifier + population target) is needed so not ALL modes are smoothed to
  stationary (else ~10/0/0 instead of 1/3/6).
- **NN role = (a) (user, 2026-08-25):** the **selection masks themselves are the learnable
  parameters** — the "network" is the differentiable selection-encoder (a masked linear
  projection). NO separate predictor network. The gap loss optimizes the masks directly.
- **Implication:** this is exactly **linear SFA with binary selection masks**. Degeneracy is
  now UNAVOIDABLE without constraints (a bare masked projection minimizing gap → zero/constant
  mask). ⇒ unit-variance + decorrelation constraints are MANDATORY, not optional.
- **Status:** NN role locked (a). Still OPEN before coding: exact anti-collapse constraints
  (confirm variance+decorrelation), distance metric (L1/L2), horizon (t,t+1 vs window), and the
  governor that preserves 6 chaotic modes (classifier + population target?).

## D-011 — Anti-collapse = decorrelation; training = online pairwise; diversity is EMERGENT

- **Date:** 2026-08-25
- **A2 (user, CONFIRMED):** anti-collapse constraint = **decorrelation as full whitening
  `Cov[s] = I`**. Unit-variance diagonal forbids the zero/constant collapse (and empty masks);
  zero off-diagonal keeps the K kernels distinct. Both collapse modes closed by one constraint.
- **A3 (user):** training is **online / streaming pairwise**: minimize gap on consecutive
  time-step pair `(t_k, t_{k+1})`, update masks, advance to `(t_{k+1}, t_{k+2})` carrying the
  updated masks, never revisit the old pair; repeat over many epochs. Goal: after many epochs,
  the K kernels show a SPREAD of stationary / cyclic / chaotic time-series lines.
- **Emergent-governor mechanism (agent synthesis, user's intent):** the slow subspace is small;
  decorrelation forces the first kernels into the slow (stationary/cyclic) patterns and LOCKS OUT
  the rest, which stay chaotic. ⇒ the stationary/cyclic/chaotic distribution EMERGES from
  decorrelation + limited slow subspace — NOT from an explicit classifier gate. Refines D-007:
  classifier is post-hoc labeling; it is not the thing forcing the distribution at this stage.
- **Open:** whitening (`Cov[s]=I`) vs off-diagonal-only decorrelation; hard constraint vs soft
  penalty; L1 vs L2 gap; pure single-pair online vs small pair-minibatches; soft-binary vs STE masks.

## D-012 — Full 3-module architecture + forecaster by gradient inversion

- **Date:** 2026-08-25
- **Decision (user):** The emulator has 3 modules:
  1. **Proper decompositor** = selection-mask encoder + whitening (`s = M·x`) [D-005/010/011].
  2. **Latent predictor** = evolve scalars forward in time (`s(t) → ŝ(t+1)`), per-channel 1D.
  3. **Forecaster** = inverse map: find state `x(t+1)` whose decomposition matches `ŝ(t+1)`.
     Mechanism (user): start from current `x(t)`, iteratively inject modifications into the
     cells where a mask = 1, repeat until `M·x → ŝ(t+1)`.
- **Agent analysis (the "blocking point"):**
  - The forecaster inverse is **under-determined**: `M` is `K×N` (16 × 8192) → 16 eqns,
    8192 unknowns. Infinitely many states give the same scalars.
  - User's "start-at-current + modify-where-mask=1" = gradient descent on `‖Mx−ŝ‖²` from `x(t)`;
    updates land on mask cells because grad = `Mᵀ(Mx−ŝ)`. Converges to the minimal-change
    pseudo-inverse: **`x(t+1) = x(t) + M⁺(ŝ(t+1) − M·x(t))`**.
  - ⇒ Forecaster moves the state ONLY within the K-dim mask subspace; the unobserved complement
    is **frozen at `x(t)`** and carried forward. Forecast expressiveness is capped at K.
  - Closed form exists → iterate only to add constraints (binary/positivity/physical bounds).
  - Implication: decomposition COVERAGE + K decide forecast quality; masks must capture the
    dynamically-active content (whatever they miss is assumed quasi-static).
- **Resolves earlier open items:** "spatial reconstruction (inverse)" = forecaster gradient
  inversion; "per-scalar predictor" = the latent predictor (module 2).
- **Open:** latent-predictor model (confirm independent per-scalar 1D; history window?); whether
  the frozen-complement behavior is intended; mask overlap handling; constraints on `x` in inversion; K.

## D-013 — Kernel semantics: masks are multi-scale regional "sensors", NOT orthogonal unmixers

- **Date:** 2026-08-26
- **Decision (user insight):** Reframes what a selection kernel IS and what "success" means.
  Corrects an earlier agent framing (kernel-as-ICA-unmixer, near-orthogonal, isolates one
  latent mode) — that framing is REJECTED.
  1. A mask is a **spatial sensor footprint**, not an unmixer. Masks need NOT be near-orthogonal;
     **overlap between masks is allowed/expected**. `s_i` being a mixture of modes is fine.
  2. Masks span **many scales / non-uniform ("not even") resolution**: from a small highly-dynamic
     region (real-life analog: Gulf Stream jet) to a basin-scale footprint (e.g. a long-range
     equatorial band). Temporal family (stationary/cyclic/chaotic) EMERGES from *where and at what
     scale* a mask looks — not from unmixing spatially-overlapping global modes.
  3. **"Deep" = many channels.** The point of the temporal latent emulator is to learn MANY (deep)
     scalar channels whose *collective* is a meaningful encoding. Quality is a property of the
     ENSEMBLE (rich enough to emulate/reconstruct), not of any single channel's purity.
- **Consequences:**
  - **Success metric** shifts from per-kernel isolation (`corr(s_i, a_k) > 0.8`) → **collective
    encoding quality**: reconstruction/predictive skill of the K channels + channel diversity/
    coverage (masks land on distinct, meaningful structures) + emergent spread of temporal
    behaviors across channels. Whitening's decorrelation is what drives the diversity (no
    duplicate footprints). [Supersedes the isolation-metric proposal; refines the still-open
    North-Star metric.]
  - **Generator (Milestone 0)** DROPS any "spatial separability" requirement. It should produce
    **spatially heterogeneous, multi-scale dynamics** — small energetic/fast regions + large-scale
    slow/coherent structures + background — with SSH/SST coupled. Latent modes may still exist as a
    HIDDEN answer key (unsupervised training, optional supervised post-hoc analysis of what each
    channel picked up); they are NOT fed to the model. [Refines D-009; consistent with D-011
    emergent-diversity mechanism.]
- **Open:** exact K ("deep" → larger K?); precise reconstruction/coverage metric formulation;
  multi-scale spatial basis for the generator (scale ladder + region placement).

## D-014 — Project file-management & tooling conventions

- **Date:** 2026-08-26
- **Decision (user):** binding project rules, cross-checked before building further.
  1. **Ephemeral artifacts → `./.tmps/`** (renamed from `./tmp/`). ALL script outputs, logs,
     cached tensors, scratch go here. Safe to delete. Overrides the framework's `./tmp/`.
  2. **`./results/`** holds meaningful/promoted results (metrics tables, plots, kept checkpoints).
     Written **ONLY with explicit user authorization** — default outputs stay in `./.tmps/`.
  3. **Hydra** owns every ML hyperparameter (`config/` tree; hydra-core 1.3.4 / omegaconf 2.3.1 in
     oceanai). No hard-coded hyperparameters; a run = its composed config (reproducibility).
  4. **TensorBoard** for online inspection: training/eval log to TB event files
     (`torch.utils.tensorboard.SummaryWriter`; tensorboard 2.20.0 in oceanai).
- **Reconciliations / notes:**
  - `.tmps/` and `.env` are NOT results; `results/` is the curated "Payload" (aligns with the
    framework's Final Destination).
  - Two-layer env: the BASE env may WRITE TB event files (writer already installed); the TOOLS
    env runs the TB server for VIEWING and any heavy viz. Writing ≠ contaminating base.
  - `GenConfig` dataclass in `src/data/synthetic.py` is to be migrated under Hydra `config/` when
    the training loop is built (kept as a plain dataclass only for the smoke test).
- **Open:** exact `config/` layout; the "meaningful result" authorization gate wording; TB log
  schema (scalars: loss components / whitening error / per-channel variance; images: masks,
  scalar series).

## D-015 — v0 encoder + training defaults (all Hydra-tunable)

- **Date:** 2026-08-26
- **Decision:** Build the settled v0 core (CLAUDE.md) with these defaults for the "minor knobs"
  CLAUDE.md explicitly allows defaulting at build. Nothing locked — all live in `./config/`.
  - Mask parameterization: **soft-binary `sigmoid(w/temp)`**, temp annealed `1.0 → 0.1` (geometric).
    Straight-through binary deferred.
  - Mask **domain: full-volume `(V,H,W)`** (resolves the D-008 lean).
  - `s_i` **normalization: divide by soft count** `Σ mask`.
  - **Whitening: SOFT penalty** `‖Cov[s]−I‖_F²` with weight `λ_white` (default 1.0). Hard
    whitening transform deferred.
  - **Slowness: L2** temporal gap, summed over channels, mean over pairs.
  - **Regime: Adam pair-minibatch SGD** — each step samples B random consecutive pairs (B=128);
    whitening covariance estimated over the batch. (CLAUDE.md "small pair-minibatches" knob.)
  - Logging: **TensorBoard** (D-014); config via **Hydra** (D-014).
- **Rationale:** soft penalty + minibatch is streaming-friendly, differentiable, GPU-cheap on the
  tiny 2000-step field; whitening blocks M=0 collapse & duplicate kernels; decorrelation drives
  the D-013 channel diversity. Defaults are the reversible/low-risk choices; hard whitening / STE
  are the upgrade paths if v0 underperforms.
- **Open:** λ_white tuning; anneal schedule; whether to add running (vs batch) covariance;
  per-scalar predictor (module 2) still unbuilt; concrete collective metric.

## D-016 — Mask init is multi-scale random; the temperature anneal is OFF by default

- **Date:** 2026-08-26
- **Decision (agent, empirical — reversible Hydra knobs):**
  - `model.init: multiscale` — each channel's logits are a smooth random field with its OWN
    length scale (geometric ladder `sigma_min=1 → sigma_max=16` cells), `init_std=1.5`.
  - `model.temp1: 1.0` — i.e. NO temperature anneal for now (`temp0 = temp1 = 1.0`).
- **Rationale (measured):**
  - The old `0.01*randn` init makes every mask ≈ 0.5 everywhere, so every channel reads the same
    spatial mean: the channel correlation matrix is **rank-1 from step 0** (`cond = K/eps = 16000`)
    and whitening is degenerate before training starts. Multi-scale init breaks this AND is the
    concrete form of D-004's "start from random masks" + D-013's multi-scale-sensor prior.
  - Annealing `temp 1.0 → 0.1` saturates the sigmoids and freezes the masks before they organize
    (SOP 02 already listed this failure mode): decorrelation stalls at off-diag corr² **0.53**
    vs **0.27** un-annealed, with grad norm decaying to 1e-6.
- **Status:** the anneal (and STE) remain the upgrade path to genuinely binary masks — but they
  must be applied AFTER the masks have organized, not during. Not yet built.

## D-017 — Whitening must be DIFFERENTIABLE and scale-invariant in the gradient

- **Date:** 2026-08-26
- **Decision (agent, empirical):** `train.whitening: corr` is the default. The loss is
  `L = mean_i var(Δs_i)/var(s_i)  +  λ_white · mean_{i≠j} corr_ij²  +  λ_var · hinge`,
  i.e. SFA's Rayleigh quotient plus an explicit off-diagonal decorrelation penalty, with every
  normalization term differentiable (no `.detach()`, no matrix inverse).
- **Rationale:** the two previously-specified variants BOTH collapse, in opposite directions
  (findings F-2). `soft` (penalty on raw `Cov`) → masks shrink to 0. `hard` (detached ZCA) → all
  K channels merge into ONE (effective rank 1.00/16), because a detached whitening matrix imposes
  no cost on redundancy. A third attempt with a detached per-channel std collapsed too: **scale
  invariance must hold in the gradient, not merely in value.**
- **Also added:** `lambda_var` anti-death hinge `relu(1 − std(s_i/√count_i))²` (inactive for live
  channels) and `grad_clip`, since nothing else forbids a literally-empty mask.
- **Kept as ablations:** `whitening: hard` and `whitening: soft` still run, for the record.
- **Status:** this is the first formulation that trains stably and produces real structure (F-3).

## D-018 — OPEN / USER DECISION: non-negative masks + mean-slowness cannot reach 1/3/6

- **Date:** 2026-08-26
- **UPDATE 2026-08-26 (after visualizing — finding F-6):** the root cause of both facts below is
  now identified and it is neither of the two originally listed. The learned kernels **migrate to
  the lowest-energy cells of the domain** (the edges), because `var(Δs)/var(s)` is a ratio and is
  therefore indifferent to amplitude: a tiny, almost purely slow tail in a dead corner beats real
  dynamics. Duplication, chaotic-mode loss, and sub-random reconstruction all follow from that.
  ⇒ the leading fix is an **amplitude / coverage term in the objective** (option (c) below), not a
  change to the mask's sign semantics. Also note the non-negativity limit (fact 1) is largely a
  GENERATOR artifact — all `φ_k` are positive bumps, so the field has non-negative spatial
  covariance by construction, unlike real ocean anomaly fields (findings F-4 caveat).
- **Two measured facts that the v0 design as specified cannot reconcile (findings F-4, F-5):**
  1. **Non-negativity caps diversity.** With `sigmoid` selection masks, decorrelation saturates at
     off-diag corr² 0.244 / effective rank 4.08 of 16, *independent of λ_white over a 500×
     range*. A non-negative sensor cannot cancel a shared positive large-scale mode. Signed
     (`tanh`) masks reach 0.070 / 8.57 — so non-negativity, not tuning, is the binding constraint.
     ⇒ **D-011's `Cov[s] = I` is unreachable in principle under D-005's selection semantics.**
  2. **Slowness minimization deletes the chaotic modes.** Untrained multi-scale masks encode all
     10 hidden modes at ensemble R² ≈ 1.0; after training the 5 chaotic modes fall to R² 0.01–0.03
     and the population is 0/14/2 against the 1/3/6 target. D-011's "emergent governor" does not
     fire.
- **Options (need the user's call — these change the design, not a hyperparameter):**
  - **(a) Allow a sign in the selection.** Keep binary semantics but over {−1, 0, +1} ("select
    this cell, positively or negatively") instead of {0, 1}. Measured to roughly double channel
    diversity. Revises D-005.
  - **(b) Stop minimizing MEAN slowness; impose the population directly.** e.g. SFA-style ordering
    (channel-dependent weights: only the first few channels are pushed slow, the rest are left
    free or pushed fast), or a distribution loss on the classifier's family histogram — which is
    what D-004/D-007 originally described and D-011 dropped in favour of emergence.
  - **(c) Add a coverage/reconstruction term** so channels must span the field, not just be slow.
    Directly serves the North Star; currently training makes reconstruction slightly WORSE than
    random masks.
- **Status:** BLOCKING for the next design step. Nothing here is pre-decided.

## D-019 — Energy floor + coverage term (fixes the dead-corner and lost-chaos failures)

- **Date:** 2026-08-27
- **Problem:** `L_slow = mean_i var(Δs_i)/var(s_i)` is a ratio, hence blind to amplitude, so it
  walked every kernel into the domain's dead edges (F-6) and discarded all fast content (F-5).
- **Decision (agent, empirical; all Hydra knobs):** add two terms to the `corr` objective.
  1. **Energy floor** `L_energy = mean_i relu(1 − e_i/e_ref)²` with `e_i = var(s_i)/count_i²` and
     `e_ref` = the field's mean per-cell temporal variance. `count²` normalization makes `e_i`
     independent of mask SIZE, so it measures "are the selected cells energetic AND mutually
     coherent" and cannot be gamed by growing/shrinking the footprint. A one-sided floor, so it
     stops fighting once a kernel is on signal. `lambda_energy = 0.2`.
  2. **Coverage / reconstruction** `L_recon = 1 − R²` of the best per-batch linear decode of the
     field from `[1, S]`, solved in closed form and left DIFFERENTIABLE. Stronger than the energy
     floor because it also penalizes REDUNDANCY: a duplicate channel buys no reduction, so its
     gradient points at the unexplained residual. `lambda_recon = 0.3`.
- **Two implementation traps, both hit:**
  - The decode MUST use standardized channels. With raw `s` (variance ~1e6) the normal-equation
    matrix has entries ~1e9, the ridge is negligible against it, and with channels still correlated
    at ~0.99 the solve is near-singular — its garbage gradient swamped every other term at ANY
    lambda (λ_recon = 0.003 already froze training at effective rank 1.14).
  - **Set the weights by matching GRADIENT norms, not loss values.** Measured at init:
    `|g_slow| = 6.7e-6`, `|g_white| = 1.7e-4`, `|g_energy| = 8.8e-4`, `|g_recon| = 5.0e-4` — while
    the loss VALUES differ by 300× in the opposite direction. The defaults above are
    gradient-matched to `lambda_white`.
- **Measured effect (5000 steps, K=16; findings F-7):** kernels leave the dead corners
  (`min e/e_ref` 0.05 → 0.82); every hidden mode is encoded again (chaotic-mode ensemble R²
  0.01–0.03 → 0.79–1.00); balanced reconstruction no longer degrades (0.861 → 0.880 vs 0.881
  untrained); effective rank 3.77 → 3.89; mask overlap IoU 0.170 → 0.111; the 31× slowness spread
  is preserved.
- **Also learned:** `lambda_slow = 10` makes everything WORSE (effective rank 2.95, spread 2.3×) —
  the slowness term is the one that destroys diversity, so it should stay weak.
- **Still open:** duplication (effective rank ~4 of 16) is unchanged — that is the separate F-4
  non-negativity ceiling, not something an energy/coverage term can fix.

## D-020 — Per-channel SCALE LADDER keeps a diversity of mask types

- **Date:** 2026-08-27
- **Observation (user):** "the mask is shrinking (they see only small region and are getting
  smaller) ... we should diverse the type of the masks." Confirmed by measurement — trained mask
  footprints, from a ~4095-cell (50% of domain) init:

  | run | min / median / max cells (of 8192) |
  |---|---|
  | slow + decorr only | 5 / 104 / 404 |
  | + energy | 6 / 54 / 218 |
  | + energy + recon | 7 / **28** / 186 |

- **Cause:** two terms both reward shrinking. Decorrelation prefers disjoint footprints, and
  `e_i = var(s_i)/count_i^2` IS the mean pairwise covariance of the selected cells, which is
  maximized by a tiny tightly-coherent patch. So D-019's energy floor made the shrinkage WORSE
  (median 104 -> 28) — it fixed WHERE kernels look but pushed them all to the smallest scale, and
  the D-013 "multi-scale sensor" picture decayed into 16 small patches.
- **Decision:** add a per-channel footprint ladder. Channel `i` gets a target count on a geometric
  ladder from `size_max_frac` (0.30 of the volume, basin-scale) down to `size_min_frac` (0.002, a
  small energetic patch), penalized only outside a tolerance factor `size_tol=2.0`:
  `L_size = mean_i relu(|log(count_i/tgt_i)| - log(size_tol))^2`, `lambda_size = 0.3`.
  A guide, not a pin — it releases inside the tolerance band.
- **Measured (5000 steps, K=16):** footprints now span **8 .. 1143 cells (0.1% .. 14% of domain)**
  in a clean monotone ladder, and **mask overlap IoU halves, 0.111 -> 0.061**, while effective rank
  (3.57 vs 3.89), balanced R² (0.880) and the slowness spread (26.5x) are all unchanged. Scale
  diversity is therefore nearly FREE — it costs none of the other objectives.
- **Note on weighting:** `|g_size|` is 35x `|g_white|` at init, so gradient-matching would give
  `lambda_size = 0.03`. 0.3 is deliberately stronger: the term is a hinge that vanishes once the
  ladder is met (final `L_size` = 0.003), so it pulls the masks onto the ladder early and then
  stops mattering. 0.03 / 0.1 / 0.3 all give equivalent metrics.
- **Still open:** the ladder diversifies SCALE but not LOCATION — the figures show nested
  concentric footprints on the same few energetic structures, which is why effective rank stays
  ~3.6 of 16. A location-spreading mechanism (or the F-4 generator fix) is the next lever.

## D-021 — Generator rev2: fixes the three defects the field figures exposed

- **Date:** 2026-08-27
- **Context:** user asked for a fix proposal for the three F-8 defects. All knobs default to the
  OLD behaviour in `GenConfig`; the new values live in `config/config.yaml`, so every run recorded
  before today still regenerates its exact original field (`evaluate.py` relies on this).
- **Fix 1 — dead borders** (`place_full_domain`, `periodic_x`): centres over the full grid and a
  wrapping x axis (longitude), so no edge-clipped margin. Dead cells **26.3% → 3.4%**.
- **Fix 2 — SSH ≈ SST** (`sst_tau`, `n_private_ssh/sst`): SST becomes a first-order AR1 response to
  the SSH forcing, `rho = exp(-1/tau)`, plus 2+2 modes visible in only one variable. A fixed
  3-step shift was meaningless against periods of 60–300; an AR1 with tau=20 damps amplitude AND
  shifts phase (~64° at period 60). `corr(SSH,SST)` domain-mean **0.981 → 0.400**.
- **Fix 3 — all-positive patterns** (`signed_patterns`): zero-mean band (positive core, negative
  surround), **dipoles** (`bump(c1) − bump(c2)`) and **low-wavenumber wave packets** for the
  medium scale. Small chaotic eddies stay monopoles — physically right, and the sign structure
  belongs to the large/medium scales. Negative cell-pair correlations **0.0% → 49.0%**.
- **Iteration that the FIGURES forced (worth remembering):** the first version used
  domain-filling waves. Every metric improved (eff rank 9.56) but `field_snapshots.png` showed the
  field had become a global interference plaid — an un-enveloped wave carries |phi|=1 over all 4096
  cells and swamps every localized structure, and even the variance map came out striped. Fixed
  with `wave_env_sigma=15` (regional wave packets) + `wave_kmax=2`. The metrics alone would have
  called that a success.
- **THE PAYOFF — the F-4/D-018 ceiling is GONE:**

  | metric | legacy field | rev2 field |
  |---|---|---|
  | off-diag corr² | 0.259 (saturated, λ-independent) | **0.063** |
  | effective rank | 3.89 / 16 | **9.23 / 16** |
  | mask overlap IoU | 0.111 | **0.024** |
  | balanced recon R² | 0.880 | **0.989** |
  | slowness spread | 30.7× | **170×** |
  | hidden modes, ensemble R² | chaotic 0.79–1.00 | **all ten 0.80–1.00** |

- **⇒ D-005 is VINDICATED and D-018 is RESOLVED without changing it.** Non-negative selection
  masks were never the problem — rev2 (9.23) beats even the signed-`tanh` ablation on the old field
  (8.57). The binding constraint was the TESTBED, not the mask semantics. No signed masks needed.
- **New tension to watch:** `L_energy` stays unsatisfied (0.34, min e/e_ref 0.02) because the size
  ladder demands a footprint of 30% of the domain while the field now has sign structure inside any
  such footprint, so a large mask partly cancels itself. The energy floor and the scale ladder pull
  against each other at the largest scale. Options if it matters: exempt the top ladder rungs, or
  measure `e_i` against what a random mask of the SAME size would score.


## D-024 — Spectral band ladder: the TIMESCALE is assigned per channel, not emergent

**Date:** 2026-08-31. **Status:** BUILT and measured (branch `worktree-stationary-observer`).
**Source:** agent, derived from measurement (not a user ruling — contrast D-025/D-026).

**Context.** CLAUDE.md's v0 core says the stationary/cyclic/chaotic spread across kernel index
should EMERGE from slowness + decorrelation, with the classifier only labelling it (D-011). Two
measurements say that mechanism does not deliver: the family population is slow-dominated at every
K from 4 to 64 (F-12), and the emergent spread was never observable anyway because the labeller could not
emit "stationary" (F-11).

**Decision.** Keep everything in D-015/D-019/D-020 and ADD a per-channel timescale assignment, the
temporal twin of the D-020 size ladder. Channels get roles in the `pop_target` proportion
(1/3/6 scaled to K -> 2 slow / 5 cyclic / 9 fast at K=16), and three hinge terms:

- `L_band` — in-band power fraction, per-channel assigned rFFT band;
- `L_line` — line-vs-hump shape: cyclic rungs must BE a spectral line, slow and fast rungs must NOT;
- `L_mem`  — slow rungs only: lag-`mem_lag` autocorrelation >= `mem_target`. *(Superseded as the
  default by D-025's `L_level`; retained as `slow_objective: memory`.)*

All are ratios of powers (scale-free in value AND gradient, the D-017 rule) and all are one-sided
hinges (silent once satisfied, so they guide rather than pin — D-020's philosophy). Weights are
GRADIENT-matched to `lambda_white` at init, per the D-019 convention, never value-matched:
`lambda_band=5.0`, `lambda_line=4.0`, `lambda_mem=0.4` (measured |g_white|=3.9e-3, |g_band|=7.8e-4,
|g_line|=1.0e-3, |g_mem|=9.5e-3).

**Why all three.** The families overlap in FREQUENCY on this testbed (cycles 60/140/300, OU
tau=200), so a band alone cannot separate a slow drift from a slow cycle, and it left the slow rungs
on the wrong content entirely (corr 0.50 to a chaotic mode). `L_mem` is the sharp stationary
selector — at lag 64, OU(tau=200) gives rho=+0.73 while cycles of period 300/140 give +0.23/-0.96
— and `line_cap` covers its one blind spot (a cycle whose period divides the lag aliases to rho~1).
Because of that frequency overlap, the cyclic band range is deliberately allowed to OVERLAP the slow band
(`cyclic_period_max=320` > `slow_period_min=128`); masks were never required to be disjoint (D-013).

**Why windows, not pairs.** A lag-1 pair minibatch says nothing about a period-300 cycle. The
periodogram needs contiguous time, so the ladder slices `n_win` random windows out of `enc(field)`
(one einsum over T×8192 cells — cheap enough to redo every step). The stochasticity D-011 asks for
now comes from the window starts rather than the pair index.

**Rejected alternative — do not re-try:** matching a per-channel target autocorrelation
`exp(-L/tau_i)`. Wrong for a cyclic channel, whose ACF oscillates and whose long-lag gap reaches 4 vs
the exponential max of 2 (twice it) — it would actively suppress the cyclic family it was meant to
create.

**Consequence for the framework:** the family spread is now DESIGNED (a rung layout), not emergent.
This narrows D-011's "emergent diversity" claim: decorrelation diversifies WHERE channels look, but
not WHICH dynamics they hold. The labeller stays post-hoc and reversible (D-004) — the ladder shapes
where a channel looks in the spectrum, it never asserts a family; `role_obedience` in the probe is
the honest check on whether the assignment took.

**Does not fix:** the cyclic rungs (1 of 5 reach a clean line) or the largest slow rung. See F-12.


## D-025 — STATIONARY = CONSTANT (supersedes my "slow" reading in D-024/F-11)

**Date:** 2026-09-01. **Status:** BUILT and measured. **Source:** direct user ruling.

**The ruling.** A stationary signal is a CONSTANT one — flat in time. The generator's mode 0 (an OU
drift at tau=200, standardized to unit variance) was never stationary, and a slow red DRIFT belongs
to the `cyclic` family, not the stationary one, because it is not constant. This supersedes the
definition I used in D-024 and F-11, where "stationary" meant long-memory red noise; every
population figure reported under that reading is a slow/cyclic/fast count, not a family count.

**What changed, in three places (all measured — see F-13):**

1. **Generator rev3** (`data.stationary_constant: true`): mode 0 is a static pattern with a flat
   amplitude, un-standardized, living in the field's time mean. Verified std 0.000 (`0.000e+00`) and
   time-mean map correlating 1.000 with the pattern. Legacy default (`GenConfig`) is False so old
   runs still reproduce.
   - *Detail that matters:* a constant must BYPASS `_ar1_response` for SST. An AR1's steady state
     under constant forcing IS that constant, so routing it through would add a startup transient —
     and that function's closing `_standardize` divides a zero-variance series by ~0, which would
     have deleted the mode from SST entirely.
   - *Checked, not assumed:* field standardization `(fv - fv.mean())/fv.std()` removes a GLOBAL
     SCALAR mean, not a per-cell one, so a spatially structured static pattern survives.
2. **Labeller** (`src/probes/family.py`): flatness is tested FIRST, via `amp_ratio = std/|mean|`
   measured on the RAW series — flatness is undecidable from a standardized one, so the test is
   skipped rather than guessed when the caller cannot supply it (`amp_ratio=None`). Then cyclic (a spectral line OR
   long memory), then chaotic. 10/10 on the hidden truth, seeds 0-4.
3. **Encoder** (`L_level`, function `level_term` in `src/train/spectral.py`): the stationary rungs are asked for
   `level/(level+fluct) >= flat_target` and are EXEMPTED from `l_var`, `l_energy`, `L_band` and
   `L_line`, every one of which presumes a channel fluctuates. `flat_target=0.95` is set to match
   the labeller's cut exactly (r = 1/(1+amp_ratio), amp_ratio < 0.05 -> r > 0.952); a looser target
   would let the hinge go quiet before the channel qualified — a first attempt at 0.8 corresponded
   to `amp_ratio` 0.25. **The objective and the readout have to agree on where the bar is.**
   `lambda_level=0.13`, gradient-matched. The previous `slow_objective: memory` behaviour is kept as
   an ablation. *(`flat_target` later raised 0.95 → 0.98 by the user, 2026-09-27 — see D-027.)*

**Why the encoder could not have found a constant before.** A constant lives ENTIRELY in the channel
mean, and every v0 loss term uses the CENTERED channel `sc = s_t - mu`: slowness, whitening,
coverage and the energy floor are all anomaly-only. Two terms went further and actively forbade a
flat channel — `l_var = relu(1 - std)²` demands unit normalized std, and `l_energy` demands the
selected cells be energetic in TIME. A stationary channel was not merely unrewarded, it was
penalized. `L_level` also replaces the on-signal protection `l_energy` was providing (F-6).

**Consequence for the architecture.** The observer now has to report TWO different kinds of thing:
a level (for the static family) and an anomaly (for everything else). D-005's selection semantics
are unchanged — `s_i = <mask_i, field>` already carries both — but the LOSS had to stop being
anomaly-only. This is the minimal version of that change: per-rung exemptions rather than a full
mean/anomaly split of `s_i`, which remains available if modules 2/3 need the level separately.

**Open:** the flat channels are flat but spatially diffuse (|align| 0.27 with the injected pattern).
De-aligning the size ladder was tested and does NOT fix it (F-13).


## D-026 — Classify by global structure; families belong to the OBSERVABLES

**Date:** 2026-09-01. **Status:** BUILT (labeller + probe; no retraining needed). **Source:** user.

**Two rulings.**

1. **Classification follows a channel's global structure, not its small fluctuation and drift.**
   A channel that oscillates under fast noise on a wandering baseline is CYCLIC. Implemented as a
   4-part decomposition (`level / trend / osc / residual`, see `family.py:decompose`; `trend` = bins
   1–3, `osc` = the dominant peak's half-power band), label = flat gate, then `trend + osc` vs
   `residual`. Drift counts as structure (it is not broadband), per
   D-025's ruling that a drift is cyclic.
2. **The observer is not a mode-recovery device.** Hidden modes are just hidden modes, and in a real
   ocean/atmosphere capturing them all is barely possible. What the kernel learns is WHERE TO LOOK,
   and a family is a property of the OBSERVABLE it constructs. A stationary channel may be a single
   static mode, a persistent phenomenon, or a combination of varying signals that barely moves.

**Consequences for how we score (this is the part that changes conclusions, not just code):**

- `population` is measured against a DESIGN target (the rung ladder's 1/3/6 proportion = the mix of
  observables we ask for), NOT against the hidden mode counts. `target_is_design_not_recovery: true`
  in eval.json says so explicitly.
- `mode_recovery` (max |corr|, ensemble R^2) and `mask_align_with_pattern` are DIAGNOSTICS. Judging
  a flat channel by its alignment with the injected pattern was the wrong frame; it is reported, not
  optimized.
- The success criterion for a stationary channel is FLATNESS ALONE (`amp_ratio < 0.05`). A
  persistence check across sub-windows was considered and the user chose against it: flatness over
  the record is enough.
- Cancellation is ALLOWED but NOT ENCOURAGED — no term rewards it. `L_level` rewards flatness
  however the kernel achieves it. A new diagnostic (`cancellation_vs_independent`) reports which
  mechanism a given flat channel actually used.

**Design note — the level GATES, it does not join the argmax.** A plain argmax over all four shares
is tidier but wrong: `level² > var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its
level would score "stationary". Counter-example on the record: ch14 of run `20260901_080655`,
`amp_ratio` 0.526, plainly broadband chaotic.

**What this does NOT change:** D-005 selection semantics, D-013 (masks are sensors, overlap fine),
or the ladder mechanics in D-024/D-025. It changes the labeller and what counts as success.


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

## Pending decisions (awaiting Phase B input — do not pre-decide)

- North Star / target metric — awaiting user (Discovery Q1).
- Data & Signals (input signal / target output; sampling rate, channels, dtype) — awaiting user (Q2).
- Source of Truth (where the dataset lives; size/format) — awaiting user (Q3).
- Deliverable (checkpoint / metrics / plots / inference script) — awaiting user (Q4).
- Constraints & Rules (compute budget, reproducibility, coding/data rules) — awaiting user (Q5).
- Input + output tensor schema (Data-First) — to be defined after Discovery, then recorded
  in CLAUDE.md per Phase B.2.

## Module 2 (latent predictor) decisions — D-022, D-023, D-024 (module-2), D-028

> **Merged 2026-09-28 from `architecture/03a_module2_decisions.md`** (latent-predictor worktree; that file
> was a branch record written because `memory/` was gitignored and absent from the branch — "on merge,
> reconcile with `memory/decisions.md` rather than replacing it"). The entries below keep this file's
> text as base and fold in every fact of 03a; where 03a was the newer amended text (D-024, D-028) the
> entry is organised around it. `architecture/` is retired; this file is now the only home.
> Companion records: `memory/sop/03_latent_predictor.md` (the SOP), `memory/findings.md` module-2
> entries F-9, F-10, F-10b, F-11/F-12 (module-2 track), F-18 (the evidence).

## D-022 — Latent predictor (module 2): per-channel 1D system WITH hidden state, trained free-running

- **Date:** 2026-08-29
- **Context:** user asked to build module 2 of D-012 — predict the next feature in the compressed
  domain. Encoder (module 1) is **frozen**; nothing spatial enters this module, which only ever sees
  `S[T,K]`.
- **Decision (settled, all Hydra-tunable in `config/predictor.yaml`):**
  1. **State is mandatory.** A memoryless map `s(t) → s(t+1)` is a *function*, so it must return the
     same future every time a channel revisits a value — but `s_i` is a 1D projection of a
     higher-dimensional field and revisits each value at many phases. A circle needs 2D state; chaos
     needs a Takens embedding. Default `arch=gru` (learned recurrent state); `arch=mlp` (explicit
     16-step delay window) and `arch=linear` (learned AR) are ABLATIONS. No memoryless variant is
     offered, deliberately. *(Default history: gru at build → mlp by D-023 → gru again by D-028.
     03a's copy of this item drops the default sentence, consistent with the default having moved.)*
  2. **Per-channel parameters** (`per_channel: true`), batched `[K,d_in,d_out]` einsum so all K
     independent 1D systems still train in one pass. Honours D-012's "per-channel 1D" and keeps the
     family labels meaningful. `per_channel: false` = shared-dynamics ablation.
  3. **Residual parameterization** `z(t+1) = z(t) + dz` — persistence is the zero-output solution.
     (Ignored by `ar_mlp`, whose AR anchor already contains persistence.)
  4. **Free-running rollout in the loss, with a horizon curriculum** 1 → `horizon_max` over the
     first 30% of training, plus a teacher-forced 1-step term. MEASURED as load-bearing: see D-022
     results / F-9.
  5. **Time-ordered split with TRAIN-ONLY standardization** and a gap of `split_gap` steps. Channel
     scale spans 9.6 … 1250, so standardization is not optional.
  6. **Baselines are non-negotiable and computed in the same script:** persistence and closed-form
     least-squares AR(p). Headline metric is skill vs persistence, never raw R².
  7. **`var_ratio` = var(forecast)/var(truth) at long lead is a FIRST-CLASS metric, ranked BEFORE
     MSE skill.** This is the change of practice this session forced (see F-9).
- **Artifacts:** `predictor.pt`, `predictor_metrics.json`, `predictor_arrays.npz`,
  `predictor_eval.json`, `figs_predictor/*.png`, TB tags under `pred/` — all inside the module-1
  run dir, one dir per experiment (D-020 convention).
- **Open:** a real classifier (now blocks module 2's per-family interpretation too); whether to
  train the encoder jointly with the predictor instead of freezing it; module 3. *(Note: the
  2026-09-10 directive in D-024 puts modifying the encoder out of scope.)*

## D-023 — Module 2 default is the explicit DELAY WINDOW (`arch: mlp`), not the learned GRU state  [SUPERSEDED by D-028]

- **Date:** 2026-08-29
- **Decision (USER, made from the figures per the visualization rule):** module 2's default
  architecture is the 16-step delay-window MLP. Supersedes the `arch: gru` default that D-022 was
  first built with.
- **Reasoning (user's, from `pred_rollout_stats.png` + the comparison table in F-9):**
  - the two are TIED where it is measured most reliably — h1 +0.398 vs +0.420, h4 +0.561 vs +0.564,
    i.e. within 0.02 — while the MLP uses **3x fewer parameters** (4 624 vs 14 864);
  - an explicit Takens **embedding dimension is inspectable** (`window: 16` in the config), where a
    learned recurrent state is whatever it happened to become;
  - the MLP's apparent long-lead advantage (h64 +0.393 vs +0.246) is discounted, not credited — it
    comes partly from the SAME mean-collapse that makes the linear ablation look best (amplitude
    0.65 vs the GRU's 0.89). The user read the tie as the real signal and the h64 gap as an artifact.
- **Retained, not discarded:** `arch: gru` stays a first-class option. Its rollout amplitude is
  better (0.89 vs 0.65) and it wins ~5x on the chaotic-labelled channels, so it is the choice when
  the priority is handing module 3 a forecast that has not shrunk toward the mean.
- **Open / revisit trigger:** every per-family number behind this rests on the heuristic labeller
  (population 0/14/2 vs the 1/3/6 target). Re-read these figures once a real classifier exists.
- **Process note:** this decision is also the first application of the user's **visualization rule**
  (2026-08-29) — a figure exists to make a decision WITH the user, not for the agent to settle alone
  and report. The agent had originally set `gru` unilaterally and took the fork back.
- **Superseded 2026-09-27 by D-028** (default back to `gru`; the short-lead tie does not exist on the
  current encoder).

## D-024 — Predictor architecture: settled at 1 step, NOT at rollout  [AMENDED 2026-09-10]

*(Module-2 D-024. Title in 03a: "Architecture is closed AT 1 STEP only; nonlinearity is required for
the free run". Not the encoder's D-024 "Spectral band ladder" above — see "Numbering collisions".)*

> **AMENDED 2026-09-10.** As first recorded this said "stop searching predictor architectures; the
> lever is module 1" — full stop. That was **wrong on both halves**: the evidence behind it was
> computed at **h1 only**, and it does not hold at rollout; and module 1 is out of scope (below).
> Corrected statement and evidence follow; see F-12 (module-2 track).

- **Date:** 2026-08-31, amended 2026-09-10.
- **Corrected decision:**
  1. **At h1, architecture is settled.** Every form is interchangeable; six forms land within 0.045
     and win/lose on the same channels (r = +0.96); per-channel oracle selection buys +0.009.
     No further 1-step search is warranted.
  2. **At rollout, architecture matters.** A nonlinear predictor is *required for the free run*
     (F-12a), and memory length is a live knob.
  3. **Delay-window length is a live knob** and was never swept: window 16→128 lifts rollout
     amplitude 0.47→0.85 and costs 1-step skill +0.413→+0.289, monotonically (F-12b). Neither end is
     right; the operating point is OPEN and depends on module 3's needs.
  4. **Still standing regardless of horizon:** do NOT build the mixture-of-experts gate. F-11
     (module-2 track) is a per-channel result that does not depend on this correction.
- **Evidence AT h1** (F-10b, F-11 — six forms, same frozen encoder and split):
  - every usable form lands within **0.045**;
  - they win and lose on the **same channels**, `corr(AR, GRU)` across channels = **+0.964**;
  - per-channel oracle selection over four forms buys **+0.009**;
  - the family-specialized experts (`ou`, `osc`) win **0 of 16 channels**, because AR(1) and polar
    AR(2) are linear functions of the delay window and hence strict special cases of the MLP that
    already contains them.
- **Evidence AT ROLLOUT — this is what the original decision missed** (F-12):
  - the cross-channel correlation that justified "interchangeable" **falls apart**: mlp-vs-linear
    +0.972 at h1 → **+0.572** at h64; gru-vs-linear +0.960 → **+0.454**;
  - channels with honest amplitude (var ratio 0.5–1.5) at h64: mlp **9/16**, gru **8/16**,
    linear AR(16) **0/16**. The linear model cannot carry a single channel through 64 free-running
    steps with realistic variance; its headline h64 skill is entirely mean-collapse.
  - ⇒ **nonlinearity is not optional for the free run**, which is the regime module 3 consumes.
- **What this rejects:** the three-module / per-family predictor proposal, in both its hard-routing
  and soft-gated forms. The hard form additionally conflicts with D-013 (masks are regional sensors,
  so channels are MIXTURES, not pure family members) and with the broken labeller (0/14/2).
  The soft-gated form was attractive — the gates would double as an unsupervised classifier and
  yield the 1/3/6 population histogram — but its premise is measured false, so it would be
  machinery built on nothing. **Revisit only if module 1's channels change materially.**
- **What it keeps:** `ar_mlp` stays in the codebase for its *floor*, not its score — it provably
  cannot start below the closed-form baseline and the trainer asserts that every run. That
  assertion is what caught F-10.
- **Remaining levers, in order:**
  1. **Rollout realism inside module 2** — choose the delay-window operating point (the
     accuracy-vs-amplitude trade in item 3 above).
  2. **The classifier**, still unbuilt, still blocking the interpretation of module 2's per-family
     results as well as module 1's.
  3. **Module 3**, whose interface is ready.
- **Encoder is OUT OF SCOPE (user directive, 2026-09-10): do not modify the encoder.** The original
  entry proposed changing module 1's objective (adding the never-implemented D-004/D-007
  family-distribution term) on the grounds that its slowness objective is what makes the channels
  near-linear. That proposal is **withdrawn** — module 1, the generator and their probes are not to
  be touched. The observation about slowness (the `L_slow`-uniform-pressure observation) stands as an
  explanation of why the channels look as they do; it is not a work item.

## D-028 — Module 2 default is the GRU (`arch: gru`); supersedes D-023

- **Date:** 2026-09-27
- **Decision (USER, made from the F-18 figures per the visualization rule).** The default predictor
  is the per-channel GRU. **Supersedes D-023** (default `mlp`).
- **Numbering note:** D-025 … D-027 were taken by the parallel encoder track; D-028 is the next free
  number. See "Numbering collisions with the encoder track" below.
- **Evidence** (F-18, encoder `flat98fix_seed0`, same predictor config as all earlier runs):

  | | h1 | h4 | h64 | amplitude @h64 | honest ch. | acc horizon | params/ch |
  |---|---|---|---|---|---|---|---|
  | **GRU** | **+0.713** | **+0.746** | +0.264 | **0.80** | **15/16** | **40.0** | 929 |
  | MLP | +0.667 | +0.692 | +0.241 | 0.57 | 9/16 | 35.3 | 289 |

- **Why D-023 fell:** D-023 chose the MLP because it *tied* the GRU at short lead (h1 +0.398 vs
  +0.420 on the old Aug-27 encoder) for 3x fewer parameters. On the current channels there is no
  tie — the GRU leads on every axis, wins all 9 chaotic channels at h1, and is the first trained
  model to beat the closed-form AR baseline at 1 step since the F-10 fix. For module 3, amplitude
  0.80 vs 0.57 means a far less damped field to invert.
- **Retained:** `arch: mlp` stays available — smaller, and its embedding dimension is readable off
  `window`. Its 16-step window was tuned on the old encoder and has not been re-swept on current
  channels (F-12b), so the MLP's gap may be partly a window effect.

## D-029 — Observer assumption for the module-2 memory theory: STABLE (assumed) and SUFFICIENT (required)

**Date:** 2026-09-28. **Source:** user ruling ("I don't think that W1 is a problem ... I suppose that a
dynamical system has a stable observer" / "I agree with sufficient observer"), in the discussion recorded
in `memory/sessions/2026-09-28_predictor-memory-sight-theory.md`.

- **A1 — stable observer (ASSUMED).** The observer (encoder) is trained once on the training set and then
  frozen; a frozen observer maps the same input to the same trajectory, so the trajectory memory is
  well defined. The memory is defined RELATIVE TO ONE FROZEN OBSERVER VERSION (e.g. `flat98fix_seed0`).
  Retraining with a changed objective is a design change (new observer, new memory), not a flaw of the
  theory — this retires the "retrained encoder changes everything" half of W1.
- **W1′ — sufficient observer (REQUIRED, open).** Stability is not enough: the observed trajectory must
  determine the future, i.e. every dynamic relevant to the forecast must be visible in some channel
  (the control-theory analogue: a stable observer exists only if the system is observable).
  W1′ absorbs W2 (per-channel vs joint input): both ask whether the observed trajectory is sufficient.
- **Evidence that W1′ is not automatic** (`memory/measurements.md`, runs `fixavg_seed{0..4}`, same data
  and objective): seeds 2 and 3 carry no period-60 cyclic channel (periods 286/143/143/143/286 and
  286/286/143/143/143) while seeds 0, 1, 4 do. Each is stable once frozen; two are blind to one cycle.
  **⚠ CORRECTED by F-19 (same day): this inference was WRONG.** Dominant FFT period ≠ visibility; seeds 2 and 3
  forecast the period-60 mode at corr +0.99 from a single channel. W1′ remains the right REQUIREMENT, but no
  seed tested so far violates it for the cyclic family.
- **Test (run 2026-09-28):** zero-parameter analog forecaster on each `flat98fix_seed{0..4}` encoder
  output. A1 + sufficiency predict ~seed-invariant skill; insufficiency predicts skill loss on the
  channels/cycles a seed does not see. Result: F-19 — sufficiency holds for all cyclic modes on all 5 seeds.

## D-030 — Module 2 is a STREAMING emulator: frozen weights, append-only memory; streaming evaluation protocol

**Date:** 2026-09-29. **Source:** user ruling ("we have to get out from traditional ML training-inference
scheme ... from very next day we can use it by appending state" / "weights frozen, only memory grows"), in
`memory/sessions/2026-09-28_predictor-memory-sight-theory.md`. Builds on D-029 (A1 stable, W1′ sufficient).

**Scheme.** The emulator is trained once for one specific dynamic (here pseudo-SSH/SST), e.g. on 10 years
of data, then used operationally from the very next day:

    psi_{t+a} = P( psi_{t+1} | M_t ),      M_{t+1} = M_t ∪ { psi_{t+1} }

- **Observer:** frozen after training (A1). Maps each new field to its state psi_t.
- **Predictor weights:** FROZEN after training. No online gradient updates (config (2)'s "online
  training" is dropped; the W8 collapse onto one model is thereby explicit).
- **Memory M_t:** the emulator's own captured trajectory; APPEND-ONLY — each newly OBSERVED state is
  appended (by time index); forecasts are never appended (W10: no error feedback).
- **Input:** a single time step psi_{t+1} is legitimate — the history lives in M_t, not in the input.
  "Sight" = how much of M_t the predictor uses.

**Evaluation protocol (module-2 standard from now on).** Mirrors "train on 10 years, use from the next day":
1. Train the observer on `[0, t_tr)` only (NOT the full record), freeze it.
2. Train the predictor on `[0, t_tr)`, freeze it.
3. Stream t = t_tr … T−1: append the observed psi_t to M, forecast psi_{t+1 … t+a}, reveal, score
   (strictly causal "predict, reveal, update"). No zero-state restarts, no fixed library.
Currently t_tr = 1400, T = 2000 (the existing split).

**What this invalidates / reframes.**
- F-18 and F-19 used an encoder trained on the FULL `[0,2000)` (no time split in `src/train/train.py`),
  i.e. an observer that saw the scored period; and F-18's GRU restarts from a zero hidden state with 32
  warmup steps at every launch; F-19's analog library was frozen to the train split. All three depart
  from D-030 — their numbers are indicative, not D-030 results.
- Open: W3 — new days whose dynamics are absent from the training period cannot be held by the memory.

## Numbering collisions with the encoder track

The module-2 (`latent-predictor`) branch and the parallel `stationary-observer` (encoder) session both
continued the shared D-/F- sequence from the same starting point, so three IDs now mean two different
things (flagged in both tracks' records at merge time):

| ID | module 2 | encoder |
|---|---|---|
| **D-024** | architecture settled at 1 step, not at rollout | spectral band ladder: timescale assigned per channel |
| **F-11** | family-specialized predictor experts buy nothing | readout could not say "stationary" |
| **F-12** | architecture matters at rollout; window trade-off | K is not the lever on the family population |

Both versions of each live in this file / `memory/findings.md`, disambiguated in their headings
("module-2" / "(encoder track)", "(module-2 track)"). Cross-references are ambiguous as a result —
e.g. the encoder's D-025 cites "D-024/F-11" meaning *its own* entries. Left unresolved deliberately:
renumbering either side's records is a merge-time decision for the repository owner. Everything after
the collision (encoder D-025 … D-027, F-13 … F-17; module 2 F-18, D-028) is unique.
