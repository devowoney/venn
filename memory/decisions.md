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
> Companion records: SOP 03 (`decisions.md`) (the SOP), `memory/findings.md` module-2
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
in `progress.md` → Session narratives, 2026-09-28 (predictor-memory-sight-theory).

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
`progress.md` → Session narratives, 2026-09-28 (predictor-memory-sight-theory). Builds on D-029 (A1 stable, W1′ sufficient).

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

## D-031 — The emulator is a 3-part recognition process; module 2 = history forecaster trained and scored on separate halves

**Date:** 2026-09-29. **Source:** user rulings in the `history-forecastor` session (progress.md narrative
2026-09-29), signed off by the user ("yes, record D-031 and F-21"). Refines D-012 and D-030. Spec: SOP 04.

- **Three parts, brain analogy** (naming fixed by the user 2026-10-01): (1) **encoder = eye-hippocampus = observer**
  (module 1); (2) **latent emulator = prefrontal cortex = history predictor** (module 2, this decision); (3) **decoder
  = cerebellum = memory interpreter** (module 3: mapping, reconstruction, data assimilation). Superseded names:
  "eye-lobe" for the encoder, "unconscious memory" for the decoder, and an intermediate draft (same day) that called
  module 2 the "hippocampus" and module 3 the "prefrontal cortex". All three
  are trained on ONE series of the dynamical system (here pseudo-SSH/SST), then frozen.
- **History = the observer-captured trajectory** `psi_0..psi_t` (all K channels). The forecaster takes NO input
  beyond it: it reads the history and predicts `psi_{t+1}` (primary) and, as a "how far" readout, `psi_{t+a}` up to
  a = 64 via direct heads. No private state; its own forecasts are never appended (W10).
- **Correction to D-030's "single time step input":** inference needs a LONG series of observations to INITIALIZE
  the history. Protocol: the history is initialized with the whole training series, then grows by appending each
  newly observed state; weights stay frozen.
- **Training vs validation data:** ONE long field `data.T=4000`, cut into equal halves — training `[0,2000)`
  (encoder via `train.t_train=2000`, forecaster weights, standardization, family labels, readouts) and validation
  `[2000,4000)` (scoring only). Why one field and not two generator calls: the generator's single RNG draws the
  spatial patterns AND the dynamics, so a second seed is a different system; one continuous trajectory keeps
  "history initialized on the training series" meaningful. Hyperparameters are chosen on a development slice at
  the end of the training set (`hf.t_fit=1750`), never on validation.
- **Default model:** causal transformer over the full history (token = all K channels + increment, rotary
  positions, zero-initialized residual heads = persistence at init); dropout 0.4, weight decay 0.3 (dev sweep, F-21).
- **Readout standard:** RMSE (training-std units) against BOTH references — persistence (each forecast's own
  initial condition held constant) and climatology (training mean) — plus corr, amplitude, hidden-mode readout.
  Persistence alone is not enough: beyond ~8 steps it is worse than climatology (F-21).

## D-032 — Observer (eye-lobe) track: North Star, protocol, and the reconstruction score R² / noise ceiling

**Date:** 2026-09-30. **Source:** user rulings in the `feature-observer` session (progress.md 2026-09-30). Evidence:
F-24, F-25. Full derivation of the score: `memory/reconstruction_score.md`.

- **Branch.** Observer work lives on `feature-observer` (worktree `.claude/worktrees/feature-observer`, branched from
  `history-forecastor` @ `7815f61`), never on `history-forecastor` (user: "the correction with this agent should be in
  another branch"). The old `stationary-observer` branch was deleted locally (tree identical to main `868f8eb`).
- **North Star (observer).** "A stable observer (over training time) that makes sufficient meaningful signal to
  reconstruct the original states." Hypothesis under test: long enough data → stable activated pixels capturing the
  three families.
- **Protocol.** Training `[0,8000)`, validation `[8000,10000)` of one `T=10000` field; `train.max_steps=8000`
  (user: "to see whole dynamic"); comparison arm training `[0,2000)` on the same field.
- **Stability measure.** Feature flag `train.snap_every` (default 0 = off, artifacts unchanged): masks saved every N
  steps (`masks_snap`, `snap_steps`, plus `t_fit`). "Activated pixel" = mask > 0.5; scored as IoU vs the previous
  snapshot (is it still moving?) and vs the final mask (how far has it travelled?).
- **Reconstruction score = R² / noise ceiling** (user: "yes, plot R²/ceiling"). R² = balanced (per-cell
  standardized), decoder fitted on the training slice, scored on validation. Ceiling C = mean_c(1 − σ²/Var(y_c)),
  σ = `data.obs_noise`. R² = C × R²_state, so R²/C = the fraction of the recoverable state reconstructed; raw R² is
  NOT comparable across seeds (C = 0.49-0.88 from the pattern layout alone).
- **Known limit, recorded with the decision:** on the M=10 testbed (14 independent signals) any K ≥ 14 readings reach
  R²/C ≈ 1, so the score cannot rank observers there; it discriminates only when K < 2(M−1) − 4 (F-25).
- **Open, not decided:** (a) harder testbed (e.g. M=40, 74 signals) with K sized to it; (b) the family ladder at
  many modes (0 stationary channels at M=40); (c) a longer budget + stop rule for "stable".

## D-033 — Brain naming of the three parts; module 3 = frontal cortex (decoder) maps the latent forecast to the field

**Date:** 2026-10-01. **Source:** user rulings in the `memory-intepreter` session (progress.md 2026-10-01), naming
signed off ("4. Yes"). Amends the naming of D-031 / SOP 04; refines D-012's module 3. Spec: SOP 05.

- **Branch.** Module-3 work lives on `memory-intepreter` (worktree `.claude/worktrees/memory-intepreter`, branched
  from `main` @ `b8a4e7b`, which holds both the observer and the history-forecaster tracks).
- **Names (supersede D-031's).** (1) **eye-lobe** = encoder / observer (module 1); (2) **hippocampus** = history
  forecaster (module 2; D-031 / SOP 04 called it "prefrontal cortex"); (3) **frontal cortex** = decoder (module 3;
  D-031 called it "unconscious memory"). Older records keep their wording; read "prefrontal cortex" there as the
  hippocampus.
- **Job of the frontal cortex (user: "the core idea").** Forecast in the latent history, then map the latent forecast
  to the physical field: `x_hat(t+a) = Dec( S_hat(t+a) )`. Replaces D-012's pseudo-inverse
  `x(t) + M+(s_hat - M x(t))` as the latent -> field map.
- **"Information hole" = (a):** field state that the eyes' current reading does not capture (eyes fewer than the
  field's independent signals); the decoder may read the latent HISTORY to fill it. NOT cross-variable inference
  (no SSH -> SST or SST -> SSH).
- **Testbed:** eyes at the level of the modes (observer sweep, F-25: ceiling once K ~ D = 2(M-1) - 4) -> the existing
  M=10 / K=16 setup (`hf10k_enc_seed*`, train [0,8000), validation [8000,10000)), where the state is FULLY observed.
  The M=40/K=16 hole regime is not the default (user: "Nope").

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

---

## SOPs — the settled pipeline, stage by stage (formerly `memory/sop/`)

> Golden Rule: if logic changes, update the relevant SOP section below BEFORE changing the code it
> describes. Code docstrings cite these as "SOP 00" … "SOP 04". SOP 00 is the overview; 02a is the
> plain-words companion to SOP 02. SOP 04 describes the `history-forecastor` branch's module-2 design.

---

## SOP 00 — Pipeline Overview (A.N.T. layer A)

> Golden Rule: if logic changes, update this SOP (and the relevant 0x SOP) BEFORE changing code.
> Scope: the whole VENN emulator. Detailed per-stage SOPs live in sibling `0x_*.md` files.

### A.N.T. mapping (per LLMAIProjectInstruction.md Phase A)

- **A — Architecture** (the SOP sections of `decisions.md`): these SOPs. Goals, tensor contracts, call order, failure modes.
- **N — Navigation**: which script runs when (this doc's call order). No heavy compute here.
- **T — Tools** (`./src/`): deterministic, testable modules. `data/ models/ train/ eval/`.
  All ephemeral I/O routes through `./tmp/`.

### Modules & call order (D-012)

```
                 field[T,V,H,W]
                      │
   (1) Decompositor:  │  selection-mask encoder + whitening      s = M·x
                      ▼
                 S[T,K]  (K scalar channels)
                      │
   (2) Latent predictor: per-channel 1D dynamics                 s(t) → ŝ(t+1)
                      ▼
                 Ŝ[T,K]
                      │
   (3) Forecaster:    │  gradient inversion  min‖M·x − ŝ‖²        x(t)+M⁺(ŝ−M·x)
                      ▼
                 x̂[T,V,H,W]   (moves only within K-dim mask subspace; complement frozen)

   Classifier (post-hoc, reversible): S[:,i] → {stationary, cyclic, chaotic}. Labels only.
```

### Kernel semantics (D-013 — read before touching the encoder)

- A mask is a **spatial sensor footprint at some scale/resolution**, NOT an orthogonal unmixer.
  Masks may OVERLAP; a scalar being a mixture of latent modes is fine.
- Masks span many scales: small energetic patch (Gulf-Stream-like jet) ↔ basin-scale footprint
  (equatorial band). Temporal family emerges from *where/what scale* a mask looks.
- **"Deep" = many channels.** Value is a large bank of diverse channels whose *collective* is a
  meaningful encoding. Success = collective encoding quality + channel diversity + emergent
  temporal spread — NOT per-kernel isolation.

### Tensor contract (Milestone 0)

| symbol | shape | dtype | notes |
|---|---|---|---|
| `field` | `[2000, 2, 64, 64]` | float32 | var0 = SSH-like, var1 = SST-like; per-var standardized |
| `S` | `[2000, K]` | float32 | K=16 for the first smoke test; K a config knob afterward |

### File-management & tooling conventions (D-014 — binding)

- **`./.tmps/`** — ALL ephemeral artifacts: script outputs, logs, cached tensors, scratch.
  Safe to delete. (Renamed from `./tmp/`; overrides the framework default.)
- **`./results/`** — curated payload (metrics tables, plots, kept checkpoints). Write **only with
  explicit user authorization**; default outputs go to `./.tmps/`.
- **`./config/`** — Hydra config tree. Every ML hyperparameter lives here; nothing hard-coded. A run
  is fully defined by its composed config (reproducibility).
- **TensorBoard** — training/eval log to TB event files via `torch.utils.tensorboard.SummaryWriter`
  (base env writes; tools env / server views). Log loss components, whitening error, per-channel
  variance (scalars) and masks + scalar series (images).

### Build order (Phase L, current)

1. SOP 01 → `src/data/synthetic.py` — generate `field` + hidden truth.
2. Minimal PROVISIONAL K=16 encoder (`src/models/encoder.py`) — plumbing only, defaults flagged.
3. Smoke-test probe (`src/probes/smoke.py`) — assert shapes/dtype, one forward + one backward,
   log tensor stats. This is the Phase L.2 handshake.

Open (confirm before the REAL training build, not the smoke test): mask parameterization
(soft-binary anneal vs STE), mask domain (full-volume vs per-layer), `s_i` normalization,
whitening hard vs soft, predictor history window, concrete collective metric, deeper K.

---

## SOP 01 — Synthetic SSH/SST Generator (Milestone 0)

> Golden Rule: update this SOP before changing `src/data/synthetic.py`.
> Realizes D-009 (synthetic testbed) as reframed by D-013 (multi-scale sensors, hidden truth).

### Goal

Produce a deterministic (seeded) synthetic field with **spatially heterogeneous, multi-scale
dynamics**, so that *which region/scale a mask sees* determines the temporal character it reads.
Latent modes are recorded as a **hidden answer key** for post-hoc analysis — they are NEVER fed
to the model (training is unsupervised).

### Output contract

| name | shape | dtype | notes |
|---|---|---|---|
| `field` | `[T=2000, V=2, H=64, W=64]` | float32 | var0=SSH-like, var1=SST-like; each var standardized to ~0 mean, unit var over (t,x,y) |
| `truth` | dict | — | hidden key: per-mode `{scale, family, phi[H,W], amp[T], vars, lag}` + `config` + `seed` |

Reproducible from `(seed, GenConfig)` alone. Smoke test writes to `./.tmps/` as `.npz` (ephemeral).

### Latent process  `field_v(t,x,y) = Σ_k w_{v,k} · a_k(t − lag_{v,k}) · φ_k(x,y) + noise_v`

Modes organized by SCALE (this is the D-013 heterogeneity). Default counts keep the recorded
1/3/6 family mix (D-009), now mapped onto a scale ladder:

| family | count | scale | spatial φ_k (rev2, D-021) | temporal a_k(t) |
|---|---|---|---|---|
| stationary | 1 | LARGE | equatorial band, **zero-mean** → positive core, negative surround | **CONSTANT** `amp = stationary_amp`, un-standardized (rev3, D-025). Legacy: Ornstein–Uhlenbeck, long τ — a slow DRIFT, which is now classed `cyclic` |
| cyclic | 3 | MEDIUM | **dipole** (`bump(c1) − bump(c2)`) and low-wavenumber **wave** (`cos(2π(kx·x/W + ky·y/H) + φ)`), alternating | sinusoids, periods P≈[60,140,300] + small noise |
| chaotic | 6 | SMALL | tight Gaussian monopoles σ≈2–4 (eddies), centres over the FULL domain, periodic in x | 2× Lorenz (σ=10,ρ=28,β=8/3) → x,y,z = 6 series |

- **φ_k** normalized to max |φ|=1. **a_k** standardized to unit variance (guards Lorenz scale) —
  EXCEPT the constant stationary mode, whose whole point is that it is not standardized (rev3).
- **SSH/SST coupling (rev2, D-021):** SST is a **first-order (AR1) response** to the SSH forcing,
  `sst_k(t) = ρ·sst_k(t−1) + (1−ρ)·a_k(t)`, `ρ = exp(−1/sst_tau)` — a physical low-pass (SST
  integrates flux), which damps amplitude AND shifts phase. For `sst_tau=20` against period 60 the
  phase lag is ~64°, so `corr(SSH,SST) → ~0.45` instead of the 0.98 the old fixed 3-step lag gave.
  Additionally `n_private_ssh` / `n_private_sst` modes appear in ONE variable only (previously all
  10 modes appeared in both). Encoded per-mode via `w_{v,k}`.
- **noise_v:** small additive Gaussian (obs noise), std = `obs_noise` × field std.

### Three defects this SOP's rev2 fixes (findings F-8 → D-021)

Diagnosed by LOOKING at the field (`src/probes/plots_field.py`), not from metrics:

1. **Dead borders.** Centres were drawn from `uniform(12,H−12)` / `(6,H−6)` and every pattern was a
   LOCAL bump, so a ~25% border ring had near-zero temporal variance — and that is exactly where
   the encoder's kernels fled (F-6). Fix: centres over the FULL domain, **periodic in x**
   (longitude wraps), plus basin-scale wave patterns that have support everywhere.
2. **SSH ≈ SST** (cell corr 0.86, domain-mean 0.981) — the second variable was nearly redundant.
   Fix: AR1 response + private modes, above.
3. **All-positive φ_k** → 100% of cell pairs positively correlated → channel decorrelation is
   ARITHMETICALLY impossible for a non-negative mask (F-4). Fix: dipoles, waves, zero-mean band.
   Small eddies stay monopoles (physically right); the large/medium scales carry the sign structure.

**Back-compatibility:** every rev2 knob defaults to the OLD behaviour in `GenConfig`, and the new
values live in `config/config.yaml`. Runs recorded before 2026-08-27 therefore regenerate their
exact original field from their own saved config (`src/probes/evaluate.py` depends on this).

### rev3: the stationary mode is CONSTANT (D-025, user ruling 2026-09-01)

**The defect.** `_ou_series` ends with `_standardize`, so mode 0 -- the "stationary" one -- was
emitted at UNIT VARIANCE: exactly as much temporal energy as the three sinusoids and the six Lorenz
channels. It was a slow wanderer (tau=200), not a stationary signal, and the testbed therefore
contained NO flat mode for the encoder to find. Every "the encoder cannot capture the stationary
mode" conclusion drawn before 2026-09-01 was measured against a mode that was not stationary.

**The fix** (`stationary_constant: true`, GenConfig default False so older runs still reproduce):

```
amp_0[t] = stationary_amp        # flat, NOT standardized -> var = 0
field    += amp_0 * phi_0        # a static spatial offset, unchanging in t
```

Two details that matter:

- **SST must bypass the AR1 response.** A constant's AR1 steady state IS that constant, so routing
  it through `_ar1_response` would add a startup transient -- and that function's closing
  `_standardize` divides a zero-variance series by ~0, which would delete the mode from SST
  entirely. The generator now detects `std(amp) < 1e-12` and passes the constant through.
- **The constant survives field standardization.** `fv = (fv - fv.mean()) / fv.std()` removes a
  GLOBAL SCALAR mean, not a per-cell one, so a spatially structured static pattern is preserved.
  Verified: the field's time-mean map correlates **1.000** with `phi_0`, at rms 0.818 against a
  field std of 1.0.

**Consequence downstream.** A constant mode has zero variance, so anything that standardizes a
series or correlates against one degenerates on it: `mode_r2` now returns `None` for it (a
regression with an intercept "explains" any constant perfectly) and the probe judges it with
`stationary_capture` / `amp_ratio` instead. The family labeller tests flatness FIRST, on the raw
amplitude. See D-025 and SOP 02.

### Determinism & config

- `numpy.random.default_rng(seed)` only (no global RNG). `GenConfig` dataclass holds all knobs
  (T, grid, counts, periods, τ lag, OU τ, obs_noise, energetic-site centers).
- Log the resolved config + seed into the returned `truth` and into `progress.md` on each run.

### Edge cases / failure modes

- **Lorenz blow-up / NaN** → integrate with small dt + subsample; standardize a_k; assert finite.
- **lag indexing t<τ** → hold the initial value (clamp index at 0), do not wrap.
- **zero-variance var** (all-noise cancels) → guard std by ε before standardizing.
- **dtype drift** → cast final `field` to float32 explicitly; assert shape/dtype at the boundary.

### Verification (Phase L.2 handshake — `src/probes/smoke.py`)

1. `field.shape == (2000,2,64,64)`, `dtype == float32`, all finite.
2. Minimal PROVISIONAL K=16 selection encoder → `S.shape == (2000,16)`, finite.
   (encoder defaults — soft-binary `sigmoid(logit/T)`, full-volume domain, normalize by Σmask —
   are PROVISIONAL plumbing, not the final design; see open points in SOP 00.)
3. One slowness loss `L=mean Σ_i (s[t+1]−s[t])²` → `L.backward()`; assert mask grads finite,
   grad-norm > 0 (gradients actually reach the masks).
4. Log per-var field stats (mean/std/min/max), S stats, L, grad-norm. Success/failure line.

### Visual verification (added 2026-08-27)

```
conda run -n oceanai python -m src.probes.plots_field            # newest run, with kernel overlay
conda run -n oceanai python -m src.probes.plots_field --no-run   # field only
```

Produces, in `<run>/figs/`: `field_snapshots.png` (SSH/SST in time + mean + temporal-std maps),
`field_modes.png` (the hidden answer key: every phi_k and a_k(t)), `kernel_activation.png` (where
the kernels look, weighted by each one's drop-one unique contribution), and `field.gif` (the field
evolving with kernel outlines pulsing at their instantaneous activation).

**Always look at these before trusting a metric.** They exposed four generator weaknesses that the
scalars did not (findings F-8): SSH and SST are 0.98-correlated (V=2 nearly redundant), every
phi_k is strictly positive (which caps channel decorrelation, F-4), a ~25% border ring of the
domain has near-zero variance (the dead zone the kernels fled into, F-6), and the chaotic modes
are spatially tiny and temporally intermittent.

---

## SOP 02 — v0 Encoder + Training (slowness + whitening)

> Golden Rule: update this SOP before changing `src/models/encoder.py` or `src/train/train.py`.
> Realizes the settled v0 core (CLAUDE.md): learnable selection masks, slowness loss, whitening
> constraint, online pairwise SGD. Defaults recorded in D-015; all are Hydra-tunable.

### Goal

Learn K selection masks whose K scalar channels are **slow** (temporally coherent) under a
**whitening** constraint (unit variance + decorrelation). Whitening blocks the trivial optima
(M=0 collapse, duplicate kernels); decorrelation is what drives channel diversity (D-013).

### Tensors & call order

```
field[T,V,H,W]  --SelectionEncoder-->  S[T,K]   (s_i = <mask_i, field>, norm="none")
   per training step:
     idx ~ U[0,T-2], batch B          -> pairs (t, t+1)
     s_t, s_tp1 = enc(field[idx]), enc(field[idx+1])   # [B,K] each
     C       = cov(center(s_t))                         # [K,K], DIFFERENTIABLE
     v       = diag(C)                                  # [K] channel variance
     R       = C / sqrt(v_i v_j)                        # [K,K] correlation, unit diagonal
   mode "corr" (DEFAULT, D-017):
     L_slow   = mean_i var(s_tp1 - s_t)_i / v_i         # SFA Rayleigh quotient
     L_white  = mean_{i!=j} R_ij^2                      # decorrelation
     L_var    = mean_i relu(1 - std(s_i/sqrt(count_i)))^2  # anti-death hinge
     L_energy = mean_i relu(1 - e_i/e_ref)^2            # energy floor, e_i = v_i/count_i^2  (D-019)
     L_size   = mean_i relu(|log(count_i/tgt_i)| - log(tol))^2   # SCALE LADDER, tgt_i geometric (D-020)
     W        = argmin_W ‖X - [1,S]W‖^2                 # per-batch closed form, DIFFERENTIABLE
     L_recon  = ‖X - [1,S]W‖^2 / ‖X‖^2                  # = 1 - R^2, coverage       (D-019)
     L        = λ_slow·L_slow + λ_white·L_white + λ_var·L_var
                + λ_energy·L_energy + λ_recon·L_recon + λ_size·L_size
                + ladder terms: L_band, L_line (cyclic rungs), L_struct (fast rungs), L_level
                  (stationary rungs) -- see the D-024 / D-025 / D-027 sections below
   mode "hard" / "soft": ABLATIONS. Both collapse -- see "Failure modes" below.
```

**Why an energy/coverage term is REQUIRED, not optional (D-019, finding F-6):** `L_slow` is a
ratio, so it is blind to amplitude. Left alone it walks every kernel into the lowest-energy cells
of the domain (the edges), where the field is the far tail of the big slow mode: tiny amplitude,
almost purely slow, near-perfect ratio. Scale-invariance is mandatory to prevent the mask collapse
(F-2) and simultaneously rewards looking where nothing happens. `e_i = v_i/count_i^2` is the fix at
the per-channel level (`count^2` normalization makes it independent of mask SIZE, so it measures
"are the selected cells energetic AND mutually coherent"). `L_recon` is the fix at the collective
level, and additionally penalizes REDUNDANCY: a duplicate channel adds no explanatory power, so its
gradient points at the unexplained residual.

**The one rule that matters here (D-017):** every normalization in the loss must be
DIFFERENTIABLE. Autograd treats a `.detach()`-ed variance as a constant, so a "scale-free" loss
written with a detached std is *not* scale-free to the gradient, and the optimizer will shrink
every mask to zero. Three separate collapses (findings F-2) all trace back to this. No `.detach()`
in the loss path, no matrix inverse, and the ratio `var(Δs)/var(s)` written explicitly.

### What the observer IS, and how channels are classified (D-026, user ruling 2026-09-01)

**The observer is not a mode-recovery device.** Hidden modes are just hidden modes; in a real ocean
or atmosphere, capturing all of them is barely possible. What the kernel learns is WHERE TO LOOK, and
a family is a property of **the observable the channel constructs** — not of some injected mode. A
stationary channel may be a single static mode, a persistent phenomenon, or a COMBINATION of varying
signals whose sum barely moves. All three are legitimate.

This changes the SCORING, not just the code:

| quantity | status |
|---|---|
| `population` vs 1/3/6 | a **DESIGN** target — the mix of observables we ask the bank to build (the rung ladder). NOT a mode-recovery score. |
| `role_obedience` | the real check on the ladder: did each channel become what its rung asked for? |
| `amp_ratio < 0.05` | the ONLY criterion for a stationary channel. No persistence check (user's call: flatness over the record is enough). |
| `mode_recovery` (max\|corr\|, ensemble R²) | DIAGNOSTIC. |
| `mask_align_with_pattern` | DIAGNOSTIC. Judging a flat channel by its alignment with the injected pattern was the wrong frame. |
| `cancellation_vs_independent` | DIAGNOSTIC: HOW a flat channel got flat (see below). |

**Cancellation is allowed but not encouraged** — no loss term rewards it. `L_level` rewards flatness
however the kernel achieves it. To tell the mechanisms apart, the probe projects every hidden mode
onto the flat channel's footprint and compares the net fluctuation to the independent-addition
baseline `sqrt(Σ sd_k²)`:

```
< 0.7   destructive interference beyond chance   -> a cancelling COMBINATION
~ 1.0   incoherent addition                      -> a genuinely quiet footprint / persistent phenomenon
> 1.3   the contributions reinforce
```

Measured on the ladder run `20260901_080506`, ch1: 3 hidden modes reach the footprint, index
**1.01** — so that stationary observer is a quiet footprint, not a cancellation. (Normalize against
INDEPENDENCE, not against the plain sum of the parts: 3 equal independent parts already give 0.58 of
the sum, so a "net/sum = 0.63" reading looks like cancellation while being exactly what chance
predicts. The wrong baseline invents findings.)

#### Classification follows GLOBAL STRUCTURE, not the residue

A channel that visibly oscillates while carrying fast noise on a wandering baseline is CYCLIC. My
earlier ACF-recurrence test read the RAW series, where the noise and the drift dominate the
autocorrelation, so it answered a question about the residue and called such channels non-cyclic.
`family.decompose` now splits a channel four ways:

```
level    = mean(x)                        -> gates the stationary label via amp_ratio
trend    = periodogram bins 1..3          -> fewer than 4 cycles in the record: at that resolution
                                             a drift and a "cycle" are indistinguishable, so this
                                             band IS the drift band. Counts toward CYCLIC (D-025).
osc      = the dominant peak's HALF-POWER band (bins >= 4), min +-2 bins. Adapts to the peak: ~1 bin
           for a pure tone, wide for a quasi-periodic hump -- a fixed window would have scored a
           broad hump as residual, i.e. called a real oscillation chaotic.
residual = the rest, above bin 4          -> genuinely broadband
```

`label = flat gate (amp_ratio) first, then trend+osc vs residual`.

**Why the level GATES instead of joining the argmax.** A plain argmax over all four shares is
tidier but wrong: `level² > var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its
level would score "stationary". Counter-example on the record: ch14 of run `20260901_080655`,
amp_ratio 0.526, plainly broadband chaotic.

**Measured (post-hoc — a label is a readout, so no retraining is needed to change one):**

| | truth accuracy | population |
|---|---|---|
| rev3 truth modes | **10/10** | 1/3/6 |
| rev2 truth modes | 9/10 — the "miss" is the old OU mode 0 now labelled **cyclic** (trend share 0.63), which is D-025 as ruled | — |
| rev3 ladder `20260901_080506` | — | 2/5/9, role obedience **1.00** |
| rev3 baseline `20260901_080438` | — | 0/11/5 |

Channel-level: ch2-6 of the ladder run are cyclic at osc share 0.70-0.93. ch9/ch11 stay chaotic at
osc share 0.41/0.32 against residual 0.59/0.67 — real oscillatory character with broadband still
dominant, which is the honest reading of "has cyclic characteristics but is not 100% cyclic".

**Known limit:** a non-sinusoidal periodic signal puts power in harmonics, which land in `residual`
and bias the vote toward chaotic. Harmonic folding is a deferred refinement.

### STATIONARY = CONSTANT (D-025, user ruling 2026-09-01) — read this before the ladder section

The section below was written with "stationary" meaning a SLOW, long-memory channel. That is not
this project's definition. **A stationary signal is a CONSTANT one — flat in time.** Consequences,
all measured (finding F-13):

- **The testbed had no flat mode at all.** `_ou_series` ends in `_standardize`, so the old mode 0
  was emitted at UNIT VARIANCE — as much temporal energy as the sinusoids. Generator rev3
  (`data.stationary_constant: true`) makes mode 0 a static pattern with a flat amplitude, living in
  the field's TIME MEAN. Verified: amp std 0.000, and the field's time-mean map correlates 1.000
  with the mode-0 pattern (rms 0.818 vs field std 1.0).
- **A slow red DRIFT is `cyclic`, not stationary** (user's call: it is not constant). The labeller
  therefore tests FLATNESS first, on `amp_ratio = std/|mean|` measured on the RAW series — a
  standardized series carries no amplitude information, so the test is skipped rather than guessed
  when the caller cannot supply it. 10/10 on the hidden truth over seeds 0-4.
- **The loss was blind to a constant, and partly hostile to one.** Every term uses the CENTERED
  channel (`sc = s_t - mu`), so a static signal is invisible to slowness, whitening, coverage and
  the energy floor; and `l_var = relu(1 - std)^2` plus `l_energy` actively PENALIZE a flat channel.
  So the stationary rungs now get `L_level` and are exempted from `l_var` / `l_energy` / `L_band` /
  `L_line`:

```
level_i = |mean_t s_i|      fluct_i = std_t s_i        r_i = level_i / (level_i + fluct_i)
L_level = mean over SLOW rungs of relu(flat_target - r_i)^2
```

  `flat_target` MUST sit at or past the labeller's cut or the hinge goes quiet before the channel
  qualifies: `r = 1/(1 + amp_ratio)`, and flat means `amp_ratio < 0.05`, so `r > 0.952`. A first
  attempt at 0.8 corresponded to `amp_ratio` 0.25, i.e. a channel that still visibly moves. 0.95 sat
  exactly ON the cut and left no margin: the push faded just before the line and rungs parked at
  0.05–0.07. **Default since 2026-09-27: `flat_target = 0.98`** (amp_ratio ≈ 0.02) — flat channels
  0.8 → 1.6 per seed, obedience 0.93 → 0.97, no cost to cyclic coverage or reconstruction (D-027).
  No rung reaches 0.02 (best 0.023), so the limit is now what the masks can align with, not the push.
  `L_level` also replaces the protection `l_energy` was giving those rungs (F-6): it is what keeps a
  flat channel on the static STRUCTURE instead of in an empty corner.

**Measured, rev3 field, K=16, 5000 steps, seed 0:**

| | baseline (no ladder) | ladder + `L_level` |
|---|---|---|
| run | `20260901_080438` | `20260901_080506` |
| population (s/c/ch) | 0/14/2 | **2/5/9** (target proportion 1.6/4.8/9.6) |
| role obedience | n/a | **1.00** |
| flattest channel `amp_ratio` | 0.778 (*still moving*) | **0.023** |
| flat channels (< 0.05) | 0 | 2 |
| balanced recon R^2 | 0.7763 | 0.7765 |

The baseline provably cannot produce a stationary observer; nothing in it gets near flat. Note rev3
scores 0.78 balanced recon where rev2 scored 0.99 — the new field has a large static component that
anomaly-based decoding does not explain, and both runs are equally affected.

**Figure caveat (fixed, but know why).** `plots.py` used to standardize each channel before
plotting, dividing out the std — the very quantity that defines flatness. A channel fluctuating at
2% of its level was drawn as dynamic as a sinusoid. It now scales by RMS about zero and prints
`amp_ratio` per lane, so a constant channel reads flat. Second time a figure has misled us on this
project (cf. F-6): check what a plot normalizes by before trusting it.

**Tested and rejected:** de-aligning the size and timescale ladders (`slow_size_frac=0.08`) to
un-pin the flat rungs from the largest footprints. It halved the footprint (2590 -> 1318 cells) but
did not improve alignment with the injected pattern (|corr| 0.27 either way) and cost role obedience
(1.00 -> 0.94). Default 0.0.

**Still open:** the flat channels report a stable level but align with the injected pattern at only
|corr| 0.27 — they settle on a diffuse static average or the pattern's negative lobe. Open question
for the user: must the observer ISOLATE the static pattern, or only report a stable level?

### Spectral band ladder — the TIMESCALE assignment (D-024, added 2026-08-31)

**The problem it solves.** `L_slow` is ONE objective shared by all K channels, so the channels
compete for the same globally-slowest content. Raising K therefore does not redistribute the
dynamical families — it buys more near-duplicates of whatever regime dominates. Measured on the
K sweep (`.tmps/runs/20260831_0736*`), populations read 3/0/1 (K=4), 4/2/2 (K=8), 12/0/4 (K=16),
22/3/7 (K=32), 36/10/18 (K=64): slow-dominated at every K, almost no clean cyclic channel.
**K is not the lever on which families appear.** (These numbers only became visible after the
labeller was fixed — see F-11 and `src/probes/family.py`.)

**The mechanism** is the temporal twin of the D-020 size ladder: stop asking every channel for the
same thing, and ASSIGN each one a target band, penalizing only outside it.

```
roles     = rung_roles(K, pop_target)      # [1,3,6] scaled to K -> K=16: 2 slow / 5 cyclic / 9 fast
band      = band_plan(...)                 # [K,n_bins] which rFFT bins each rung should occupy
S_win     = n_win random CONTIGUOUS windows of length win_len, sliced from enc(field)
p_i(k)    = periodogram of channel i, averaged over windows, DC bin dropped
bandfrac_i= in-band power / total power
linefrac_i= power within +-2 bins of the peak / total power
rho_i(L)  = lag-L autocorrelation, L = mem_lag

L_band = mean_i relu(band_target - bandfrac_i)^2
L_line = mean_i { relu(line_target - linefrac_i)^2   for CYCLIC rungs   (be a line)
                { relu(linefrac_i  - line_cap  )^2   otherwise          (be a hump, not a line)
L_mem  = mean over SLOW rungs of relu(mem_target - rho_i(mem_lag))^2
L     += λ_band·L_band + λ_line·L_line + λ_mem·L_mem
```

**Why three terms and not one.**
- The families OVERLAP in frequency on this testbed (cycles at 60/140/300, OU at tau=200), so a
  frequency band cannot separate a slow drift from a slow cycle. `L_line` does that, via the same
  line-vs-hump statistic the labeller uses (0.94 for the truth's cycles, 0.67 for its OU).
- `L_band` alone left the slow rungs on the WRONG content: tau_e 36/27 against the truth's 106,
  best-matching hidden mode a *chaotic* one at corr 0.50 (run `20260831_091525`). "Power at periods
  >= 128" is too weak — any large red-ish footprint satisfies it. `L_mem` is the sharp version:
  at lag 64, OU(tau=200) gives rho=+0.73 while the period-300 and period-140 cycles give +0.23 and
  -0.96. With `L_mem` on, that rung reached tau_e 87 and corr **0.92** with the true stationary mode.
- A cycle whose period nearly divides `mem_lag` aliases back to rho~1, so `L_mem` alone would admit
  a metronome; `line_cap` on the same rung rejects it. The two terms are complementary.

**Rejected alternative (do not re-try):** matching a target lag-L autocorrelation `exp(-L/tau_i)`
per channel. It is wrong for a cyclic channel, whose ACF oscillates and whose long-lag gap reaches
4 — twice the exponential maximum — so it would actively suppress the cyclic family it was meant
to create. Frequency-domain band + line shape has no such bias.

**Why windows, not pairs.** A lag-1 pair minibatch says nothing about a period-300 cycle. The
periodogram needs contiguous time, so the ladder slices `n_win` random windows out of `enc(field)`
(one einsum over T x 8192 cells — cheap enough to redo every step). The stochasticity that D-011
asks for now comes from the window starts rather than the pair index.

**Measured (K=16, 5000 steps, seed 0, same config otherwise):**

| | baseline `spectral.enable=false` | + band/line ladder | + `L_mem` (default) |
|---|---|---|---|
| run | `20260831_091454` | `20260831_091525` | `20260831_091908` |
| population (s/c/ch) | 10/1/5 | 6/1/9 | 5/1/10 |
| role obedience | n/a | 0.75 | 0.69 |
| slow rung -> true stationary | (unassigned) | corr 0.50 to a *chaotic* mode | **corr 0.92, tau_e 87** |
| worst hidden-mode maxcorr | 0.25 (m6, m9) | 0.65 | 0.61 |
| effective rank / 16 | 9.23 | 10.08 | 9.56 |
| balanced recon R^2 | 0.9891 | 0.9891 | 0.9891 |

Reading: the fast rungs obey almost perfectly (9-10 of 9-10 land chaotic, gap1 ~0.10-0.13 against
the truth's 0.07-0.20, where the baseline's "chaotic" channels sat at 0.02-0.03), every hidden mode
is now seen by some channel at |corr| >= 0.61 (baseline left two chaotic modes at 0.25), and
reconstruction is unchanged — the ladder redistributes the channels without costing coverage.
`figs/features.png` shows the bank stratified slow -> cyclic -> fast down the channel index, where
the baseline's families are interleaved with no order.

**Known open gap (NOT fixed by this ladder).** The cyclic rungs still mostly fail: 1 of 5 reaches a
clean line (`linefrac` 0.92; the rest sit at 0.52-0.60 and get labelled stationary). A non-negative
regional mask sums everything under its footprint, and the basin-scale slow content leaks into every
region, so a channel cannot cancel its own red background — the same non-negativity limit as F-4,
now showing up in the time domain. Candidate levers, in order of cost: (a) a temporal high-pass in
the definition of `s_i` for cyclic/fast rungs, which changes the observer's semantics (D-005) and
needs a user decision; (b) more sign structure in the generator's mid-scale patterns; (c) accept it
and let module 2 handle mixed channels. The second slow rung also fails, and it is the LARGEST
footprint on the size ladder — the D-021 conflict between `L_energy` and the biggest rung. Not
aligning the two ladders is the obvious next test.

### Fast-rung shape term `L_struct` (D-027, added 2026-09-10 → 2026-09-27)

**The problem.** Channels were "ambiguous cyclic or chaotic" (user, 2026-09-10). Measured on run
`20260901_080506`: 7 of 16 channels sat within the ambiguity band, **all of them fast rungs**, at
structure share 0.33–0.44 against the labeller's cut at 0.5. The cyclic rungs were decisive
(0.74–0.93). The hidden chaotic modes score 0.17–0.35, so the fast channels really were
contaminated by oscillatory content — an encoder problem, not a labeller problem.

**Root cause: the objective and the readout asked different questions.** The labeller votes on
`(trend + osc) / total` vs `residual / total` (D-026). The training hinge `line_cap` scored power
within ±2 bins of the peak, capped at 0.75, on 512-step windows. It was measured **silent on all nine
fast rungs** (they sat at 0.15–0.45), so nothing ever pushed them to be broadband. Same lesson as
`flat_target`/`FLAT_THR` (D-025), in a harder form: there the two disagreed on the *threshold*, here
on the *quantity*.

**The term** (`structure_term` in `src/train/spectral.py`):

```
S_all    = enc(field)                          # [T,K] full record -- the same measurement the probe makes
p_i(k)   = |rFFT(s_i - mean)|^2, DC dropped
trend_i  = power in bins 1..3                  # < 4 cycles in the record: the drift band
osc_i    = power in the dominant peak's half-power band above bin 4, at least ±2 bins
struct_i = (trend_i + osc_i) / total_i         # = the labeller's `cyclic_share`
L_struct = mean over FAST rungs of relu(struct_i - struct_cap)^2      struct_cap = 0.30
```

- It asks a fast rung for at least 70% broadband power. The cap is calibrated on the physics: the
  hidden chaotic modes measure 0.17–0.35.
- It never filters the signal. `s_i = <mask_i, field>` is unchanged (D-005); the gradient moves the
  mask toward cells whose signal is broadband.
- Peak location and band edges are detached (integer set memberships); the power summed inside them
  is differentiable. A ratio of powers → scale-free in value and gradient (D-017).
- `lambda_struct = 2.3`, gradient-matched to `lambda_white` at init, K=16 seed 0:
  `|g_white| = 5.1e-3`, `|g_struct| = 2.3e-3`.

**Why fast rungs ONLY (`struct_rungs: fast`, default).** The first variant also asked the cyclic
rungs for `struct ≥ 0.85`. `trend + osc` credits ANY clean peak and never checks the rung's assigned
octave, so on seed 0 all five cyclic rungs moved to the period-60 cycle, the cleanest line in the
field (baseline: 286/143/61/61/61). The cap on the fast side favours no mode. The cyclic rungs keep
`L_line` (`line_target = 0.85`); `L_line` is switched off on the fast rungs, whose shape `L_struct`
now owns. `struct_rungs: all` reproduces the first variant; `shape_objective: line` reproduces D-024.

**Measured (K=16, 5000 steps, 5 seeds each, runs `.tmps/runs/<arm>_seed{0..4}`):**

| arm | ambiguous ch / seed | fast struct max | distinct cyclic periods (of 3) | flat ch / seed | role obedience |
|---|---|---|---|---|---|
| `base` — `shape_objective: line` | 6.8 | 0.44 | **2.4** | 0.8 | 0.91 |
| `struct` — L_struct on all rungs, λ 3.5 | 3.0 | 0.31 | 1.0 | 0.6 | 0.91 |
| `fastonly` — L_line averaged over 5 rungs (bug) | 1.2 | 0.31 | 1.4 | 0.8 | 0.93 |
| `fixavg` — L_line averaged over 14, flat_target 0.95 | 2.2 | 0.31 | 2.6 | 0.8 | 0.93 |
| **`flat98fix` — default, + flat_target 0.98** | **1.2** | 0.31 | **2.6** | **1.6** | **0.97** |

"Ambiguous" = structure share in 0.35–0.65 (|margin| < 0.30), flat channels excluded. The 0.30 is a
convenience, not a calibration: hidden chaotic mode m6 sits exactly on its edge (margin −0.30).
Coverage is unchanged (balanced recon R² 0.7765 vs 0.7765, seed 0).

**Fixed — the `L_line` averaging (2026-09-27).** The first fast-only version averaged `L_line` over
the 5 cyclic rungs instead of all 14 dynamic rungs, making it 2.8× stronger per cyclic rung. Seed-0
diagnosis (runs `traj_base_*`, `traj_fast_*`, each `max_steps=N` run is bit-identical to the first N
steps of the full run): at step 1 every cyclic rung already sits on the period-60 cycle; within ~20
steps `L_band` pulls ch2/ch3 to their octaves in the baseline (in-band 0.18 → 0.74, 0.08 → 0.55),
while the stronger "be a clean line" held them on period 60, which already is one (in-band stuck at
0.22 for 4,980 steps). Restoring the per-rung weight gave 286/143/61/61/61 on seed 0
(`ablate_linenorm_seed0`, `lambda_line 1.43`). `spectral_terms` now averages the selected rungs over
`chan_w`. Result `fixavg`: distinct cyclic periods 2.6 (baseline 2.4); the fast-rung cap is kept.

**Remaining ambiguity is no longer on the fast rungs** (fast struct ≤ 0.33 on every seed). The 2.2 per
seed are stationary rungs that miss the flat cut and then enter the vote (5 cases: seeds 0, 1, 3, 4)
and cyclic rungs on slow cycles with chaotic contamination (6 cases, seeds 2 and 3; e.g. seed 3 ch6:
a period-140 cycle with chaotic bursts, structure 0.63).

**Still open, built into the ladder:** `L_band` loses on two cyclic rungs in every arm — ch6's octave
(23–43) contains no cycle, ch4's (64–128) sits just above period 60 — and 5 rungs share 3 cycles, so
some seed always misses one cycle (seed 3: no rung on period 60).

**Always check timescale coverage, not just ambiguity.** The first variant scored well on the
ambiguity count while every cyclic rung sat on one cycle. `.tmps/score_runs.py` now prints each cyclic
rung's dominant period and the number of distinct periods; the kernel/signal figures
(`src/probes/plots.py`) are what exposed it.

### Time split `train.t_train` (feature flag, added 2026-09-29 for module 2 — D-030/D-031)

The observer may learn from the PAST only. `train.t_train: null` (default) = the historical behaviour, every
statistic and sample from the full record `[0,T)`. `train.t_train: 1400` = the pair minibatch, the spectral
windows, the flat-level term, `e_ref`, `cell_mean` and `x_ref` all come from `field[:1400]`; nothing after it
reaches a gradient. The saved `artifacts.npz` `S` is still encoded over the FULL `[0,T)` with the frozen masks,
so module 2 can score on days the observer never saw. Required by the history forecaster (SOP 04).

### Input normalization `train.input_norm` (feature flag, added 2026-10-01 — EXPERIMENT, not a default)

What the eye sees. The generator already standardizes each VARIABLE globally (one mean/std for SSH, one for SST);
inside a variable the cells keep their natural loudness, so energetic cells dominate every `s_i`.
`none` (default) = that field, the historical behaviour. `cell_z` = every cell z-scored, `(x - mean_c) / std_c`:
equal loudness per cell, but the time mean is gone, so the stationary mode (which lives ONLY in the time mean,
rev3) becomes invisible and the flat rungs lose their target. `cell_scale` = `x / std_c`: equal temporal loudness,
time mean kept. Both lift pure-noise cells (std ~ obs_noise) to unit variance. Statistics from `[0, t_fit)` only
(`train.t_train`). Helper: `src/data/synthetic.py: normalize_input`; every probe that RE-ENCODES the field from
masks must apply it (`evaluate`, `observer_stability`, `plots_observer`); probes reading `artifacts.npz` `S` are
consistent automatically. Reconstruction targets stay the generator's field.

### v0 defaults (D-015 — all overridable via ./config/)

| knob | default | alt (deferred) |
|---|---|---|
| mask parameterization | soft-binary `sigmoid(w/temp)` | straight-through binary; signed `tanh` (ablation) |
| mask init | **multiscale** (own length scale per channel, D-016) | random / flat (flat is degenerate) |
| temp anneal | **OFF** (`temp0 = temp1 = 1.0`, D-016) | anneal → 0.1 AFTER masks organize; STE |
| mask domain | full-volume `(V,H,W)` | per-layer `(H,W)` |
| `s_i` normalization | **none** (whitening owns the scale) | soft-count / sqrt-count (ablation) |
| whitening | **`corr`** (differentiable off-diag decorrelation, D-017) | `hard`/`soft` (both collapse) |
| energy floor | `lambda_energy=0.2` (gradient-matched) | off (kernels flee to dead corners) |
| coverage | `lambda_recon=0.3` (gradient-matched) | off (chaotic modes get discarded) |
| scale ladder | `lambda_size` on a geometric footprint ladder (D-020) | off (all masks shrink to ~0.3% of domain) |
| band ladder | `spectral.enable=true`, weights gradient-matched (D-024) | off (families do not redistribute at ANY K) |
| stationary rungs | `spectral.slow_objective=level` + `L_level` (D-025: flat = constant) | `memory` (reads stationary as a slow DRIFT -- ablation) |
| shape term | `spectral.shape_objective=structure`, `struct_rungs=fast`: `L_struct` caps fast rungs, `L_line` on cyclic rungs (D-027) | `struct_rungs=all` (cyclic rungs collapse onto one cycle); `line` (D-024, fast rungs drift to the boundary) |
| stationary mode | `data.stationary_constant=true` (rev3) | false = legacy OU drift at unit variance |
| slowness gap | L2 | L1 |
| optimizer / regime | Adam, pair-minibatch (B random consecutive pairs) | pure single-pair online |

### Reproducibility & I/O (D-014)

- All hyperparameters in `./config/config.yaml` (Hydra). A run = its composed config.
- Run dir under `./.tmps/runs/<timestamp>/`; artifacts + figures written there.
- **ONE TensorBoard dir per run:** both `train` and `evaluate` write into `<run>/tb/`,
  separated by tag namespace (train `loss/*`,`var/*`,`energy/*`; eval `masks/*`,`S/*`,
  `collective/*`). Never a second event dir -- two per run is unreadable for a human.
- Seeds fixed (numpy generator for data, `torch.manual_seed` for model).
- Artifacts (masks, final S, metrics.json) → `.tmps/`. Promotion to `./results/` needs user OK.

### TensorBoard log schema

- train scalars: `loss/total`, `loss/slow`, `loss/white`, `loss/var_hinge`, `loss/energy`,
  `loss/recon`, `loss/size`, `loss/band`, `loss/line`, `loss/mem`, `loss/level`,
  `band/flat_ratio_max`, `band/frac_{min,mean}`,
  `band/line_{min,max}`, `band/rho_lag_max`, `energy/density_{min,max}`, `mask/count_{mean,min,max}`,
  `var/{mean,min,max}`, `slow_per_ch/{min,max}` (emergent slow↔fast spread), `grad_norm`, `temp`.
- eval (SAME `tb/` dir): `masks/trained`, `masks/init` (images), `S/ch*` (the K series),
  `collective/*`.
- Figures (`src/probes/plots.py` → `<run>/figs/`) are the PRIMARY readout, not the scalars:
  `masks.png`, `masks_on_energy.png`, `features.png`, `feature_var.png`. Always look at them.

### Edge cases / failure modes

- **Collapse (M→0, s→0):** the FAILURE MODE OF `whitening: soft` — observed 2026-08-26: masks go
  to 0, `var→2.5e-6`, `white→K=16` (i.e. `‖0−I‖_F²`), `slow→3e-7`. The gap term is quadratic in the
  mask scale while the penalty is bounded by `K`, so shrinking always wins. `whitening: hard` is
  the fix: the loss is measured on the whitened `z`, which is scale-free, so collapse buys nothing.
- **Ill-conditioned whitening:** if `cond` explodes, two channels have become near-duplicates;
  `white_eps` floors the eigenvalue and `grad_clip` caps the spike. Persistent high `cond` means K
  exceeds the usable slow subspace.
- **Batch < K:** covariance rank-deficient → require `B ≥ 2K` (default B=128 ≫ K=16).
- **Anneal too fast:** masks freeze before organizing → keep temp1 not too small; it's tunable.
- **NaN:** guard denom `clamp_min(1e-6)`; assert finite S each log step.

### Verification

```
conda run -n oceanai python -m src.train.train            # ~5000 steps, seconds on an H100
conda run -n oceanai python -m src.probes.evaluate        # reads the newest .tmps/runs/*
```

Healthy run (measured 2026-08-26, K=16): `offcorr2` falls 0.99 → ~0.27 and flattens by ~2500
steps; `var` stays O(1e2..1e5) (NOT falling toward 0); `hinge` stays 0.0000; `slow_ch` spreads by
>20× between the slowest and fastest channel. The probe should then report effective rank ≳ 4,
mask IoU ≲ 0.2, and several channels with |corr| > 0.8 against a hidden mode.

Known-bad signatures: `var` decaying monotonically past ~1e0 → a scale-invariance leak in the
loss; `offcorr2 → 0` with `var → 0` → total mask death; effective rank ≈ 1 → the constraint is not
differentiable.

### Known structural limits (NOT bugs — see D-018, findings F-4/F-5)

- Decorrelation saturates at off-diag corr² ≈ 0.244 / effective rank ≈ 4 of 16 because the masks
  are NON-NEGATIVE and cannot cancel a shared large-scale mode. λ_white is irrelevant (identical
  at 1, 20, 100, 500). Largely a GENERATOR artifact: every `φ_k` is a positive bump, so 100% of
  cell pairs are positively correlated (findings F-4 caveat).
- **The family population is NOT a function of K** (D-024, measured 3/0/1 -> 36/10/18 over
  K=4..64, slow-dominated throughout). A shared objective gives K copies of one optimum; the
  timescale must be ASSIGNED per channel. Same shape of problem as the mask-size collapse (D-020).
- Minimizing MEAN slowness discards the chaotic modes — FIXED by `lambda_recon` (D-019): chaotic
  ensemble R² 0.01–0.03 → 0.79–1.00. The energy floor alone does NOT fix it (0.01–0.06).
- `lambda_slow` should stay SMALL. At 10 it collapses diversity (eff rank 2.95, spread 31× → 2.3×).
- Masks shrink from the ~50%-of-domain init to a median of ~28 cells (0.3%) unless `lambda_size`
  holds them on the ladder; `L_energy` accelerates the shrinkage because mean pairwise covariance
  is maximized by a tiny coherent patch (D-020).

---

## SOP 02a — Loss guide — what every term asks for, in plain words

Companion to SOP 02 (the SOP, which holds the formulas' history, measurements and
decisions). This file is the **plain-language reference**: what each term asks the encoder for, why it
exists, when it is silent, and what it cannot do. Code: `src/train/train.py` and
`src/train/spectral.py`. Weights: `config/config.yaml`. State as of 2026-09-27 (D-027).

---

### 1. Vocabulary

| word | meaning |
|---|---|
| **field** | the synthetic ocean: `[T=2000, V=2 (SSH, SST), 64, 64]`, standardized to std 1 |
| **mask / kernel** | one learned map in [0, 1] over the grid (both variables). Bright = selected cells |
| **channel** `s_i(t)` | what mask *i* reads: `s_i(t) = <mask_i, field_t>`, the weighted sum of the selected cells. One number per time step. K = 16 channels |
| **observer / encoder** | the bank of K masks. It decides **where to look**; it never filters or transforms the signal |
| **count** | a mask's total weight = its footprint size in cells |
| **family** | what kind of signal a channel carries: **stationary** (does not move: constant), **cyclic** (moves with structure: an oscillation or a slow drift), **chaotic** (moves broadband: no clean rhythm) |
| **rung** | a channel's **assigned slot** on a ladder: a pre-set target the channel is asked to meet. The name comes from ladder steps — rungs are ordered, each one a step up or down in size or timescale |
| **role** | which family a rung is asked to produce. K = 16 → **2 stationary, 5 cyclic, 9 fast (chaotic)** rungs, in the 1/3/6 proportion. Channel order: ch0–1 stationary, ch2–6 cyclic, ch7–15 fast |
| **size ladder** | each channel gets a target footprint, from basin-scale (ch0, ~30% of the domain) down to a small patch (ch15, ~16 cells) |
| **band ladder** | each channel gets a target frequency band: stationary rungs slow, fast rungs fast, and the 5 cyclic rungs one **octave** each, tiling periods 320 → 24 steps (ch2 longest, ch6 shortest) |
| **octave** | a band of periods where the longest is about twice the shortest |
| **hinge** | a penalty that is exactly 0 once a target is met, and grows only on the wrong side: `relu(target − x)²`. Hinges **guide** a channel into a zone; they do not pin it to a value |
| **scale-free** | written as a ratio, so multiplying a mask by any number changes neither the loss nor its gradient. Mandatory here: every non-scale-free version made the masks shrink to zero (D-017) |
| **gradient-matched weight** | each λ is set so that, at initialization, that term's gradient is as strong as `L_white`'s. Loss *values* differ by ~300× and would mislead (D-019) |
| **baseline** | the configuration before the change being tested, run on the same seeds |
| **seed** | fixes all randomness of a run: the synthetic field itself, the initial masks, the sampled time steps. Seeds 0–4 = five different random oceans |
| **ambiguous channel** | structure share between 0.35 and 0.65, i.e. the labeller's cyclic/chaotic vote is close to a coin flip (the 0.30 margin is a convenience, not calibrated) |

---

### 2. The big picture: three kinds of question

The loss has **11 terms** (9 active by default). They answer three different questions:

1. **Is the bank usable at all?** Channels alive, on signal, distinct, covering the field, at a
   spread of sizes. → `L_slow`, `L_white`, `l_var`, `l_energy`, `l_size`, `l_recon`.
2. **Which timescale does each channel watch?** → `L_band` (where in frequency).
3. **Which family does each channel become?** → `L_level` (does it move at all?),
   `L_line` (cyclic rungs: is it one clean oscillation?), `L_struct` (fast rungs: is it broadband?).

One sentence to remember: **`L_band` decides where a channel looks in frequency, `L_line` / `L_struct`
decide what shape of signal it finds there, and `L_level` decides whether it moves at all.**

Why group 1 alone cannot produce families: `L_slow` gives every channel the same instruction and
`L_white` only makes them different in *place*, not in *dynamics*. The K sweep proved it: 4 to 64
channels, always slow-dominated (F-12). So the families are **assigned** by the ladder, and the
labeller checks afterwards whether the assignment took (`role_obedience`).

---

### 3. Group 1 — a usable bank

#### `L_slow` — "change slowly"  (λ = 1.0)

```
L_slow = mean_i  var(s_i(t+1) − s_i(t)) / var(s_i)
```

- **Plain words:** how much a channel changes from one step to the next, relative to how much it
  varies overall. Small = slow. Equals `2(1 − lag-1 autocorrelation)`: 0 for perfectly smooth, 2 for
  white noise (period-300 sinusoid ≈ 0.0004, period 60 ≈ 0.011, chaotic ≈ 0.36).
- **Why a ratio:** `var(Δs)` alone is won by shrinking the mask. Dividing by `var(s)` removes scale.
- **Limits:** one instruction shared by all channels, so they all chase the same slowest content.
  Blind to a constant (it only sees fluctuation). Weak in practice: ~24× smaller gradient than
  `L_white` at init. Kept as a tiebreak; removing it measured slightly worse.
- This is Slow Feature Analysis's objective; `L_white` is its constraint.

#### `L_white` — "don't copy another channel"  (λ = 1.0, the reference weight)

```
L_white = mean over pairs i≠j of corr(s_i, s_j)²
```

- **Plain words:** the average squared correlation between channels. 0 = all unrelated, 1 = all
  copies. Trained runs end ≈ 0.06 (typical |corr| ≈ 0.25).
- **Why correlation, not covariance:** covariance is won by shrinking masks; a detached whitening
  makes redundancy free. Both failed (the `soft` / `hard` ablations, D-017).
- **Limits:** makes channels different, not different *kinds*. Compares values at the same moment
  only (a delayed copy is not penalized). With non-negative masks, overlapping footprints correlate,
  so it pushes masks apart and small — one reason `l_size` exists.

#### `l_var` — "don't fade to nothing"  (λ = 1.0)

```
l_var = mean_i relu(1 − std(s_i) / sqrt(count_i))²
```

- **Plain words:** a floor on each channel's fluctuation. `std/sqrt(count)` ≈ 1 for a mask over
  unrelated cells, larger for cells moving together, → 0 as the mask fades.
- **Silent** for every live channel; acts only on a dying one (VICReg's variance term).
- **Exempt:** stationary rungs (they are meant not to fluctuate).
- **Limits:** keeps a channel alive, not useful — it can sit in a quiet corner (F-6). Leans slightly
  against channels built from cancelling parts.

#### `l_energy` — "look where the field moves"  (λ = 0.2)

```
e_i = var(s_i) / count_i²                      average covariance between the selected cells
l_energy = mean_i relu(1 − e_i / e_ref)²       e_ref = 0.371, the field's mean per-cell temporal variance
```

- **Plain words:** the selected cells must be energetic **and** move together, at least as much as an
  average single cell varies. Independent of footprint size.
- **Not a quiet safety net:** active through most of training; at convergence every dynamic channel
  sits right at the floor (0.93–1.28).
- **Exempt:** stationary rungs.
- **Side effect:** the cheapest way to score high is a tiny coherent patch, so it pulls masks small;
  `l_size` holds it back.

#### `l_size` — "keep your assigned footprint"  (λ = 0.3)

```
l_size = mean_i relu(|log(count_i / target_i)| − log 2)²
```

- **Plain words:** each channel has a target footprint on the size ladder (ch0 ≈ 30% of the domain →
  ch15 ≈ 16 cells). Free within a factor 2 of the target, penalized outside.
- **Why:** left alone, every mask shrank to a ~28-cell patch, and the "multi-scale sensor" became 16
  small patches (D-020).

#### `l_recon` — "together, cover the whole field"  (λ = 0.3)

```
X      = field at 128 random moments, each cell's time mean removed          [128, 8192]
Wd     = best linear decoder of X from [1, standardized channels], solved exactly per batch
l_recon = unexplained fraction of X  =  1 − R²
```

- **Plain words:** can the 16 channel values, combined linearly, rebuild the field's motion?
  Collective (one number for the bank), spatial, ignores time order.
- **Also penalizes redundancy:** a duplicate channel explains nothing new.
- **Values:** ≈ 0.006 in training (energy-weighted R² ≈ 0.994). The probe's "balanced" R², where
  every cell counts equally, is ≈ 0.78 — the more demanding number.
- **Limits:** blind to the static pattern (means are removed), so stationary channels earn nothing
  here.

---

### 4. Group 2 — which timescale

#### `L_band` — "put your power in your assigned band"  (λ = 5.0)

```
bandfrac_i = power inside rung i's band / total power      (periodogram over 4 random 512-step windows)
L_band     = mean over dynamic rungs of relu(0.5 − bandfrac_i)²
```

- **Plain words:** at least half of a channel's fluctuation must sit in its assigned frequency band.
  Fast rungs: periods ≤ 24. Cyclic rungs: one octave each — ch2 171–512, ch3 102–256, ch4 64–128,
  ch5 39–73, ch6 23–43 steps (window-bin edges; the design range is 320 → 24).
- **Why:** this is what makes channels slow or fast. `L_slow` cannot: it asks the same of everyone.
- **Limits:** the families overlap in frequency on this testbed (cycles at 60 / 140 / 300), so a band
  alone cannot tell a slow cycle from a slow drift — that is the shape terms' job. It holds cyclic
  rungs weakly: ch6's octave contains no cycle at all, and ch4's sits just above the period-60 cycle,
  so both end on period 60 in every run. 5 cyclic rungs share 3 cycles.
- **Exempt:** stationary rungs (a constant has all its power in the mean, which the spectrum drops).

---

### 5. Group 3 — which family

#### `L_level` — stationary rungs: "don't move"  (λ = 0.13, target 0.98)

```
r_i     = |mean(s_i)| / (|mean(s_i)| + std(s_i))       1 = perfectly constant, 0 = pure fluctuation
L_level = mean over stationary rungs of relu(0.98 − r_i)²
```

- **Plain words:** the channel's level must dwarf its fluctuation. The labeller calls a channel flat
  when `amp_ratio = std/|mean| < 0.05`, i.e. `r > 0.952`; the target 0.98 (amp_ratio ≈ 0.02) sits past it.
- **Why it must exist:** a constant lives entirely in the channel's **mean**, and every other term
  works on the centred channel. Worse, `l_var` and `l_energy` actively forbid a flat channel — hence
  the exemptions.
- **Hard part:** the static pattern is spatially zero-mean, so a non-negative mask must align with
  one signed lobe of it; a random mask of the same size is nowhere near flat (amp_ratio ≈ 66).
- **Why 0.98, not 0.95:** 0.95 sat exactly on the readout's cut, so the push faded to nothing just
  before the line and stationary rungs parked at amp_ratio 0.05–0.07. 0.98 (amp_ratio ≈ 0.02) keeps
  pushing past the cut: flat channels 0.8 → 1.6 per seed, at no cost elsewhere. No rung reaches 0.02;
  the remaining limit is what a non-negative mask can align with, not the target.

#### `L_line` — cyclic rungs: "be one clean oscillation"  (λ = 4.0)

```
linefrac_i = power within ±2 bins of the spectral peak / total power    (512-step windows)
L_line     = sum over cyclic rungs of relu(0.85 − linefrac_i)²  /  number of dynamic rungs (14)
```

- **Plain words:** at least 85% of the power in one narrow peak. The hidden cyclic modes measure 0.94.
- **Why divided by 14 and not 5:** each cyclic rung must keep the weight it had when `L_line` also
  covered the fast rungs. Dividing by 5 made it 2.8× stronger per rung, and it then pinned the cyclic
  rungs on the period-60 cycle they start on, beating `L_band` in the first ~20 steps (seed-0
  diagnosis, 2026-09-27).
- **Limits:** it credits whichever peak is cleanest, regardless of the rung's octave. `L_band` is the
  only thing that says *which* cycle.

#### `L_struct` — fast rungs: "be broadband"  (λ = 2.3)

```
spectrum of the FULL series (2000 steps), mean removed, split into:
  trend    = bins 1–3 (periods ≥ 667: fewer than 4 cycles in the record — drift)
  osc      = the strongest peak above bin 4 and its neighbours down to half its power (≥ ±2 bins)
  residual = everything else — broadband
struct_i = (trend_i + osc_i) / total_i          = the labeller's own "cyclic_share"
L_struct = mean over fast rungs of relu(struct_i − 0.30)²
```

- **Plain words:** at most 30% of a fast channel's power may be drift or one dominant rhythm; at least
  70% must be broadband. The cap is the physics' own bar: the hidden chaotic modes measure 0.17–0.35.
- **It never filters the signal.** The gradient moves the mask toward cells whose signal is broadband.
- **Why it exists:** the labeller calls a channel cyclic if `struct ≥ 0.5`. The old fast-rung cap
  (`L_line` at 0.75 on a ±2-bin window) measured a different quantity and was silent on all nine fast
  rungs while six sat at the coin-flip boundary. The objective must measure what the readout measures.
- **Why fast rungs only:** on cyclic rungs, "at least 0.85 structured" credits any clean peak, and all
  five moved to the period-60 cycle (F-16).
- **Limits:** slow power that is neither the peak nor the very slowest drift counts as "broadband", so
  fast rungs can still carry 30–40% of their power at periods ≥ 100.

#### Ablation-only terms (off by default)

- **`L_mem`** (λ 0.4, `slow_objective: memory`): stationary rungs keep a long-lag autocorrelation.
  It encodes the old reading of "stationary" as a slow drift, superseded by D-025.
- **`L_line` on fast rungs** (`shape_objective: line`): the D-024 original, capping line fraction at
  0.75. Silent in practice.

---

### 6. Who is exempt from what

| term | stationary rungs (ch0–1) | cyclic rungs (ch2–6) | fast rungs (ch7–15) |
|---|---|---|---|
| `L_slow`, `L_white`, `l_size`, `l_recon` | yes | yes | yes |
| `l_var`, `l_energy` | **exempt** | yes | yes |
| `L_band` | **exempt** | yes (own octave) | yes (periods ≤ 24) |
| `L_level` | **yes** | — | — |
| `L_line` | — | **yes** | — |
| `L_struct` | — | — | **yes** |

---

### 7. Common properties

- **Every ratio is scale-free** in value and gradient: a mask cannot win by shrinking or growing.
- **Every ladder term is a one-sided hinge:** silent once satisfied, so it guides rather than pins.
- **Weights are gradient-matched at init** to `L_white` (D-019), then corrected by measurement where
  needed.
- **The objective and the readout must agree** on both the quantity and the threshold (F-13, F-15);
  otherwise a term goes quiet before the labeller is satisfied.
- **Never judge a run by one metric.** The ambiguity count looked good while all cyclic rungs sat on
  one cycle (F-16); the kernel and signal figures (`src/probes/plots.py`) showed it.

---

## SOP 03 — Latent Predictor (module 2)

> Golden Rule: update this SOP BEFORE changing `src/models/predictor.py`,
> `src/train/train_predictor.py` or `src/probes/evaluate_predictor.py`.
> Realizes **module 2 of D-012**: evolve each scalar channel forward in the compressed domain,
> `s(t) → ŝ(t+1)`. Defaults recorded in D-022; all are Hydra-tunable.

### Goal

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

### Why the predictor needs a HIDDEN STATE (the design crux)

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

### Channel independence (D-012 "per-channel 1D")

Every channel gets its **own parameters** (`per_channel: true`, default). Channel `i`'s predictor
never sees channel `j` — that is what makes the emulator a bank of independent 1D systems and what
lets the classifier's family labels mean something per channel. Implemented as **batched**
per-channel weight tensors `[K, in, out]` with `einsum`, so all K systems train in one pass.
`per_channel: false` (one shared predictor, channels folded into the batch) is available as an
ablation — it tests whether the channels share a common dynamics.

### Tensors & call order

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

### Baselines (computed in the same script, on the same split — non-negotiable)

| baseline | definition | why |
|---|---|---|
| `persistence` | `ẑ(t+1) = z(t)` | the honest zero, and a *strong* one for slow channels |
| `linear AR(p)` | per-channel least squares on the train split, closed form | separates "the channel is linear-predictable" from "the network learned something" |

Headline metric = **skill score** `1 − MSE_model / MSE_persistence` at each horizon, per channel and
aggregated. Positive = better than persistence. Reported per family label so we can see, e.g., that
the win is concentrated in the cyclic channels.

### Metrics written (all per channel and aggregated, TRAIN and VAL)

- `mse_h`, `r2_h`, `skill_h` for `h = 1 … horizon_eval` (free-running rollout).
- `acc_horizon`: first horizon at which rollout correlation with truth drops below 0.5 — the
  channel's practical predictability time. Expect: stationary ≫ cyclic ≫ chaotic.
- rollout **variance ratio** `var(ŝ)/var(s)` at long horizon: catches the classic failure where the
  model minimizes MSE by decaying to the mean. A model that flatlines scores *well* on MSE and is
  useless to module 3 — this ratio is the flag.

### Figures (mandatory — the standing rule is never to judge a run from metrics alone)

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

### Failure modes to watch (write findings when hit)

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

### Call order

```
1. python -m src.train.train                      # module 1, produces .tmps/runs/<ts>/artifacts.npz
2. python -m src.train.train_predictor  predictor.encoder_run=.tmps/runs/<ts>
                                                  # -> <ts>/predictor.pt, predictor_metrics.json, tb/
3. python -m src.probes.evaluate_predictor --run .tmps/runs/<ts>
                                                  # -> predictor_eval.json + figs_predictor/*.png
```

The predictor run logs into the **same** `<run>/tb/` directory as the encoder (D-020 convention:
one TensorBoard dir per run), under `pred/` tags.

### Measured on the CURRENT encoder — 2026-09-27 (`flat98fix_seed0`)

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

### Measured on the OLD encoder (Aug 27, superseded by the table above) — 2026-08-29 / 2026-08-31 (K=16 encoder run, generator rev2)

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

---

## SOP 04 — History Forecaster (module 2: latent emulator = prefrontal cortex = history predictor)

> Golden Rule: update this SOP BEFORE changing `src/models/history_forecaster.py`,
> `src/train/train_history_forecaster.py` or `src/probes/eval_history_forecaster.py`.
> Branch `history-forecastor` (2026-09-29). Supersedes SOP 03's per-channel step predictor as the module-2
> design under test; SOP 03 is kept as the record of the earlier line.

### The emulator as a recognition process (user, 2026-09-29)

| part | brain analogue | VENN module | job |
|---|---|---|---|
| (1) | eye-hippocampus | encoder / observer (SOP 02) | field `x_t` -> latent state `psi_t` (K=16 scalars) |
| (2) | **prefrontal cortex** | **latent emulator / history predictor (this SOP)** | history `psi_0..psi_t` -> `psi_{t+1}` |
| (3) | cerebellum | decoder / memory interpreter: mapping, reconstruction, data assimilation | `psi` <-> field |

All three parts are trained on ONE series of the dynamical system (here pseudo-SSH/SST). Weights are then
frozen (D-030).

### Training data vs validation data (user, 2026-09-29)

One LONG pseudo-ocean field of the same system, cut into two equal halves:

| set | time steps | used for |
|---|---|---|
| **training** | `[0, 2000)` | eye-hippocampus masks (SOP 02, `train.t_train=2000`), forecaster weights, standardization stats, channel family labels, hidden-mode readout |
| **validation** | `[2000, 4000)` | scoring only — no statistic, gradient or hyperparameter choice ever touches it |

Why: the question "can the prefrontal cortex forecast?" is only answered on days it never saw, from a history that
the operational emulator would really have. Same system, same trajectory, continued: the history at the first
validation day is the whole training series, so the validation set is exactly "trained on the past, used from
the next day". Hyperparameters are chosen on a DEVELOPMENT slice at the end of the training set (`t_fit=t_dev`),
never on validation.

Generator: `data.T=4000` (one call, seed = encoder seed). Why not two independent generator calls: the generator's
single RNG draws the spatial patterns AND the dynamics, so a second seed is a different system, not a second
realization of this one.

### The two premises this module realizes (user, 2026-09-29)

1. **The dynamics are captured by the observer, and that capture is the HISTORY.** The history at time t is
   the sequence of every latent state observed so far, `H_t = (psi_0, ..., psi_t)`.
2. **The forecaster needs no input beyond the history.** It reads `H_t` and predicts the latent state at the
   end of it, `psi_{t+1}` (and, as a readout of "how far", `psi_{t+a}` for a = 2..A).

Correction by the user (2026-09-29) to D-030's "single time step input": **inference takes a long series of
observations to INITIALIZE the history.** Operationally: the history is first initialized with the whole
training series `[0, t_tr)` = `[0, 2000)`, then grows by appending each newly observed state. Forecasts are NEVER appended
(no error feedback, W10).

### Data schema

```
upstream (frozen eye-hippocampus, SOP 02 with data.T=4000, train.t_train = t_tr = 2000):
  artifacts.npz : S [T=4000, K=16]  (encoded over the FULL record; masks learned on [0, t_tr) only)

standardize per channel with TRAIN-ONLY stats (channel std spans ~10 .. ~1000):
  z = (S - mu[:t_tr]) / sd[:t_tr]                      [T, K]

model  P : z[B, L, K]  ->  zhat[B, L, A, K]
  zhat[:, t, a-1, :] = forecast of z_{t+a} made from z_0..z_t ONLY  (causal)
  A = 64 leads; lead 1 is the primary target
```

### Model (`src/models/history_forecaster.py`)

- **Token = one time step, all K channels jointly**: `[z_t, z_t - z_{t-1}] -> Linear -> d`. The increment is
  computed FROM the history (not an extra input); it supplies the rate of change, which a purely spatial
  observer does not put into a single `psi_t`.
- **Causal self-attention** over the whole history (full sight), pre-norm blocks, rotary (relative) positions:
  attention depends on the LAG between two states, not on absolute time, so the history can keep growing past
  any length seen in training.
- **Direct heads, residual form**: `zhat_{t+a} = z_t + head_a(h_t)`. Heads start at zero -> the untrained model is
  exactly persistence, and learns only the departure from it.
- No private state: everything the model knows at time t is `H_t`.

### Training (`src/train/train_history_forecaster.py`)

- Fit only on targets `< t_fit` (`t_fit = t_tr` for the final run; `t_fit = t_dev < t_tr` for a DEVELOPMENT
  run that chooses hyperparameters on `[t_dev, t_tr)` — never on the scored period).
- Each step: B sequences `z[s0 : t_fit]`, `s0 ~ U[0, max_start)` -> long histories of varied length.
- Loss = MSE (standardized units) over every position t with `t - s0 >= min_sight` and every lead a with
  `t + a < t_fit`; leads weighted equally.
- AdamW, grad clip, TensorBoard `tb/` in the Hydra run dir `.tmps/runs_hf/...`.

### Inference = streaming evaluation (`src/probes/eval_history_forecaster.py`)

Because P is causal, ONE forward pass over `z[0 : T-1]` returns, at every t, exactly the forecast a streaming
emulator would have made with the history initialized on `[0, t_tr)` and grown by appending up to t.
(Checked by a causality test: perturbing `z_{t'}` for t' > t leaves the forecast made at t unchanged.)

Scored launches: `t = t_tr - 1 ... T - 1 - A` (all targets in the unseen period). Per lead a and channel k:

| metric | definition | reads |
|---|---|---|
| RMSE | `sqrt(mean (forecast - truth)^2)`, standardized units (1 = one training std) | the error itself, plotted against the two references below |
| RMSE persistence | forecast `z_{t+a} = z_t`: EACH forecast's own initial condition (launch state) held constant | "nothing changes"; grows with lead, exceeds climatology beyond ~8 steps |
| RMSE climatology | forecast = training mean (0) | "knows nothing"; ~1 at every lead: the fair long-lead reference |
| skill | `1 - MSE(model) / MSE(persistence)` | > 0 = better than "nothing changes" |
| corr | Pearson corr(forecast, truth) over launches | phase / pattern |
| amplitude | `std(forecast) / std(truth)` over launches | 1 = honest; -> 0 = mean-collapse (F-9) |
| horizon | first lead with corr < 0.5 | "how long" |

Grouped by the channel's FAMILY, labelled from its training part only (`src/probes/family.py`).
Hidden modes (answer key): a ridge readout `psi -> truth amplitudes` fitted on `[0, t_tr)`, applied to the
forecast; ceiling = the same readout on the TRUE future state (what a perfect forecast would give).

### Verification (Milestone-0 alignment)

1. causality test passes (max |change| of past forecasts under a future perturbation == 0);
2. lead-1 skill > 0 on the unseen period, on every seed;
3. amplitude near 1 (no mean-collapse) — rank by amplitude first (F-9);
4. figure checked with the user: forecast vs truth per family + skill/corr/amplitude vs lead.

### Chaotic channels carry the uncertainty: ensemble head (feature flag `hf.ensemble.enable`, user 2026-09-29)

Why: a point forecast trained on MSE converges to the conditional MEAN. For a chaotic channel that mean decays toward
climatology after a few steps (F-21: chaotic amplitude 0.75 at h64, RMSE 0.87 vs climatology 0.95) — one line cannot
be precise past ~1 Lyapunov time. User rulings: the uncertainty takes the form of an ENSEMBLE OF SAMPLES; it is
allowed on the CHAOTIC channels only ("if the eye-lobe is seeing well, stationary and cyclic dynamics have a high
reliability"); precision and uncertainty come from ONE model.

```
h_t = causal-attention(z_0..z_t)                          (shared, as before)
base_{t+a} = z_t + head_a(h_t)                            [A,K]   deterministic, every channel
eps_m ~ N(0, I_noise_dim), m = 1..M                       one draw per member per launch (shared over leads)
member_m = base + c * g(h_t, eps_m)                       [M,A,K] c_k = 1 if channel k is chaotic, else 0
```

- `c` = family labels of the TRAINING half (`src/probes/family.py`, same rule as the probe), saved with the run so
  training and evaluation use the same mask. Stationary / cyclic channels: all members identical = the base forecast.
- One eps per member per launch -> a member is a coherent 64-step possible future, not independent noise per lead.
- Loss: MSE of `base` on the non-chaotic channels + `lambda_crps` x fair ensemble CRPS on the chaotic channels
  (`mean_i |x_i - y| - sum_{i!=j} |x_i - x_j| / (2M(M-1))`, unbiased for finite M), same (position, lead) mask.
- `hf.ensemble.enable: false` (default) reproduces F-21 exactly.

Evaluation adds, on the same launches as F-21 (baseline = the F-21 deterministic finals):
| metric | definition | good |
|---|---|---|
| RMSE of ensemble mean | as RMSE, forecast = member mean | <= deterministic model |
| CRPS | fair ensemble CRPS (deterministic model: = MAE) | lower |
| spread / error | `sqrt((M+1)/M) * member std` over RMSE of the mean | ~ 1 = honest uncertainty |
| member amplitude | std over launches of ONE member / true std | ~ 1 = members are realistic, no mean-collapse |

#### Spread calibration on the held-out end of the training set (`hf.ensemble.calibrate`, 2026-09-29)

Why: trained on ONE trajectory, the ensemble learns the chaotic futures partly by heart, so on unseen days it is
OVERCONFIDENT (dev slice: spread/error 0.29-0.35 instead of ~1). History noise (`hf.input_noise` 0.1 / 0.3) barely
helped (0.35 / 0.43) and cost lead-1 precision (chaotic RMSE 0.10 -> 0.13 / 0.25). A per-lead inflation fitted on
held-out data fixes the spread: fitted on dev half 1, scored on dev half 2, spread/error 0.76-1.19, chaotic CRPS
-12 to -17 %, inflation 1.7 (h1) -> 3.6 (h64).

- Protocol: fit on `[0, t_fit=1750)`; after training, choose `s_a` per lead (grid 1..6, minimum fair CRPS over the
  chaotic channels, 32 members) on launches whose targets lie in `[t_fit, t_tr)` = `[1750, 2000)`; store it in the
  model buffer `spread [A]`. Validation `[2000, 4000)` is untouched.
- Applied inside the model around the ensemble mean: `dev' = mean_m(dev) + s_a (dev - mean_m(dev))`. The mean — and
  so the RMSE of the best estimate — is unchanged; only the spread is corrected. Requires `t_fit < t_tr`.
- Cost: the ensemble model is fitted on 1750 steps, its deterministic reference (F-21) on 2000.

#### Long-series variant (user, 2026-09-30): training `[0,8000)`, validation `[8000,10000)`

Why: F-21/F-22 trace the overconfidence and the missing chaotic precision to memorization of ONE 2000-step
training trajectory. Same system, same generator seed, 4x longer training record, SAME validation length (2000).
- Eye-lobe retrained: `data.T=10000 train.t_train=8000` (runs `.tmps/runs/hf10k_enc_seed{0..4}`).
- Forecasters: deterministic `hf.t_tr=8000` (fit `[0,8000)`); ensemble `hf.t_tr=8000 hf.t_fit=7750` (spread
  calibration on `[7750,8000)`, the same 250-step held-out slice as before). All other hyperparameters unchanged, so
  the only change against F-21/F-22 is the training length.

## SOP 05 — Frontal Decoder (module 3, "frontal cortex")

> Golden Rule: update this SOP BEFORE changing `src/models/frontal_decoder.py`, `src/train/fit_frontal_decoder.py`,
> `src/probes/eval_frontal_decoder.py` or `config/frontal_decoder.yaml`. Branch `memory-intepreter` (2026-10-01). D-033.

### Place in the emulator

| part | brain analogue | VENN module | job |
|---|---|---|---|
| (1) | eye-lobe | encoder / observer (SOP 02) | field `x_t` -> latent state `S_t` (K channels) |
| (2) | hippocampus | history forecaster (SOP 04) | history `S_0..S_t` -> `S_hat(t+a)`, a = 1..64 |
| (3) | **frontal cortex** | **frontal decoder (this SOP)** | latent (history) -> field; `S_hat(t+a)` -> `x_hat(t+a)` |

All three fitted on the same training set `[0, t_tr)` and frozen (D-030/D-031).

### Data-First schema

- In: frozen eye run (`artifacts.npz` `S[T,K]`, field regenerated from its `.hydra/config.yaml` + seed);
  frozen hippocampus run (`model.pt`, forecasts `S_hat[T-1, A=64, K]`, ensemble model -> mean of 32 members).
- Decoder input: latent window `[S(tau - L) for L in dec.lags]` -> `[len(lags) * K]`. In a forecast from launch t,
  window steps after t take the forecast `S_hat`, earlier steps the observed history.
- Out: field `x_hat[V=2, H=64, W=64]` (physical units).

### Model

Closed-form ridge, linear, in balanced units (per-cell standardization with training stats, the D-032 score's
units): `x_hat = mu_y + sd_y * (b + W . (window - mu_s)/sd_s)`. Why linear: the testbed field is a linear
superposition and every channel a linear read of it. `dec.lags = [0]` (default; fully observed testbed, history
adds nothing); `dec.ridge = 1e-3` (relative to the sample count). `dec.t_tr` must equal the encoder's `train.t_train`.

### Run

```bash
python -m src.train.fit_frontal_decoder dec.encoder_run=<eye run> hydra.run.dir=.tmps/runs_dec/<name>
python -m src.probes.eval_frontal_decoder --dec .tmps/runs_dec/<name>* --hf <hippocampus runs> --out .tmps/eval_dec/<name>
```

The probe pairs decoder and hippocampus through their shared encoder run.

### Readout standard

Per lead a = 1..64, validation launches only (all targets in `[t_tr, T)`): field balanced R^2 / noise ceiling, pooled
and per variable, for: forecast (hippocampus -> decoder), decoded TRUE latent (decoding ceiling = perfect
hippocampus), persistence `x(t)`, climatology, D-012 pseudo-inverse. Gap ceiling - forecast = hippocampus error;
1 - ceiling = decoding error. Plus field snapshots (truth / decoder / pseudo-inverse at leads 1, 8, 16, 64).
Note: persistence and the pseudo-inverse carry the observation noise of `x(t)`, so they sit below 1 already at lead 1.

#### Standardization stays inside the history predictor (user ruling, 2026-10-01)

Question raised by the user: the history comes out of the observer in raw latent units (`S`, levels up to ~-1247,
stds 19-364), while the prefrontal cortex reads and forecasts `z = (S - mu)/sd` (training stats, `norm.npz`) and hands
`S_hat = mu + sd z_hat` to the cerebellum, which rescales with its own statistics. The map is affine and invertible
(corr(raw, z) = 1.000 per channel, shapes identical), but it weights every channel equally and inflates the ~5 %
wobble of the stationary channels to unit variance. Options offered: move the standardization into the observer, or
let the forecaster read the raw history; and whether stationary channels should keep their level.
**Ruling: keep the current standardization** -- per-channel z-score inside the history predictor, stationary channels
scaled by their std like the others. All reported RMSE/CRPS stay in z units (1 = one training std of the channel).

#### Chaotic-memory ensemble: the uncertainty comes FROM the chaotic history (`hf.ensemble.mode: memory`, user 2026-10-02)

Why (user): weather-style initial-condition perturbation does not fit this configuration — (1) random (even Gaussian)
noise is not a good perturbation; (2) the model holds a history memory, so it is unclear where a perturbation would
go; (3) restricting the spread to the chaotic OUTPUT channels is arbitrary. The user's idea: the system's history and
memory contain the chaotic signal, and that is where the ambiguity comes from. Hold the chaotic history SOFTLY (a
learned noise-ensemble technique is fine there) and build the ensemble from it: each member is one plausible reading of
the ambiguous chaotic past; the ensemble represents the uncertainty, it is not a perturbation of the initial condition.

Contrast with the multi-scenario mode `mode: multi_scenario` (formerly `output`; F-22/F-23, the kept design): there the noise enters AFTER the history is read and is
masked onto the chaotic OUTPUT channels. In `mode: memory` it enters the MEMORY (the token embeddings attention reads),
only through the chaotic part, and the spread reaches EVERY output channel as far as its future depends on the chaotic past.

```
token_t     = inp_nc([z_t, dz_t] * (1 - c))  +  inp_c([z_t, dz_t] * c)          exact non-chaotic part + chaotic part
sigma_t     = softplus(W_s [z_t, dz_t] * c + b_s)             [d]   how loosely the chaotic memory is held at step t,
                                                                     learned from the CHAOTIC history only
token_t^m   = token_t + sigma_t * g(eps_m)                    eps_m ~ N(0, I_noise_dim): ONE code per member, shared by
                                                                     every past step = one coherent hypothesis of the past
h_t^m       = causal-attention(token_0^m ... token_t^m)       the same blocks read each member's memory
member_m    = z_t + head(h_t^m)                               [A,K]  all channels, same direct heads as the deterministic model
```

- `members = 0` -> no variation (eps = 0): the deterministic forecast, used by the causality check.
- Loss (user choice): fair CRPS on the CHAOTIC channels + MSE on the non-chaotic channels averaged over EVERY member
  (= MSE of the mean + member variance), so stationary/cyclic channels spread only where it lowers their error.
- Spread calibration unchanged (per-lead `spread` buffer, fitted on chaotic-channel CRPS over `[t_fit, t_tr)`), applied
  around the member mean on ALL channels.
- `b_s` initialized at -3 (sigma ~ 0.05): the model starts as the deterministic forecaster and learns how loose to be.
- Cost: attention runs once per member (M=8 in training) -> ~8x the deterministic training time.
- Readout: same probe, same launches; compared against `mode: multi_scenario` (F-23 runs) and the deterministic twin. New
  check: spread/error on the NON-chaotic channels (no longer 0 by construction).

**Status: REJECTED (user 2026-10-02).** First run (`.tmps/runs_hf/rb10k_mem_seed{0..4}`): the memory IS shifted
(|sigma*code| ~0.19-0.27 vs |token| ~1.4) but the forecaster learned to ignore it; raw member std 0.002-0.008, the
calibration hit its x6 ceiling, chaotic spread/error 0.04-0.09, chaotic RMSE h8 0.466 vs 0.346 (multi-scenario).
User's clarification of the intended idea: the uncertainty is generated by the CHAOTIC signal — when the predictor
reads the memory it should output multiple scenarios for the chaotic prediction. That is the multi-scenario mode,
which is kept as the design. Its spread is state-dependent: corr(member spread, |error|) over validation launches
+0.33 (h1) / +0.57 (h8) / +0.39 (h64); seed 0 h8: |error| 0.07 where the scenarios agree most (lowest 20 % spread)
vs 0.50 where they disagree most. The `memory` code path is kept only to reproduce this record.

#### Naming: `output` mode -> MULTI-SCENARIO mode (user 2026-10-02)

`hf.ensemble.mode: multi_scenario` (default) is the ensemble of F-22/F-23, formerly `output`. The old name is still
accepted when loading, so existing checkpoints reload unchanged.

#### Multi-scenario for the WHOLE family, spread set by a predefined certainty score (user 2026-10-02)

Why (user): make scenarios for every family, and let a predefined score say how likely the prediction is — e.g. a
stationary history predicted a few steps ahead is quite certain -> small spread; chaotic memory makes the next
prediction uncertain -> bigger spread. User choices: score = held-out error per CHANNEL x LEAD (not per family, so it
does not depend on the observer's labels); scenarios trained with fair CRPS on ALL channels.

- `hf.ensemble.families: all` (default `chaotic` = F-22/F-23 unchanged): the multi-scenario mask covers all K channels,
  so the generator writes scenarios for every channel; loss = fair CRPS on every channel (the MSE term is empty).
- **Certainty score** `score[a, k]` = RMSE of the scenario MEAN at lead a, channel k, on launches whose targets lie in
  the held-out end of the training set `[t_fit, t_tr)` = `[7750, 8000)`; saved as `score.npy` [A,K] with the run.
- **Spread from the score** (`hf.ensemble.spread_from: score`; default `crps` = the per-lead grid of F-22/F-23): after
  training, per lead and channel, `s[a, k] = score[a, k] / raw_spread[a, k]` (raw spread = finite-M corrected member
  std on the same launches), clipped to [0.1, 50]; members are widened around their mean by `s[a, k]`. So on held-out
  data each channel's spread equals its expected error at each lead: certain channels/leads -> narrow fan, uncertain
  -> wide. The SHAPE of the scenarios (which situation gets more spread) stays learned from the history by CRPS.
- `spread` buffer is now [A, K] (a per-lead [A] buffer from older runs is broadcast to [A, K] on load).
- Readout: same probe and launches; compare with the chaotic-only multi-scenario ensemble (F-23 rerun): RMSE of the
  mean and CRPS per family and lead, spread/error per family (now non-zero for every family), spread-skill correlation.
