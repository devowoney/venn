# findings.md - Research, Discoveries, Constraints

> Living project memory (per LLMAIProjectInstruction.md, Phase B.3 Research).
> Purpose: research notes, discoveries, references, and constraints.
> Reuse existing patterns and components when possible; note them here.

## Constraints observed so far

- environment.yaml present at project root: conda env `oceanai`, Python 3.12.
  - Scientific stack: numpy, scipy, pandas, xarray (>=2025.1, required for zarr v3 I/O), netcdf4, h5netcdf, zarr, dask, dask-image, numba, xesmf.
  - CUDA 12.8.1 toolkit + cudnn; ML stack via pip: torch, torchvision, timm, lightning, einops, hydra-core, omegaconf, tqdm.
  - Testing: pytest.
  - Implication: this is an oceanography / scientific-ML project. Per the spec's opening line,
    "LLMs are probabilistic; scientific logic must be deterministic" - deterministic logic belongs in ./src/ (Layer T).
- **Project goal:** an ML-based emulator for (ocean) signals.
- **Two-layer environment** (see decisions.md D-003):
  - Base `oceanai` (conda): training/inference core — PyTorch + scientific stack + CUDA.
  - Tools venv (separate): visualization, TensorBoard, notebook/localhost kernels. Isolated
    from base. A `requirements-tools.txt` (or equivalent) is expected but not yet created.

## Project concept / emulator architecture (authoritative — venn_idea.txt, 2026-08-24)

Goal: an **ML-based emulator of dynamical (ocean) signals** that compresses spatial
fields into many independent scalar channels and evolves each scalar in time.
(Project name "venn" ≈ the population of channels partitioned across three
dynamical families — see below.)

Pipeline / design:

1. **Encoding — global binary-mask kernels.**
   - Each scalar channel is produced by a **global convolution kernel**: a full-domain
     **binary (0/1) mask**. Because each kernel covers the ENTIRE spatial state, each
     scalar represents a **global spatial pattern**.
   - Many such kernels → many scalar **time series**. Mechanically each scalar is a
     projection: `s_i(t) = <mask_i, field(t)>` (global inner product / full-size conv).

2. **Independent 1D temporal evolution.**
   - Each scalar channel evolves **independently — NO coupling between channels.**
   - Each scalar is treated as its own **1D dynamical system**.
   - Rationale (user): simpler, more interpretable, more stable.

3. **Three dynamical families** for each scalar time series:
   1. stationary
   2. cyclic (periodic)
   3. chaotic

4. **Learned temporal classifier (post-hoc, reversible).**
   - A small model (1D CNN or tiny RNN) reads each scalar time series and assigns its
     family. Learns temporal signatures automatically.
   - Assignment is **post hoc** (after observing the series) and **reversible**: a
     channel can switch families during training as its behavior changes.

5. **Kernel self-organization (the key twist).**
   - Initialize ALL kernels as **random binary masks (0/1)** → scalars start **chaotic**.
   - Then modify kernels so the **population distribution** of temporal behaviors across
     all channels matches a **desired target distribution** over {stationary, cyclic,
     chaotic}. NOT forcing each kernel into a category — shaping the whole population.
   - Result: a **self-organizing loop** — classifier + target distribution guide kernels
     to gradually specialize; categories emerge and can flip during training.

**DEFINING PRINCIPLE (user, 2026-08-25): the full-size kernel makes a *selection*
("vision"/choice) among the spatial data — this is THE key difference from a CNN.**
- A CNN kernel is small, local, weight-shared, and slides to detect *local* features.
- This kernel is full-domain, does NOT slide; its role is to *choose which spatial cells
  feed a scalar* = a **global selection mask**. Selection is inherently binary in spirit.
- Implication: weight parameterization must preserve selection semantics. Free continuous
  weights (plain CNN/linear kernel) are REJECTED — they dissolve the choice into a weighted
  average. Use binary / soft-binary (a learned selection), not an unconstrained encoder.
- Leaning (not final): **soft-binary via `sigmoid(w/T)` with temperature T annealed down**
  (soft selection → hard choice; fits random-init→chaotic→self-organize), or
  **straight-through binary** for hard {0,1} from the start.

Open questions to firm up during Discovery (do not assume):
- **Binary mask learning:** confirm soft-binary-anneal vs. STE (both realize "learned
  selection"). Gumbel-softmax also possible. Free-continuous ruled out (see principle above).
- **Target distribution:** the desired proportions of stationary/cyclic/chaotic channels,
  and how "distribution match" is scored (loss on the classifier's population histogram?).
- **Per-scalar dynamics model:** what actually predicts each 1D scalar forward in time
  (or is the emulator only encode + classify + population-shaping at this stage?).
- **Spatial reconstruction (inverse):** how/whether a full spatial state is rebuilt from
  the scalars (the earlier "search for best state matching the predicted feature").
- Number of kernels K; spatial grid/resolution; input variables; time sampling & length
  (Data-First tensor schema).
- Measurable North Star metric + held-out evaluation protocol.

## Data-First schema (draft — ocean, user 2026-08-25)

- **Input:** 3D ocean state as a 4D tensor `field[time, vardepth, lat, lon]`.
  - `time` — time steps.
  - `vardepth` — channel axis = number of 2D layers = `var × depth` (each variable at
    each depth level is one 2D map).
  - `lat, lon` — spatial grid.
- **Encoding:** `K` global kernels; kernel_i is a full-domain selection mask →
  scalar `s_i(t)`. Per-time state → vector of `K` scalars → `K` scalar time series.
- **Open (to confirm before coding the encoder):**
  - Mask domain: does each kernel select over the FULL volume `(vardepth, lat, lon)`
    (one mask over everything → truly global pattern) or per-layer over `(lat, lon)`?
    Leaning full-volume ("entire size kernel"), but confirm.
  - Normalization of `s_i` (e.g. divide by #selected cells) so scalar magnitude reflects
    the field, not how many cells the mask happens to pick.
  - `K` (latent width / compression ratio) — TBD together.

## Target family distribution — a CURRICULUM (user 2026-08-25)

- Not a fixed target. Training schedule:
  1. **Start 0 / 0 / 100** (stationary / cyclic / chaotic) — matches random-binary init
     (all scalars chaotic at t=0).
  2. Hold until the emulator predicts a **reasonable ocean state**.
  3. **Then** study/refine the target proportion toward the **natural ocean signal mix**.
- Implication: the population-distribution target is a moving target driven by a curriculum;
  the "reasonable ocean state" gate needs a concrete metric (TBD).

## Validation strategy — SYNTHETIC SPATIO-TEMPORAL TESTBED (user 2026-08-25)

- **Lorenz dropped** (had no spatial field → only tested the temporal half). Replaced by a
  **synthetic spatio-temporal correlated dataset** that exercises the FULL pipeline
  (mask-encoder → scalars → 1D dynamics → classifier → self-organization).
- **Testbed:** generate spatio-temporally correlated dynamics with **2 pseudo-variables
  imitating SSH and SST** (sea-surface height & temperature). Both are surface fields →
  `var=2, depth=1 → vardepth=2`, so `field[time, 2, lat, lon]`, same schema as the ocean case.
- **Why it's a good testbed:** it is generated, so we control (and know) the ground truth —
  the true spatial patterns AND the true dynamical family of each. That lets us validate
  BOTH the encoder (do learned masks recover the injected patterns?) and the classifier
  (do assigned families match the injected ones?). SSH↔SST can be built correlated to mimic
  real cross-variable coupling.
- **Proposed generator (recommend, confirm):** modal superposition
  `field_v(t,x,y) = Σ_k a_{v,k}(t) · φ_k(x,y)`, where φ_k are spatial patterns and each
  temporal coefficient a_k(t) is assigned a KNOWN family — stationary / cyclic (sinusoid) /
  chaotic (e.g. logistic/Lorenz-driven/noise-AR). SSH & SST share some modes (coupling) plus
  own modes. Gives ground-truth patterns + families for free.
- Sequencing: **Milestone 0 = synthetic SSH/SST testbed, THEN ocean.**
- **Open params (decide together):** grid size (lat,lon); time length & sampling; number and
  mix of injected modes/families; SSH↔SST coupling strength; K kernels for the test.

### Confirmed Milestone-0 testbed parameters (user, 2026-08-25)

- Grid: **64 × 64** (lat × lon). Time length: **2000** steps. Variables: **2** (SSH, SST).
  → `field[2000, 2, 64, 64]`.
- Ground-truth modes: **10 total = 1 stationary / 3 cyclic / 6 chaotic** (chaos-dominant).
- `K` kernels: proposed **16** (> 10 modes, room to spare) — adjustable.
- Note: because injected truth is 1/3/6, the curriculum's "natural" ENDPOINT target for this
  testbed is known ≈ **1/3/6**; training still STARTS at 0/0/100 (random binary init). This
  gives a concrete success check for the self-organization loop.

## Learning mechanism — temporal-gap minimization (user 2026-08-25)

- Masks init random → all scalars chaotic (0/0/100 start). A **neural network** is introduced;
  training signal = **minimize each mode's gap between time neighbors**
  `L = Σ_i Σ_t d(s_i(t+1), s_i(t))`, with gradients optimizing the **selection kernels**.
- Effect: selects spatial patterns whose scalar varies slowly between steps → pulls modes from
  chaotic toward coherent / slowly-varying. This is the engine that moves off the all-chaotic init.

### CRITICAL caveat — degeneracy / collapse (agent flag, must address)

- Pure gap-minimization has trivial optima:
  1. **Constant collapse:** `s_i=const` (or empty mask `s_i≈0`) → zero gap. Drives modes to
     STATIONARY/zero — opposite of keeping 6 chaotic.
  2. **Mode collapse:** all K kernels converge to the SAME slowest pattern → K duplicates, not
     K distinct modes.
- Required anti-collapse constraints:
  - **Zero-mean + unit-variance** per scalar `s_i` (forbids constant/zero).
  - **Decorrelation / orthogonality** across the K scalars (forces K *distinct* modes).
- **Prior art:** this = **Slow Feature Analysis (SFA)** (minimize temporal derivative s.t. unit
  variance + decorrelation). Project's novelty = binary *selection* masks + learned classifier +
  population target. Reuse SFA's constraint structure. [Phase B.3 research to expand.]
- **Tension:** gap-minimization rewards slowness → unchecked it makes ALL modes stationary
  (~10/0/0), not 1/3/6. The classifier + population-distribution target must GOVERN how many
  kernels are pushed toward smoothness (ties to curriculum D-007).

### Open questions on the loss (to confirm — see decisions D-010)

- NN role = **(a) LOCKED** (user 2026-08-25): masks are the learnable params (differentiable
  selection-encoder = masked linear projection); no predictor. ⇒ pure **linear SFA w/ binary
  masks** → anti-collapse constraints are now MANDATORY.
- Anti-collapse = **whitening `Cov[s]=I`** (A2 CONFIRMED, D-011): unit variance + decorrelation;
  closes both zero/constant collapse and kernel duplication.
- Training = **online pairwise streaming** (A3, D-011); diversity of families expected to EMERGE
  from decorrelation + small slow subspace (not an explicit classifier gate).
- How is `d(...)` measured (L2/L1) and over what horizon (only t,t+1 or a window)?

## EMPIRICAL — first working v0 training runs (2026-08-26, Milestone 0)

Everything below is measured on the synthetic testbed (`field[2000,2,64,64]`, 10 hidden modes
1/3/6, K=16), via `src/probes/evaluate.py`. Runs live in `.tmps/runs/`.

### F-1. The testbed does separate the families (sanity check passed)

Per-mode lag-1 gap variance `var(Δa)/var(a)`: stationary 0.011 | cyclic 0.011–0.021 |
chaotic 0.066–0.201. So chaotic modes are 6–18× "faster" than slow ones — a slowness objective
CAN discriminate here. (Was worth checking: at this sampling rate every mode has ρ₁ > 0.9.)

### F-2. Three ways the slowness+whitening objective collapses (all observed, all fixed)

1. **Soft penalty `‖Cov[s]−I‖_F²` → masks go to zero.** The gap term is quadratic in mask scale
   while the penalty is bounded by K, so shrinking always wins. Measured: `var→2.5e-6`,
   `white→16.0 = ‖0−I‖_F²`.
2. **Detached ZCA whitening → all K channels become ONE channel.** A detached whitening matrix is
   not a constraint: duplicate channels leave near-zero eigenvalues that the eps floor never
   re-inflates, and a zero-amplitude direction contributes nothing to the gap. Measured:
   effective rank **1.00/16**, all 16 channels identical, all correlating 0.86 with the same mode.
3. **Detached per-channel std → masks go to zero again.** Scale-invariance must hold in the
   GRADIENT, not just in value; autograd treats a detached std as a constant. Measured:
   `var` 1e6 → 1e-40 within 150 steps.

**Fix (D-017):** write both terms so they are scale-invariant to autograd —
`L = mean_i var(Δs_i)/var(s_i) + λ·mean_{i≠j} corr_ij²`, i.e. SFA's Rayleigh quotient plus a
differentiable decorrelation penalty. No matrix inverse, no detach. Stable; nothing to game.

### F-3. What the working v0 actually learns (K=16, 5000 steps)

- Masks become **small, well-separated regional sensors** (5–400 cells, mask IoU 0.17) — the
  D-013 picture, and NOT imposed: it is what decorrelation selects.
- Channels **latch onto individual hidden modes**, unsupervised: |corr| = 1.00 (m0), 0.99 (m2),
  0.94 (m6), 0.83 (m3), 0.81 (m1). Mode discovery works.
- A real **slow↔fast spread emerges across channels**: per-channel gap1 spans 0.0057 → 0.178 (31×).

### F-4. ⚠ Non-negative masks cannot decorrelate (structural limit on D-005 + D-011)

Decorrelation saturates: mean off-diag corr² plateaus at **0.244** (mean |corr| ≈ 0.49) and
effective rank at **4.08/16** — 16 channels crowd onto ~4 signals. Raising λ_white 1 → 20 → 100
→ 500 changes NOTHING (0.244 at every value), so it is not a tuning problem.

**Cause (general form).** Channel correlation expands as
`corr(s_i,s_j) ∝ Σ_c Σ_c' m_i(c)·m_j(c')·Cov(x_c,x_c')`. With non-negative masks, `m_i,m_j ≥ 0`;
if the field's cell-to-cell covariance is ALSO non-negative, every term is ≥ 0 and the sum cannot
approach zero for ANY choice of cells. Measured on this field: **100.0% of cell pairs are
positively correlated, mean +0.543, none below −0.2.** A non-negative mask can *avoid* a shared
signal (don't select there) but can never *cancel* one. Concretely, channels 0 and 2 correlate
+1.000 and decorrelating them requires `ch0 − 0.94·ch2` — a subtraction, unreachable with 0/1.

**⚠ CAVEAT (2026-08-26, corrects the first framing): this is largely a GENERATOR ARTIFACT.**
Every spatial pattern in `src/data/synthetic.py` is a strictly positive Gaussian bump
(`_gauss_bump`, `_equatorial_band`), so `field = Σ_k a_k(t)·φ_k(x)` with `φ_k ≥ 0` has
non-negative spatial covariance BY CONSTRUCTION. Real ocean anomaly fields do not: EOFs have
positive and negative lobes and teleconnections give genuinely anti-correlated regions. On a
sign-varying field a non-negative mask CAN decorrelate, by selecting a region anti-correlated with
what another kernel sees. ⇒ the first-listed remedy (revise D-005 to allow signed masks) is not
the right lever; fixing the testbed's spatial patterns is cheaper and more faithful.

Ablation that isolated the effect — `sigmoid` vs `tanh` masks (signed = D-005-rejected, diagnostic
only):

| masks | off-diag corr² | effective rank | balanced recon R² |
|---|---|---|---|
| non-negative `sigmoid` (D-005) | 0.244 (saturated) | 4.08 / 16 | 0.863 |
| signed `tanh` (diagnostic) | **0.070** | **8.57 / 16** | 0.877 |

⇒ **`Cov[s] = I` (D-011) is unreachable with non-negative masks ON A FIELD WHOSE SPATIAL
COVARIANCE IS EVERYWHERE POSITIVE.** On such a field the whitening constraint is a soft diversity
pressure, not something that can be satisfied. The fix is to make the field realistic (sign-varying
spatial patterns), not necessarily to give the masks a sign.

### F-5. ⚠ Minimizing mean slowness actively DESTROYS the chaotic content

Per-mode ensemble R² (regress each hidden mode's amplitude on all K channels), trained vs the
UNTRAINED multiscale masks:

| modes | untrained masks | after slowness training |
|---|---|---|
| m0–m3 (stationary + cyclic) | 1.00 | 1.00 |
| m4, m5, m7, m8, m9 (chaotic) | 0.99–1.00 | **0.01–0.03** |

Random multi-scale masks encode ALL ten modes nearly perfectly; slowness training throws the fast
ones away. Balanced reconstruction R² therefore does not improve either (0.863 trained vs 0.881
untrained). Emergent family population is **0 stationary / 14 cyclic / 2 chaotic** vs the 1/3/6
target.

⇒ The **D-011 "emergent governor" hypothesis does not fire**: decorrelation does not lock later
kernels out of the slow subspace, because a soft penalty lets 16 channels sit on ~4 slow modes at
|corr| ≈ 0.5 more cheaply than moving to fast content. Mean-slowness minimization is
*anti-correlated* with both the 1/3/6 population target and with reconstruction coverage. Getting
the chaotic third of the population requires an explicit mechanism — see decisions D-018.

### F-6. ⚠⚠ THE KERNELS FLED TO THE DEAD CORNERS — visible only in the figures

Found by looking at `figs/masks_on_energy.png`, NOT by any metric (see the standing rule: always
visualize, never judge from scalars).

Plotting the learned masks over the field's per-cell temporal std shows **the kernels sit almost
entirely on the lowest-energy cells — the right and bottom domain edges — while the high-energy
structures where the dynamics actually live are left un-sampled** (only ch4/ch5, and partly
ch1/ch11, land on real energetic patches).

**Cause:** the objective `L_slow = mean_i var(Δs_i)/var(s_i)` is a RATIO, hence indifferent to
signal amplitude. In a dead corner the field is just the far tail of the big slow mode: tiny
amplitude, but almost purely slow → a near-perfect ratio. A high-energy region carries a MIX of
slow and fast content → worse ratio. So the scale-invariance that was required to stop the mask
collapse (F-2/D-017) simultaneously **rewards looking where nothing happens.**

This single mechanism explains the rest of the results at once:
- why 16 channels duplicate ~5 signals (F-4): they all crowd into the same quiet corner reading
  the same slow tail — `figs/features.png` shows ch0/2/3/12 identical, ch6/7/9/10/13/14 identical;
- why the chaotic modes are discarded (F-5): they live in the energetic patches that get avoided;
- why balanced reconstruction R² is WORSE than untrained masks (0.863 vs 0.881).

⇒ **The objective needs an amplitude/coverage term** — slowness alone, made scale-free, points the
kernels away from the dynamics. The existing `lambda_var` hinge does not do this: it only forbids a
literally dead mask and read `hinge = 0.0000` for the entire run. Candidates: weight each channel's
slowness by the field variance it captures; add a reconstruction/coverage term; or require each
mask to hold a minimum share of total field energy.

### F-7. Energy floor + coverage term fix F-5 and F-6 (D-019) — measured

Ablation at 5000 steps, K=16, gradient-matched weights. `e/e_ref` = per-channel energy density
relative to the field's mean per-cell variance (below 1 = reading quieter-than-average cells).

| run | min e/e_ref | offcorr2 | eff rank | mask IoU | balanced R² | chaotic-mode ens R² |
|---|---|---|---|---|---|---|
| baseline (slow + decorr only) | **0.05** | 0.268 | 3.77 | 0.170 | 0.861 | **0.01–0.03** |
| + energy only (λ=1) | 0.93 | 0.301 | 3.43 | 0.208 | 0.864 | — |
| + recon only (λ=1) | 0.21 | **0.972** | 1.07 | 0.315 | 0.881 | — |
| **+ both, gradient-matched (λ_e=0.2, λ_r=0.3)** | **0.82** | **0.259** | **3.89** | **0.111** | **0.880** | **0.79–1.00** |
| + both, λ_slow=10 | 0.86 | 0.347 | 2.95 | 0.193 | 0.879 | — |
| untrained multiscale masks (reference) | — | 0.99 | 1.03 | — | 0.881 | 0.99–1.00 |

Readings:
- The **energy floor alone** cures the dead-corner drift completely (min e/e_ref 0.05 → 0.93) but
  does NOT cure duplication — all kernels simply crowd onto the same energetic region.
- The **coverage term alone at λ=1 anchors the masks at init** (offcorr2 stays 0.97, eff rank 1.07,
  R² identical to untrained) — it is a "don't move" force unless weighted by gradient norm.
- **Together, gradient-matched, both failures are fixed:** kernels sit on signal, all ten hidden
  modes are encoded again, reconstruction stops degrading, and mask overlap drops by a third — with
  the 31× slowness spread intact.
- **`lambda_slow=10` is actively harmful** (eff rank 2.95, spread 2.3×). The slowness term is the
  one that collapses diversity; it belongs in the loss as a weak tiebreak, not a driver.
- **Duplication survives** (eff rank ~4 of 16, and `figs/features.png` still shows ch6/9/10/13/14
  as one signal, ch1/3/8/11 as another). This is the F-4 non-negativity ceiling and no
  energy/coverage term addresses it.

**Methodological note that cost several runs:** weight multi-term losses by matching GRADIENT
norms, not loss values. Here the values and the gradients ranked the terms in OPPOSITE order
(l_recon was the smallest value, 0.003, and the largest gradient, 8.6× l_white's).

### F-8. What the FIELD visualization shows (first look at the testbed itself, 2026-08-27)

Built `src/probes/plots_field.py` (snapshots + hidden-mode answer key + kernel-activation map +
`field.gif` with kernel outlines pulsing at their instantaneous activation). Four things are
visible that no scalar metric had reported:

1. **SSH ≈ SST — the second variable is nearly redundant.** Cell-wise `corr(SSH_c, SST_c)` = 0.86
   on average; domain-mean series correlate **0.981**. The intended decoupling (`sst_lag=3`,
   `sst_blur_sigma=2`, weights 0.8/0.5) is far too weak against timescales of 60-300 steps, so
   `V=2` costs 2x compute for almost no extra information. Cross-variable structure is a stated
   goal of the testbed (D-009) and is effectively absent.
2. **Every spatial pattern is a positive blob or band** (`field_modes.png`) — direct visual
   confirmation of F-4's arithmetic. m0 is a horizontal band spanning all x; m1-m3 are large
   overlapping blobs; m4-m9 are tiny dots (sigma 2-4 cells). Nothing has a negative lobe.
3. **The domain has DEAD BORDERS.** The generator places modes with margins (`uniform(12,H-12)`
   for cyclic, `(6,H-6)` for chaotic), so a border ring of ~25% of the domain has near-zero
   temporal variance. That is exactly the region the kernels fled into in F-6 — the trap was built
   into the testbed. Real ocean domains have no such dead frame.
4. **Chaotic modes are tiny in space AND intermittent in time** — small dots whose standardized
   amplitude sits near zero between rare large excursions. They carry very little of the field's
   spatial energy, which is why energy-weighted R² (0.99) hides their loss while balanced R² (0.88)
   and the per-mode ensemble R² expose it.

**Kernel-activation readout (same run, D-020 defaults):**
- Only **24.8% of the domain is covered by any kernel**; up to 5 kernels stack on one cell.
- **6 of 16 kernels contribute ~nothing unique** (drop-one balanced-R² loss < 1e-4), and the
  largest unique contribution is only **0.005** — because the channels are so redundant that
  removing any one costs almost nothing. This is the effective-rank-3.6 result, seen per kernel.
- Encouraging for D-020: the kernels that DO matter span the whole ladder — ch7 (101 cells),
  ch1 (843), ch0 (1143), ch15 (**8 cells**), ch4 (310), ch14 (11). Scale diversity is buying
  genuinely complementary sensors, not just cosmetic variety.

**Note on a misleading first version of this figure:** ranking kernels by `std(s_i)` just re-plots
the size ladder (a 1143-cell footprint sums 140x more cells than an 8-cell one), and it made the
big kernels look uniformly dominant. Drop-one unique contribution is the scale-fair measure and
reverses much of that ranking.


### F-11. ⚠⚠ THE READOUT COULD NOT SAY "STATIONARY" — every population number was void (2026-08-31, encoder track)

> **Numbering note (merged from `architecture/01b_findings.md`, 2026-09-28):** F-11 and F-12 exist twice in
> this file — the encoder-track entries here and the module-2-track entries further down (see
> `memory/decisions.md` "Numbering collision"). F-13…F-17 are encoder-track only. **Read F-13 before trusting
> any population number in encoder F-11 or F-12:** those were measured under a definition of "stationary"
> the user later corrected.


`evaluate.py`'s heuristic labeller called a series **cyclic** whenever its single largest FFT bin
held more than `peak_thr` (0.10) of the power. A red / memory-dominated spectrum always piles its
power in the LOWEST resolvable bin, so a slow OU process was reported as "a clean oscillation of
period T/2". The proof was sitting in every eval.json we ever wrote: applied to the GENERATOR'S OWN
stationary amplitude series, the labeller returned `cyclic` (`truth[0].label`, accuracy 0.7/1.0).
The label "stationary" was therefore unreachable, and every `population.stationary = 0` we recorded
(D-018, F-5, the 2026-08-29 module-2 entry) measured the probe, not the encoder.

**Fixed** in the new `src/probes/family.py` — one module, imported by everything that needs a family
label, so the two cannot drift again. The decisive statistic is LINE WIDTH, not peak height:
`line_frac` = power within ±2 bins of the peak / total. Truth calibration (seed 0):

| | line_frac | tau_e |
|---|---|---|
| 3 cyclic modes | 0.94 / 0.95 / 0.94 | 12-57 |
| 1 stationary mode | 0.67 | 106 |
| 6 chaotic modes | 0.10-0.46 | 3-7 |

Rule: cyclic if `line_frac > 0.80` AND `k_pk >= 4` (>=4 whole cycles observed, so a red slope at
bin 1-3 cannot qualify); else stationary if `tau_e >= 20`; else chaotic. Both thresholds sit
mid-gap in a >3x margin. **Result: 10/10 against the hidden truth on seeds 0-4** (was 7/10).

**What re-labelling the existing runs revealed — the sign was BACKWARDS.** The K sweep
`.tmps/runs/20260831_0736*`, re-scored:

| K | old population (s/c/ch) | new population |
|---|---|---|
| 4 | 0/4/0 | 3/0/1 |
| 8 | 0/8/0 | 4/2/2 |
| 16 | 0/13/3 | 12/0/4 |
| 32 | 0/30/2 | 22/3/7 |
| 64 | 0/?/? | 36/10/18 |

The encoder was never failing to capture slow content — it produces almost NOTHING BUT slow content
(12 of 16 channels red at K=16, and ZERO clean cyclic ones). That is F-5's mean-slowness pathology,
hidden for five sessions behind a labeller that shouted "cyclic" at every red spectrum. The
stationary mode itself was always well captured: `mode_recovery.m0(stationary).max_abs_corr` = 0.92
(K=4), 0.62 (K=8), 0.93 (K=16), 0.82 (K=32), ensemble R² ~1.00 throughout.

**Lesson (third instance of the same one, after F-6 — the figures caught the dead corners — and F-9 —
`var_ratio` caught the mean-collapse ranking):** a metric that CANNOT emit an outcome
is worse than a missing metric, because it reads as evidence against the model. Validate a readout
against known ground truth BEFORE using it to judge a model. `labeller_truth_accuracy` was in the
eval output the whole time at 0.7 (for five sessions) and nobody — me included — treated it as a
blocker. `src/probes/validate_labeller.py` now makes "validate the readout against ground truth first"
executable.

### F-12. K is NOT the lever on the family population (2026-08-31, encoder track)

Measured across K=4..64 (the re-scored table in F-11; read those populations as slow/cyclic/fast —
see F-13): slow-dominated at EVERY K, cyclic channels essentially absent, and
the proportions barely move. Cause: `L_slow` is ONE objective shared by all K channels, so they
compete for the same globally-slowest content and each extra channel buys a near-duplicate of the
dominant regime rather than a new timescale. Confirms the user's read that "deepening the kernel"
raises slow-mode visibility somewhat but cannot reach every dynamical family.

The fix is to ASSIGN the timescale per channel — D-024's spectral band ladder, exactly the move
D-020 made for mask size. Measured effect at K=16 (5000 steps, seed 0, everything else equal):

| | baseline | + band/line | + L_mem |
|---|---|---|---|
| population (s/c/ch) | 10/1/5 | 6/1/9 | 5/1/10 |
| role obedience | n/a | 0.75 | 0.69 |
| slow rung vs true stationary mode | (unassigned) | corr 0.50 to a CHAOTIC mode, tau_e 36 | **corr 0.92, tau_e 87** (truth 106) |
| worst hidden-mode maxcorr | 0.25 | 0.65 | 0.61 |
| balanced recon R² | 0.9891 | 0.9891 | 0.9891 |

The fast rungs obey almost perfectly (gap1 0.10-0.13 vs the truth's chaotic 0.07-0.20; the
baseline's "chaotic" channels sat at 0.02-0.03, i.e. not actually chaotic), and no hidden mode is
left unseen (baseline abandoned two chaotic modes at |corr| 0.25). `figs/features.png` shows the
bank stratified slow -> cyclic -> fast down the channel index; the baseline's families are
interleaved in no order.

**Still open:** the CYCLIC rungs mostly fail — 1 of 5 reaches a clean line (`line_frac` 0.92, the
rest 0.52-0.60). A non-negative regional mask sums everything under its footprint and the
basin-scale slow content leaks into every region, so a channel cannot cancel its own red
background: F-4's non-negativity limit resurfacing in the time domain. Also the second slow rung
fails, and it is the LARGEST footprint on the size ladder (the D-021 `L_energy`-vs-biggest-rung
conflict) — so not aligning the size and timescale ladders is the cheap next test.


### F-13. STATIONARY MEANS CONSTANT — my F-11/F-12 taxonomy was wrong, and so were its populations (2026-09-01, encoder track)

**User's correction:** *"when we decided 'stationary signal', I referred that some constant signal.
Actually, your injected mode 0 in pseudo SSH-SST field, it is not stationary for me. it should be
more flat."* Also: the baseline population 10/1/5 I reported is not right, and several channels I
labelled stationary are visibly cyclic.

**Both halves check out, and the root cause is in the GENERATOR.** `_ou_series` ends with
`_standardize`, so the "stationary" mode was emitted at UNIT VARIANCE — exactly as much temporal
energy as the sinusoids and the Lorenz modes. Mode 0 was a slow wanderer with tau=200, never a
stationary signal, and no flat mode existed anywhere in the testbed for the encoder to find. My
F-11 labeller then defined stationary as "long memory, no line", which is a definition of slow red
noise, so it happily labelled 10 of 16 baseline channels stationary. Those numbers describe my
taxonomy, not the project's — **the 10/1/5 and 5/1/10 populations in F-12 should be read as
slow/cyclic/fast, not as stationary/cyclic/chaotic.**

**The definitions now in force (user's ruling, D-025):**
  * stationary = CONSTANT. Flat in time; lives in the field's time MEAN. Not "slow".
  * cyclic     = it moves and has memory. A clean oscillation, AND slow red drift (user's call:
                 a drift is not constant, so it cannot be stationary).
  * chaotic    = it moves fast and broadband.

**Generator rev3:** mode 0 is now `amp[t] = stationary_amp`, un-standardized. Verified: mode-0 amp
mean 1.000 / std 0.000e+00, and the field's time-mean map correlates **1.000** with the mode-0
pattern at rms 0.818 against a field std of 1.0 — so the constant is a strong, cleanly identifiable
static component. The labeller (flatness first, via `amp_ratio = std/|mean|` on the RAW series)
scores **10/10 on seeds 0-4** with population 1/3/6.

**Why the encoder could never have found it (the structural part).** A constant lives ENTIRELY in
the channel mean, and every term in the v0 loss uses the CENTERED channel `sc = s_t - mu`: slowness,
whitening, reconstruction and the energy floor are all computed on anomalies. Two terms go further
and actively forbid a flat channel — `l_var = relu(1 - std(s_i))^2` demands unit normalized std, and
`l_energy` demands the selected cells be energetic in TIME. A stationary channel was not merely
unrewarded, it was penalized.

**Measured on the rev3 field (K=16, 5000 steps, seed 0):**

| | baseline (no ladder) `20260901_080438` | ladder + `L_level` `20260901_080506` |
|---|---|---|
| population (s/c/ch) | 0/14/2 | **2/5/9** (target proportion 1.6/4.8/9.6) |
| role obedience | n/a | **1.00** |
| flattest channel `amp_ratio` | 0.778 — "still moving" | **0.023** (and a second at 0.049) |
| flat channels (amp_ratio < 0.05) | 0 | 2 |
| cyclic channels w/ line% >= 70 | (n/a under old labels) | 5 of 5 cyclic rungs |
| balanced recon R^2 | 0.7763 | 0.7765 |

So the baseline provably CANNOT capture a constant mode (nothing gets near flat) while the ladder
with `L_level` produces two genuinely static channels and puts every single channel in its assigned
family. Note the balanced recon R^2 is 0.78 on rev3 for both, down from 0.99 on rev2 — that is the
new field being harder (a large static component that anomaly-based decoding does not help with),
not a regression from the ladder.

**Null result worth not repeating:** de-aligning the size and timescale ladders
(`slow_size_frac=0.08`, run `20260901_080655`) halved the flat rungs' footprint 2590 -> 1318 cells
but did NOT improve their alignment with the constant mode's pattern (|corr| 0.27 either way), and
role obedience fell 1.00 -> 0.94. The flat channels' spatial diffuseness is not caused by the size
ladder. Default is back to 0.0.

**A figure was lying, again (cf. F-6).** `plots.py` standardized every channel before plotting,
i.e. divided out the std — exactly the quantity that makes a stationary channel stationary. A
channel whose fluctuation is 2% of its level was drawn looking as dynamic as a sinusoid, which is
how the ladder run first read as "ch0/ch1 are oscillating". Now scaled by RMS about zero, so a
constant channel draws as the flat line it is, and `amp_ratio` is printed on every lane. **Check what a
plot normalizes by before trusting it.**

**Remaining open:** the flat channels sit on *a* static structure but align with the injected
pattern at only |corr| 0.27 — the mask settles on the negative lobe / a diffuse static average
rather than isolating the band. Whether that matters depends on whether we want the observer to
isolate the pattern or merely to report a stable level.


### F-14. Classify on GLOBAL STRUCTURE; and the observer is not a mode-recovery device (2026-09-01, encoder track)

**User, slowing the work down to fix the frame:** *"for me ch02, 03, 04, 05 and 06 are cyclic. Even
09 and 11 have cyclic characteristics. Ok I am agree with that there are not 100 percent cyclic like
sin cosin signal. But we should classifier with there global strcutre not their small fluctuation
and drifts."* And: *"the stationary mode can live in the ouside of the signal. Hidden mode is just
hidden mode. If we can capture every hidden mode, it a perfect and ideal situdation, But it is
barely possible in complex dynamic system like ocean and atmosphere. So the Observer should decide
where we should see where we should focus on. That becomes learned kernel. if the kernel captures
stationary signal, it could be a single mode in dynamical system. But, it also can be a persist
phenomen or the combination of multi signals which shows barely varying value."*

**Two corrections, both of which invalidated something I had built.**

**(a) Structure, not residue.** My ACF-recurrence test read the RAW series, where fast noise and a
wandering baseline dominate the autocorrelation — so it answered a question about the residue and
called visibly oscillating channels non-cyclic. Replaced by an explicit 4-part decomposition
(D-026): `level` + `trend` (bins 1..3, i.e. fewer than 4 cycles in the record — at that resolution a
drift and a cycle are indistinguishable, so this IS the drift band) + `osc` (the dominant peak's
HALF-POWER band, which adapts: ~1 bin for a tone, wide for a quasi-periodic hump) + broadband
`residual`. Label = flat gate first, then `trend + osc` vs `residual`.

Design note: taking a plain argmax over all four shares, including the level, is wrong. `level^2 >
var` is merely `amp_ratio < 1`, so a channel fluctuating at 50% of its level would score
"stationary" — ch14 of run 20260901_080655 (amp_ratio 0.526, plainly broadband) is the
counter-example. The level GATES with a strict threshold; the dominance vote decides the shape of
what is left moving.

Measured with the new rule (post-hoc, no retraining — a label is a readout):

| | truth accuracy | population |
|---|---|---|
| rev3 truth modes | **10/10** | 1/3/6 |
| rev2 truth modes | 9/10 — and the "miss" is the OU mode 0 now labelled **cyclic** (trend share 0.63), which is D-025 working as ruled, not a bug | — |
| rev3 ladder `20260901_080506` | — | 2/5/9, **role obedience 1.00** |
| rev3 baseline `20260901_080438` | — | 0/11/5 |

On the specific channels: ch2-6 of the ladder run come out **cyclic** (osc share 0.70-0.93), matching
the user. ch9 and ch11 stay **chaotic** but with osc shares 0.41 and 0.32 against residual 0.59 and
0.67 — i.e. real oscillatory character, broadband still dominant. That is the honest reading of
"they have cyclic characteristics but are not 100% cyclic", and it keeps truth accuracy at 10/10;
pushing them to cyclic would require overriding the Lorenz-derived modes they track (m6).

**(b) The family belongs to the OBSERVABLE, not to a hidden mode.** The kernel's job is to decide
where to look. A flat channel may be a single static mode, a persistent phenomenon, or a combination
of varying signals whose sum barely moves — all three are valid stationary observables. So:
`mask_align_with_pattern` is DEMOTED to a diagnostic (I had been treating its low value, |corr| 0.27,
as a failure — wrong frame; I had introduced that metric the same day and had even run an experiment,
the de-aligned-ladder null result in F-13, to "fix" it), the 1/3/6 target is relabelled a DESIGN choice for the bank rather than
a recovery score, and `mode_recovery` is marked diagnostic in the probe output. See D-026 for the full
scoring consequences.

**New diagnostic that answers the user's "or a combination" case directly.** For the flattest
channel, project every hidden mode onto its footprint and compare the net fluctuation against the
independent-addition baseline `sqrt(sum sd_k^2)`:

  < 0.7  destructive interference beyond chance (a cancelling COMBINATION)
  ~ 1.0  incoherent addition (a genuinely quiet footprint / persistent phenomenon)
  > 1.3  the modes reinforce

First measurement, ladder run ch1: 3 hidden modes reach the footprint, index **1.01** -> this
stationary observer is the *persistent-region* case, not the cancellation case. Note the earlier
version of this metric normalized by the plain SUM of the parts and read 0.634, which looks like
partial cancellation but is almost exactly what chance predicts for 3 parts (0.577) -- the wrong
baseline would have invented a finding.

### F-15. The fast rungs were ambiguous because no loss term opposed it (encoder track; D-027 work, from 2026-09-10)

Run `20260901_080506`: 7 of 16 channels had structure share in 0.35–0.65, all fast rungs (0.33–0.44).
`line_cap` (±2-bin line fraction ≤ 0.75) was silent on all nine fast rungs (0.15–0.45). The hidden
chaotic modes score 0.17–0.35 and two fast channels (ch7, ch10) reached 0.16, so broadband channels
were achievable. **The objective and the readout must measure the same QUANTITY, not only agree on a
threshold** — the F-13 lesson in a harder form.

Stationary side, same run: a random mask of the same 2590-cell footprint gives amp_ratio ≈ 66, the
learned one 0.023–0.049. Flatness is a real achievement, not free averaging (my hypothesis, refuted).
It is hard because the static component is spatially zero-mean (time-mean map rms 0.795, global mean
0.0000), so a non-negative mask must align with one signed lobe. It is also fragile across seeds:
baseline flat channels per seed 2/0/1/0/1 — `flat_target 0.95` sits exactly on the 0.05 cut.

### F-16. A good ambiguity count hid a collapse of the cyclic timescales (encoder track; D-027 work)

`L_struct` on all rungs scored 3.0 ambiguous channels (baseline 6.8), but the kernel/signal figures
showed every cyclic rung on the same cycle: dominant periods 61/61/61/61/61 on seed 0 vs the
baseline's 286/143/61/61/61. `trend+osc` credits any clean peak, and period 60 is the cleanest line.
The `slowness_spread` drop (88 → 16) flagged earlier was this collapse. Fast-only (D-027) cuts
ambiguity to 1.2 but still averages only 1.4 distinct periods (baseline 2.4); seed 0 still collapses.
Cause open (at the time — resolved by F-17). **A metric about one failure mode says nothing about the
others — look at the figure.**

### F-17. The cyclic collapse was my own averaging bug, decided in the first ~20 steps (2026-09-27, encoder track)

Fast-only averaged `L_line` over the 5 cyclic rungs instead of 14, a 2.8× stronger "be a clean line"
per cyclic rung. All cyclic rungs start on the period-60 cycle (the cleanest line), and the stronger
hinge beat `L_band` before step 20; once on a clean line a rung never left (in-band 0.22 for 4,980
steps). Fixed by keeping the average over all dynamic rungs: distinct cyclic periods 1.4 → 2.6. Lesson:
**restricting a term to fewer rungs silently changes its per-rung strength when it averages; select
rungs, keep the denominator.** And the early steps decide: a hinge that is active for 5,000 steps
without moving the rung means a lock-in, not a weak weight.

## Research (PENDING - Phase B.3)

- Research has not started. It should follow Blueprint Discovery once the North Star,
  Integrations, and Source of Truth are known (see ./memory/task_plan.md).
- When done: search relevant repositories, documentation, and prior art; reuse existing
  patterns; record findings and references here.

## Module 2 (latent predictor) findings — F-9, F-10, F-10b, F-11/F-12 (module-2 track), F-18

> **Merged 2026-09-28 from `architecture/03b_module2_findings.md`** (latent-predictor worktree; a
> branch record written because `memory/` was gitignored and absent from the branch — "on merge,
> reconcile with `memory/findings.md` rather than replacing it"). This file's entries are the base;
> every additional fact of 03b is folded in. `architecture/` is retired; this file is the only home.
> Decisions: `memory/decisions.md` D-022, D-023, D-024 (module-2), D-028. SOP: `memory/sop/03_latent_predictor.md`.
>
> F-9 … F-12: measured on the Aug 27 module-1 run (K=16, generator rev2), train `[0,1400)`,
> val `[1432,2000)`, 472 free-running launches, warmup 32, scored to lead 64.
> F-18: re-measured on the CURRENT encoder (`flat98fix_seed0`, 2026-09-27) — its numbers supersede
> F-9 … F-12's; their lessons all re-confirm.

### F-9. ⚠⚠ Long-lead MSE skill RANKS THE FORECASTERS BACKWARDS — mean-collapse wins on MSE (2026-08-29)

First build of module 2 (D-022). Same frozen encoder (`.tmps/runs/20260827_151413`, generator
rev2), train `[0,1400)` / val `[1432,2000)`, 472 free-running launches, warmup 32, scored to lead 64.

| predictor | params (all 16 ch) | params/ch | skill h1 | skill h4 | skill h64 | var ratio @h64 | mean accH |
|---|---|---|---|---|---|---|---|
| GRU, learned hidden state | 14 864 | 929 | +0.420 | +0.564 | +0.246 | **0.89** | 42.3 |
| MLP, 16-step delay window | 4 624 | 289 | +0.398 | +0.561 | +0.393 | 0.65 | 42.2 |
| learned linear AR(16) | 272 | 17 | +0.356 | +0.485 | **+0.425** | **0.18** | 38.7 |
| GRU, 1-step loss only | 14 864 | 929 | +0.475 | +0.593 | **−33.07** | **16.12** | 39.5 |
| closed-form AR(8) | — | — | ~~+0.556~~ **+0.438** | — | ~~+0.300~~ **+0.295** | — | — |

> ⚠ **SUPERSEDED IN PART BY F-10.** The AR row above was computed with a leaking baseline
> (it read the target as its own lag-1 input). Corrected values are shown struck-through
> → fixed. Point 3 below is FALSE as written; see F-10.
> (03b's copy of F-9 instead removed the AR row and points 3–4 altogether: "An earlier version of F-9
> also claimed 'closed-form AR(8) beats every trained model at 1 step (+0.556)'. That claim is FALSE —
> see F-10. The AR numbers have been removed from this table for that reason; the corrected values
> live in F-10." Kept here struck-through for the record. 03b states the teacher-forced gain in
> point 2 as "~+0.05".)

1. **The ranking inverts.** The linear model has the BEST h64 MSE skill and is the WORST
   forecaster: its rollout amplitude decays to ~1e-5 of truth by lead 64 (`pred_rollout_stats.png`,
   right panel, log axis). It "wins" by becoming the climatological mean and would hand module 3 a
   flat field. The GRU has the WORST h64 MSE skill and the only honest amplitude (0.89).
   → **Rank by `var_ratio` first, then by skill.** Recorded in D-022 and SOP 03.
   → This is the same lesson as F-6 (kernels fleeing to dead corners) in a new place: a
   scale-free/ratio metric is indifferent to amplitude, and something must pin the amplitude down.
2. **The rollout curriculum is load-bearing.** Teacher-forced-only training buys +0.055 at h1 and
   makes the rollout DIVERGE (skill −33, amplitude 16x). Collapse and divergence sit either side of
   the curriculum. Never train this module on 1-step loss alone.
3. **Closed-form AR(8) beats every trained model at 1 step** (+0.556 vs +0.420). Not a bug: the
   trained models trade 1-step accuracy for rollout stability (the H=1 GRU recovers to +0.475), and
   these channels are largely LINEAR-predictable at 1 step *because module 1 optimizes slowness*.
   The encoder's own objective is what makes the linear baseline strong — worth revisiting if the
   encoder is ever trained jointly with the predictor.
4. **The hidden state earns its keep exactly where the dynamics are nonlinear.** On the two
   chaotic-labelled channels: GRU skill h1 +0.875 / +0.909 vs closed-form AR +0.135 / +0.160 (5x).
   On the smooth cyclic channels AR wins. The aggregate hides this because 14 of 16 channels carry
   the "cyclic" label.
5. **No persistence mimicry anywhere** (`pred_scatter.png`: no channel is a flat band at dz≈0), but
   the cyclic channels show clear increment UNDER-DISPERSION — the MSE-optimal shrinkage. Several
   channels (3, 5, 11, 14) also show a small constant negative drift in the predicted increment.

**Caveat:** every per-family number above uses the module-1 heuristic labeller, whose population is
0/14/2 against the 1/3/6 target. A real classifier now blocks the interpretation of module 2 as
well as module 1. *(Obsolete on the current encoder — see F-18.)*

### F-10. ⚠⚠ The AR baseline was LEAKING the target — every "AR wins" claim in F-9 is void (2026-08-31)

`ar_rollout` in `src/models/predictor.py` built each launch's history as `z[t-p+1 : t+1]`, which
**includes `z[t]` — the very value being predicted** — as its lag-1 input. With `a_0 ≈ 1` on the
smooth channels the baseline scored near-perfectly by copying the answer.

Fixed to `z[t-p : t]`. What it changes:

| quantity | leaked (reported in F-9) | corrected |
|---|---|---|
| AR(8) mean skill h1 | **+0.556** | **+0.438** |
| AR(8) mean skill h64 | +0.300 | +0.295 |
| `corr(AR skill, GRU skill)` over 16 ch | **−0.972** | **+0.964** |
| oracle `mean max(AR, GRU)` at h1 | +0.731 | +0.459 |

**Both conclusions built on it are now dead:**

1. F-9 point 3, "closed-form AR(8) beats every trained model at 1 step", is **false**. Corrected:
   AR +0.438 / GRU +0.420 / MLP +0.398 — a 0.04 spread, i.e. effectively tied. The elaborate
   explanation offered for the gap (rollout curriculum trading away 1-step accuracy; the encoder's
   slowness objective handing over near-linear channels) was explaining an artifact. The curriculum
   trade-off is still real and separately measured (the H=1 ablation), but it is not worth 0.14.
2. The **complementarity** claim — that AR and the nets win on disjoint channels and a hybrid would
   capture both — inverts completely. At r = **+0.964** they win and lose on the SAME channels:
   per-channel difficulty dominates, architecture barely matters. Per-channel oracle selection buys
   only +0.021 over AR alone. The h1 leaderboard is now essentially "how predictable is this
   channel", not "which model". **This holds at h1 only and does NOT extend to rollout**, where the
   same correlation falls to +0.57 and the linear model collapses on every channel (F-12, module-2).

**How it was caught:** by an assertion, not by inspection. `arch=ar_mlp` is initialized from the
closed-form AR solution with a zero-initialized residual branch, so it MUST reproduce the baseline's
score at step 0. The trainer now checks exactly that and refuses to run on mismatch. It fired on the
first attempt (+0.438 vs +0.556) and the investigation found the baseline wrong, not the anchor.
The leak had survived ~10 training runs and four figure reviews because a too-good baseline looks
like a strong baseline, and nothing else cross-checked it.

**Standing lesson:** a baseline needs a correctness test as much as the model does. Where two code
paths must agree by construction, assert it in code — visual inspection will not catch an index
that is off by one in the favourable direction.

#### F-10b. The AR-anchored residual predictor was built and tested — it buys ~nothing

Built `arch=ar_mlp` (`ARResidualPredictor`): `s_hat(t+1) = AR(8)·w(t) + g_theta(w(t))`, anchor
initialized at the closed-form solution, residual branch zero-initialized, anchor frozen by default
(`ar_anchor: frozen` — for a 1-step loss, regressing the AR residual and jointly fitting the sum are
the same optimization; they differ only under the rollout loss, where the AR term at lead *h*
consumes values the net helped produce, so the residual target cannot be precomputed).

All on the same frozen encoder / split, corrected AR baseline:

| model | trained on | h1 | h4 | h64 | var @h64 |
|---|---|---|---|---|---|
| closed-form AR(8) | 1-step least squares | **+0.438** | — | +0.295 | — |
| ar_mlp | 1-step residual only (H=1) | **+0.443** | +0.544 | −8.89 | 17.7 |
| ar_mlp | rollout curriculum (H→16) | +0.431 | +0.542 | +0.185 | 0.79 |
| GRU | rollout curriculum | +0.420 | **+0.564** | +0.246 | **0.89** |
| MLP (current default) | rollout curriculum | +0.398 | +0.561 | **+0.393** | 0.65 |

- A nonlinear net regressing the AR residual **directly** (H=1) gains **+0.005** at h1 — about 1%
  relative — and diverges at rollout (var ratio 17.7). The nonlinearity has almost nothing to add
  on top of a linear AR on these channels.
- Under the rollout curriculum the hybrid ends up slightly BELOW its own anchor at h1
  (+0.431 vs +0.438), which is expected and not a bug: the anchor is 1-step-optimal by
  construction, and the training objective is 16-step rollout error, so any movement trades h1 away.
- **Conclusion (h1 only — SUPERSEDED at rollout by F-12 / amended D-024; the "encoder's objective"
  lever is withdrawn by the 2026-09-10 user directive): architecture is not the lever.** Every model
  tested spans 0.398–0.443 at h1, and
  they succeed and fail on the SAME channels (r = +0.96, F-10). What varies is per-channel
  predictability, which is set by MODULE 1 — the encoder decides what the channels contain. Further
  predictor-architecture search on this testbed is not worth the compute; the open levers are the
  encoder's objective and the classifier.
- Kept anyway: `ar_mlp` gives a *guaranteed floor* (it cannot start below the closed-form baseline,
  and the trainer asserts it), which is worth having as the default reference model even though the
  learned part adds little here.

### F-11. Family-specialized predictor experts buy nothing — they are NESTED in the general form (2026-08-31, module-2 track)

User proposal (2026-08-31): since the testbed has three dynamical families, give the predictor three
modules, one per family. Tested cheaply BEFORE building any gating, by training each specialized
form on every channel and taking the per-channel best on validation (an oracle upper bound — a real
gate could only do worse).

Forms built (`src/models/predictor.py`), each designed so its family's observed rollout failure is
impossible by construction:

- `ou`  — stationary expert: AR(1) with `phi = tanh(.)`, so `|phi| < 1`; cannot diverge, reverts to
  its own learned mean. 2 params/channel.
- `osc` — cyclic expert: AR(2) in POLAR form, `r = sigmoid(.)`, poles at radius `r < 1` and angle
  `th`; holds frequency, cannot blow up. 3 params/channel. (Free AR(2) coefficients would allow
  poles outside the unit circle; the polar parameterization is what buys the guarantee.)
- `mlp` / `gru` — the existing general nonlinear forms.

**Result: the specialized experts win ZERO of 16 channels at h1.**

| form | params/ch | mean skill h1 | mean skill h64 | var ratio @h64 | channels won |
|---|---|---|---|---|---|
| `ou` | 2 | **−0.007** | +0.301 | **0.19** | 0 |
| `osc` | 3 | **−0.065** | +0.245 | **0.00** | 0 |
| `mlp` | 289 | +0.398 | +0.393 | 0.65 | 7 |
| `gru` | 929 | **+0.420** | +0.246 | 0.89 | 9 |
| per-channel ORACLE | — | +0.430 | (+0.512, see below) | — | — |

- Specialization is worth **at most +0.009** at h1 (oracle +0.430 vs best single +0.420) — noise.
- `ou` scores exactly 0.000 on most channels, i.e. it converges to `phi → 1` and *is* persistence.
- The apparent h64 win for the specialized forms (oracle +0.512 vs mlp +0.393) is the **F-9 trap
  again**: `osc` has var ratio **0.00** and `ou` **0.19** — both have collapsed to the climatological
  mean, which is how they score well on long-lead MSE. The comparison figure now carries a third
  panel with the amplitude on a log axis precisely so this cannot be misread; `osc` sits on the
  1e-4 floor across nearly every channel while its h64 bars are among the tallest. (03b: this is
  the F-9 trap **for the third time** — see "the mean-collapse trap" below; the third panel lives in
  `compare_predictors.py`.)

**Why it fails, and why this is structural rather than a tuning problem:** AR(1) and polar AR(2) are
both LINEAR functions of the delay window, so they are strict SPECIAL CASES of the delay-window MLP,
which already contains them. Specialization can therefore only act as regularization — and with
~1400 training steps per channel against ~289 parameters there is no data shortage to regularize.
The general forms subsume the experts; there is nothing left for a gate to route.

**Consequence:** do NOT build the soft mixture-of-experts / learned gate. The premise it rests on is
measured false. *(The rest of this paragraph — "architecture is not a lever", "remaining lever is
MODULE 1" — was later corrected: at rollout architecture matters (F-12, module-2), and the encoder is
out of scope (user directive 2026-09-10). The no-gate conclusion stands.)*
Together with F-10b (all architectures within 0.045 at h1, r = +0.96 across
channels) the conclusion is now firm: **predictor architecture is not a lever on this testbed.**
The remaining levers are MODULE 1 (the encoder decides how predictable the channels are — it is
optimized for slowness, which is why everything is near-linear) and the classifier.

**Side benefit kept:** "which form wins each channel" is a dynamics-based grouping obtained for
free. It gives `chaotic(2) -> gru 2` and `cyclic(14) -> mlp 7, gru 7` — i.e. it does NOT reproduce a
3-family structure either, which is independent evidence that the 16 channels are not cleanly
partitioned into the generator's three families (consistent with D-013: masks are regional sensors,
so channels are MIXTURES of modes, not pure family members).

### Cross-cutting (module-2 track): the mean-collapse trap fired THREE times

F-9 (linear ablation), F-11 (`ou`), F-11 (`osc`). Each time, long-lead MSE ranked a collapsed model
first. **Any new model comparison in this project must ship the amplitude panel from the start**,
not add it after being fooled. `var_ratio` is not a diagnostic afterthought; it is the primary
ranking key at long lead. *(Count as of 2026-08-31; F-18 records it firing twice more on the current
encoder — linear AR(16) and closed-form AR(8) — five times in all.)*

### Cross-cutting caveat (module-2 track): the family labels are unreliable

Every per-family aggregate in F-9 … F-12 (module-2) uses the module-1 heuristic labeller, whose
population is **0 stationary / 14 cyclic / 2 chaotic** against the 1/3/6 target. A real classifier
now blocks the interpretation of module 2 as well as module 1. *(Obsolete on the current encoder:
F-18 measures 2 / 5 / 9 — "the recurring caveat 'the labeller is broken (0/14/2)' is obsolete".)*

### F-12. Architecture DOES matter at rollout; delay-window length trades accuracy for amplitude (2026-09-10, module-2 track)

**Amends D-024 (module-2) and qualifies F-10b/F-11.** Two results from 2026-09-10. The "architecture
is not a lever" conclusion was computed at **h1 only**. It does not survive iteration; both results
below are about the free-running regime, which is the one module 3 actually consumes.

**(a) The equivalence between forms is an h1 artifact.**

| correlation across the 16 channels | h1 | h64 |
|---|---|---|
| mlp vs linear AR(16) | +0.972 | **+0.572** |
| gru vs linear AR(16) | +0.960 | **+0.454** |

Channels with honest amplitude (var ratio 0.5–1.5) at h64: mlp **9/16** (mean h64 skill on them
+0.338), gru **8/16** (+0.273), linear AR(16) **0/16** (—). The linear model cannot carry a single channel through 64 free-running
steps with realistic variance — its headline h64 skill (+0.425) is entirely mean-collapse.
⇒ **Nonlinearity is required for the free run**, the regime module 3 consumes. Interchangeable one
step ahead, emphatically not interchangeable under iteration.

**(b) Delay-window length is a live knob** (prompted by "what if the cycle is longer than the
window?" — generator periods are 60/140/300, so a 16-step window sees 27% / 11% / **5.3%** of a
cycle). Swept with `warmup=144` fixed so all three share identical launches:

| window | params/ch | h1 | h64 | amplitude @h64 | acc horizon |
|---|---|---|---|---|---|
| 16 | 1 057 | **+0.413** | +0.325 | 0.47 | 38.9 |
| 64 | 4 129 | +0.367 | **+0.344** | 0.66 | **45.3** |
| 128 | 8 321 | +0.289 | +0.320 | **0.85** | 39.9 |

Monotone trade: longer window buys rollout amplitude (0.47 → 0.85), costs 1-step accuracy
(+0.413 → +0.289). It does NOT fix the accuracy horizon (~40 steps, below the shortest period of
60), so memory length is not the whole story. Operating point is OPEN — it depends on what module 3
needs (amplitude fidelity or short-lead accuracy) and is not answerable from module 2 alone.

**Mechanism (theory, consistent with the figures, not separately verified):** a window need not span
a cycle — AR(2) represents a 300-step oscillation exactly (poles at radius ≈1, angle 2π/300). But
over 5% of a cycle a sinusoid is nearly a straight line, so the frequency lives in a tiny curvature
competing with `obs_noise=0.05` and the other modes mixed into the channel. Frequency error Δθ then
gives phase drift growing as h·Δθ — which is what the cyclic rollouts show (right amplitude,
sliding phase).

**Process note:** this correction came from the user asking two questions I had not tested — "what
if the cycle is bigger than the window?" and "don't we need a neural net to match the free run?".
Both were right. My sweep of six architectures had held `window=16` fixed throughout, so a whole
dimension went unmeasured while I recorded a confident closure over it.

**SCOPE (user directive, 2026-09-10): the encoder is NOT to be modified.** An encoder-side proposal
(adding the never-implemented D-004/D-007 family-distribution term, since `L_slow` applies uniform
pressure toward slowness on every channel and cannot create family diversity) was raised and then
withdrawn by the user. It stands as an explanation of why the channels are near-linear; it is not a
work item. Earlier notes listing "module 1's objective" as a next lever are superseded.

### F-18. Module 2 re-measured on the CURRENT encoder: GRU dominates; forecast more realistic, NOT more accurate (2026-09-27)

F-9 … F-12 were measured on the Aug 27 encoder (`20260827_151413`). Module 1 has since been reshaped
(timescale ladder, spectral losses, family roles, F-17 `L_line` averaging fix, `flat_target 0.98`;
commits `5e5c129` … `1812638`). Predictor code unchanged, re-run on `flat98fix_seed0` (copied into the
latent-predictor worktree's `.tmps/runs/fix98_s0*`; nothing written into the stationary-observer
session's run dir). Same predictor config as before: train `[0,1400)`, val `[1432,2000)`, 472
launches, warmup 32, horizon_max 16.

Labeller population on the new encoder: **2 stationary / 5 cyclic / 9 chaotic** (old: 0/14/2). The
family structure the module-2 docs kept caveating as "unreliable" is now there.

| predictor | h1 | h4 | h64 | amplitude @h64 | honest ch. | acc horizon |
|---|---|---|---|---|---|---|
| **GRU** | **+0.713** | **+0.746** | +0.264 | **0.80** | **15/16** | **40.0** |
| MLP (default at the time; D-023) | +0.667 | +0.692 | +0.241 | 0.57 | 9/16 | 35.3 |
| linear AR(16) | +0.588 | +0.560 | +0.443 | 0.14 | 2/16 | 32.4 |
| closed-form AR(8) | +0.684 | — | +0.386 | 0.14 | 3/16 | — |

- **GRU wins on every axis**; first trained model to beat closed-form AR at h1 since the F-10 fix.
  On the old encoder the GRU only *tied* the MLP at h1/h4 and the MLP led at h64. The MLP does not
  beat AR(8) at h1 (+0.667 vs +0.684); the GRU does (+0.713).
  **D-023's basis (a GRU/MLP tie) no longer holds** — surfaced to the user, default NOT changed.
  *(Later the same day the user decided: D-028 makes `gru` the default.)*
- **Collapse trap fired twice more** (linear AR(16) and closed-form AR(8), amplitude 0.14 each,
  posting the best h64 skills). I nearly reported "AR beats the NNs at h64" before checking AR's
  amplitude, which the trainer does not print. F-12a re-confirmed more strongly.
- Per family (GRU): cyclic horizon **62.6**, amplitude 0.89 (ch4/ch5 hold the oscillation at the
  right amplitude and period for ~4 cycles across the whole 536-step window, drifting only slowly in
  phase; ch2 and ch3 hold amplitude but run at the wrong period); chaotic h1 **+0.883**, horizon
  26.8, the free run stays on the attractor (same amplitude and texture) while decorrelating —
  correct chaotic behaviour; GRU wins all 9 chaotic channels at h1. Stationary h1 ≈ 0 (+0.050) is
  expected, not a failure: near-flat channels (`amp_ratio` 0.040 / 0.023 vs ~1.0 for the rest) whose
  small residual is magnified ~25–40× to unit variance by per-channel standardization — they *look*
  like they wander, but there is nothing predictable in it.
- Specialization oracle over 3 forms: +0.717 vs the GRU's +0.713 (+0.004). F-11 re-confirmed.
- Obsolete now: the "labeller is broken (0/14/2)" caveat throughout F-9 … F-12.
- NOT re-run on the new encoder: window sweep (F-12b), `ar_mlp`, `ou`/`osc`. The window trade-off in
  particular may have moved, since cyclic horizons now reach ~60 steps.
- **Consequences for earlier module-2 entries:** F-9 … F-12 describe the OLD encoder's channels.
  Their *lessons* (rank by amplitude before MSE; curriculum is load-bearing; nonlinearity needed at
  rollout; specialization is nested) all re-confirm here. Their *numbers* are superseded by this table.

**⚠ CORRECTION (same day, 2026-09-27) — "the channels are far more predictable" was WRONG.** The claim
compared skill-vs-persistence ACROSS encoders (h1 ~+0.40 old → ~+0.70 new). Skill's denominator is
each run's own persistence error, and on the new, faster channels persistence is a much weaker
opponent: 1-step persistence RMSE **0.154 old vs 0.269 new** (lag-1 autocorrelation 0.985 vs
0.964), same channel-std units. Skill rose because the yardstick got easier, not because the
forecast improved. **Skill vs persistence is not comparable across encoders.** Comparable metrics
(`src/probes/compare_over_lead.py`, figure `.tmps/forecast_old_vs_new.png`):

| | RMSE h1 | RMSE h16 | RMSE h64 | amplitude h64 (median) | honest ch. h64 | acc horizon (median) |
|---|---|---|---|---|---|---|
| old encoder + MLP (prev default) | 0.097 | 0.531 | 0.930 | 0.58 | 56% | 37 |
| old encoder + GRU | 0.094 | 0.570 | 1.068 | 0.67 | 50% | 46 |
| **new encoder + GRU (default)** | **0.091** | 0.570 | 1.029 | **0.77** | **94%** | 36 |

- **Accuracy — unchanged.** Absolute error and correlation-with-truth curves overlap across all
  three configurations over the whole 64-step lead; usefulness (correlation ≥ 0.5) ends at ~40 steps
  in every case, and the median accuracy horizon is no longer (36 vs 37 / 46).
- **Realism — clearly better.** The new default keeps an honest amplitude on **94%** of channels at
  h64 against ~50–56% before; its median amplitude holds at ~0.77 where the old MLP sinks to 0.58.
  The free run no longer collapses or blows up.
- Long-lead RMSE again rewards collapse: a realistic forecast that has decorrelated sits near
  RMSE √2 ≈ 1.41, a collapsed one at ≈1.0 — which is why the old MLP (amplitude 0.58) shows the
  lowest h64 RMSE. Read panel (a) beyond ~h40 with panels (c)/(d).
- **Unaffected:** everything *within* one encoder — GRU vs MLP vs linear vs AR share a
  denominator, so D-028 and "first trained model to beat closed-form AR at 1 step" stand.

One-line summary: **on the current encoder the forecast is more REALISTIC, not more ACCURATE.**

---

### F-19. Analog forecaster × 5 seeds: every observer SUFFICIENT for every cyclic mode; "blind seed" claim wrong (2026-09-28)

Probe `src/probes/analog_seed_test.py` (zero-parameter analog forecaster = config (1) of the memory theory, D-029),
encoders `flat98fix_seed{0..4}` (in `stationary-observer/.tmps/runs/`), trainer split train `[0,1400)` / val
`[1432,2000)`, 376 val launches (same for every sight: history window of 128 inside val), leads 1–64,
sight L ∈ {8,16,32,64,128}, k ∈ {1,10} analogs. Outputs: `.tmps/analog_seed/{results.json,summary.txt,
analog_seed_test.png}`. NB: each seed reseeds the GENERATOR too (hidden truth differs per seed), so this tests
sufficiency per seed, NOT observer reproducibility.

Sufficiency test = analogs matched in OBSERVER space, forecast = the analogs' future HIDDEN truth; corr with
the true mode at lead 16. Best single channel chosen on the 1st half of launches, scored on the 2nd half.

| seed | P61 single / joint | P143 single / joint | P286 single / joint | chaotic (mean) single / joint |
|---|---|---|---|---|
| 0 | +0.98 / +0.85 | +0.92 / −0.19 | +0.87 / +0.70 | +0.53 / +0.32 |
| 1 | +0.99 / +0.71 | +0.90 / +0.86 | +0.81 / +0.30 | +0.71 / +0.28 |
| 2 | +0.99 / +0.50 | +0.99 / +0.51 | +0.68 / +0.69 | +0.43 / +0.17 |
| 3 | +0.99 / +0.87 | +0.99 / +0.73 | +0.62 / +0.19 | +0.81 / +0.32 |
| 4 | +0.97 / +0.85 | +0.76 / +0.31 | +0.06 / +0.18 | +0.31 / +0.05 |

(L=16, k=1. At L=128 the in-sample best single channel reaches 0.94–1.00 on every cyclic mode of every seed.)

1. **Seeds 2 and 3 are NOT blind to the period-60 cycle** (+0.99). D-029's first evidence line (no cyclic rung
   has a dominant ~61-step period) was WRONG: a channel's dominant FFT period is not what it can see — the cycle
   is carried as secondary content by some channel. → Measure visibility by forecastability of the truth, not
   by labels/peaks. Same lesson as F-11 (encoder track): the READOUT, not the observer, was wrong.
2. **Sufficiency holds for the cyclic family on all 5 seeds** (A1 + W1′ supported). Chaotic modes are
   partially forecastable (+0.31…+0.81 at lead 16) — not separable from their own predictability limit here.
3. **Joint 16-channel analogs are WORSE than a single channel** for every family. Not evidence against W2:
   a 16·L-dim Euclidean analog search suffers the dimension explosion (W5, Van den Dool); joint chaotic corr
   FALLS with sight (+0.18 @16 → +0.03 @128). Naive retrieval cannot exploit joint information; a learned
   metric (attention) might — open.
4. **Sight has a family-dependent optimum (W5 confirmed, W9 direction supported):** cyclic modes improve up to
   L=128 (phase identification), chaotic modes peak at L=16 then fall; own-channel k=1 skill peaks at L=8 (h1)
   and L=16–32 (h64).
5. **Own channels vs the GRU (indicative only — F-18 is seed 0 with warmup-32 launches):** analog single k=1
   L=16: skill h1 +0.31 / h16 +0.24 / h64 +0.29, amplitude 1.04 (honest). GRU (F-18): h1 +0.713, h64 +0.264,
   amplitude 0.80. The GRU is far better at short lead; the analog matches it at h64 with honest amplitude.
   k=10 raises skill (h1 +0.57, h64 +0.53) but amplitude collapses to 0.52 at h64 — **W6 mean-collapse
   confirmed** (rank by var_ratio first, F-9). Joint k=1 over-disperses (amplitude 1.6–1.9).
