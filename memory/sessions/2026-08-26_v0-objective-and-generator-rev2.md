# Session 2026-08-26 → 08-28 — v0 encoder objective made to work, then generator rev2

**Transcript:** `~/.claude/projects/-home-sysadmin-jlee-venn/2f0d3a1d-a1b6-48bc-89e5-7e9017ed5e43.jsonl`
**Produced:** D-016 … D-021, findings F-1 … F-8.
**Phase:** L complete → early S (STYLIZE). Modules 2 (latent predictor) and 3 (forecaster) still unbuilt.

---

## 1. Where the session started

`src/train/train.py` + `config/config.yaml` existed from the previous session. The hard-whitening
fix had been *written into config but never run*. The last actual run had collapsed
(`var≈2.5e-6`, `white=16`). No evaluation existed beyond training-loss scalars, and no figures.

## 2. The arc, in order

### 2a. Three collapses, one root cause (D-017, F-2)

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

### 2b. First honest readout → two apparent structural limits (F-4, F-5)

Built `src/probes/evaluate.py` (per-channel descriptors + family labels, population, effective
rank, mask overlap, balanced reconstruction R², per-hidden-mode ensemble R²). It said:

- decorrelation saturated at off-diag corr² **0.244**, effective rank **4.08/16**, and λ_white
  from 1 → 500 changed *nothing*;
- slowness training **destroyed the chaotic modes**: ensemble R² 1.0 (untrained masks!) → 0.01–0.03.

A signed-`tanh` mask ablation reached 0.070 / 8.57, so I concluded non-negativity was the binding
constraint and proposed possibly revising D-005. **That conclusion was later shown wrong — see §2e.**

### 2c. USER RULE: never judge from metrics, always visualize (F-6)

> *"don't judge the result your self. Always make visualization on it."*

Built `src/probes/plots.py`. The first figure overturned the metric-based diagnosis immediately:
**the kernels had migrated to the dead corners of the domain** — the lowest-energy cells — because
`var(Δs)/var(s)` is a *ratio*, blind to amplitude: a tiny, almost purely slow tail in an empty
corner scores near-perfectly, while an energetic region carries a mix of slow and fast content.
That one mechanism explained the duplication, the lost chaotic modes, and the sub-random
reconstruction simultaneously. No scalar in the eval had surfaced it.

### 2d. Energy + coverage terms (D-019, F-7)

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

### 2e. Housekeeping + scale ladder (D-020)

User directives: remove the git commit (kept every file on disk; repo now has no commits), the
`.gitignore` now excludes `.claude/` and `memory/`, and train+eval TensorBoard merged into ONE
`<run>/tb/` per run. User also observed masks were shrinking — confirmed: from a ~4095-cell init to
a **median of 28 cells**, and *my energy term had made it worse* (median 104 → 28) because mean
pairwise covariance is maximized by a tiny coherent patch. Fix = per-channel geometric footprint
ladder (`L_size`), a tolerance-band hinge, not a pin. Footprints then spanned 8 → 1143 cells and
mask IoU halved (0.111 → 0.061) at no cost to any other metric.

### 2f. Field visualization → three generator defects (F-8) → rev2 (D-021)

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

## 3. Files created / changed this session

| file | state |
|---|---|
| `src/train/train.py` | loss rewritten: `corr` whitening + energy + recon + size + hinge; grad clip |
| `src/models/encoder.py` | multiscale/random/flat init, `signed` ablation flag, `soft_count()` |
| `src/data/synthetic.py` | rev2 patterns (dipole, wave packet, zero-mean band), periodic x, AR1 SST, private modes |
| `src/probes/evaluate.py` | NEW — post-hoc readout → `eval.json` + TB |
| `src/probes/plots.py` | NEW — masks / masks-over-energy / features / feature-variance |
| `src/probes/plots_field.py` | NEW — field snapshots / hidden modes / kernel activation / `field.gif` |
| `config/config.yaml` | all of the above as Hydra knobs; `hydra.job.chdir: true` |
| `memory/sop/01_…md`, `02_…md` | SOPs updated ahead of each code change (Golden Rule) |
| `memory/{decisions,findings,progress}.md` | D-016…D-021, F-1…F-8, log entries |

## 4. Reproduce

```bash
conda run -n oceanai python -m src.train.train                    # ~5000 steps, seconds on the H100
conda run -n oceanai python -m src.probes.evaluate                # newest run → eval.json + TB
conda run -n oceanai python -m src.probes.plots                   # 4 figures → <run>/figs/
conda run -n oceanai python -m src.probes.plots_field             # field figures + field.gif
```
Best run of the session: `.tmps/runs/20260827_151413/` (figures in its `figs/`).
Runs are ephemeral under `.tmps/` (D-014); nothing has been promoted to `results/`.

## 5. Open thread — pick up here

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

## 6. Standing rules confirmed this session

- **Always visualize before judging a result** — it overturned the diagnosis twice here.
- **The user owns git.** No commits; work stays local. `memory/` and `.claude/` are git-ignored but
  still maintained. (Update 2026-09-28: `memory/` is now the tracked home of the project record;
  `architecture/` was retired and merged into it.)
- Ephemeral output → `.tmps/`; `results/` only with explicit authorization; Hydra owns every
  hyperparameter; ONE TensorBoard dir per run.
- Update the SOP in `memory/sop/` *before* changing the code it describes.
