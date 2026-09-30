# Reconstruction score — R², the noise ceiling, and why we report R² / ceiling

> Standalone note (2026-09-30, branch `feature-observer`). Answers: "R² represents reconstruction, so why divide it
> by a noise ceiling?" Decision: **D-032** (`decisions.md`). Evidence: **F-25** (`findings.md`), numbers in
> `measurements.md` ("Reconstruction score"). Code: `src/probes/observer_stability.py` (`oos_recon_r2`, `noise_ceiling`).

## TL;DR

- **R²** scores the reconstruction of the **observed field** = state + injected noise.
- The noise is unpredictable by construction, so no observer can reach R² = 1. The best any observer can reach is
  the **noise ceiling C** = the state's share of the field's variance.
- Exactly: **R² = C × R²_state**. So **R² / C = R²_state** = how much of the *state* the eye reconstructs.
- C is a property of the **field** (how much of the grid carries signal), not of the eye. Raw R² mixes the two;
  R² / C isolates the eye. Example: seed 3 has R² = 0.483 but C = 0.491 → **R²/C = 0.983**.

---

## 1. The data

For every cell c (8192 = 2 variables × 64 × 64) and time t, the generator (`src/data/synthetic.py`) builds

$$ y_c(t) = f_c(t) + \varepsilon_c(t) $$

- $f_c(t)$ — the **state**: the superposition of the mode patterns (stationary, cyclic, chaotic). This is what the
  observer should capture.
- $\varepsilon_c(t)$ — the **injected observation noise**: i.i.d. Gaussian, independent of $f$ and of every other cell
  and time step, variance $\sigma^2 = 0.05^2 = 0.0025$ (`data.obs_noise = 0.05`). Each variable is standardized to
  unit variance **before** the noise is added, so σ is the same for every seed and every cell.

Independence gives

$$ \operatorname{Var}(y_c) = \operatorname{Var}(f_c) + \sigma^2 . $$

## 2. The R² we compute ("balanced", out-of-sample)

1. **Standardize each cell** with training statistics, so every cell counts equally ("balanced"):
   $$ z_c(t) = \frac{y_c(t) - \mu_c}{s_c}, \qquad s_c^2 = \operatorname{Var}(y_c). $$
2. **Linear decoder** from the K channels $s_i(t) = \langle \text{mask}_i, y(t) \rangle$ (standardized) back to each cell,
   least squares on the training slice `[0, 8000)`:
   $$ \hat z_c(t) = \beta_{c0} + \sum_{i=1}^{K} \beta_{ci}\, s_i(t). $$
3. **Score on validation** `[8000, 10000)`:
   $$ R^2 = 1 - \frac{\sum_c \sum_t \big(z_c - \hat z_c\big)^2}{\sum_c \sum_t \big(z_c - \bar z_c\big)^2}. $$
   Every cell has variance ≈ 1, so this pooled R² ≈ the average over cells of the per-cell $R^2_c$.

Note: the **decoder** is fitted; the **eye** (masks) is whatever snapshot is being scored. That is why untrained eyes
can score high (section 6).

## 3. The noise ceiling

Split the standardized cell into its state part and its noise part:

$$ z_c = \tilde f_c + \tilde\varepsilon_c, \qquad \tilde f_c = \frac{f_c-\mu_c}{s_c}, \quad \tilde\varepsilon_c = \frac{\varepsilon_c}{s_c}. $$

Their shares of the (unit) variance:

$$ \operatorname{Var}(\tilde f_c) = \frac{\operatorname{Var}(f_c)}{\operatorname{Var}(y_c)} \equiv C_c,
\qquad \operatorname{Var}(\tilde\varepsilon_c) = \frac{\sigma^2}{\operatorname{Var}(y_c)} = 1 - C_c . $$

The reconstruction error splits the same way:

$$ z_c - \hat z_c = \underbrace{(\tilde f_c - \hat z_c)}_{\text{state missed}} + \underbrace{\tilde\varepsilon_c}_{\text{noise}} . $$

The noise is independent of everything the decoder sees, so the cross term averages to 0:

$$ \mathbb E\big[(z_c-\hat z_c)^2\big] = \mathbb E\big[(\tilde f_c-\hat z_c)^2\big] + (1 - C_c) . $$

Divide by the total variance (= 1):

$$ R^2_c = C_c - \mathbb E\big[(\tilde f_c-\hat z_c)^2\big] \;\le\; C_c = 1 - \frac{\sigma^2}{\operatorname{Var}(y_c)} . $$

$C_c$ is the **noise ceiling of cell c**: even a perfect observer ($\hat z_c = \tilde f_c$) leaves the noise. The bound
holds for ANY observer and ANY decoder (linear or not), because ε is unpredictable. The field ceiling is the average:

$$ C = \frac{1}{N}\sum_c C_c . $$

(Neglected: a channel contains the noise of the cells in its footprint, so a cell's own noise leaks into the
channels by ~1/footprint size — negligible.)

## 4. Why divide by the ceiling

Define the **state R²** — how well the decoder reconstructs the state alone, relative to the state's own variance:

$$ R^2_{\text{state},c} = 1 - \frac{\mathbb E\big[(\tilde f_c-\hat z_c)^2\big]}{\operatorname{Var}(\tilde f_c)}
= 1 - \frac{\mathbb E\big[(\tilde f_c-\hat z_c)^2\big]}{C_c} . $$

Substitute into section 3:

$$ \boxed{\,R^2_c = C_c \cdot R^2_{\text{state},c}\,} $$

The measured R² is a product of two factors:

| factor | meaning | property of |
|---|---|---|
| $C_c$ | how much of the cell is state at all | the **field** (where the patterns sit) |
| $R^2_{\text{state},c}$ | how much of that state the eye reconstructs | the **eye** — what we want to judge |

Pooled over cells:

$$ \frac{R^2}{C} = \frac{\sum_c C_c\, R^2_{\text{state},c}}{\sum_c C_c} $$

i.e. **R²/C is the state R², averaged over cells with each cell weighted by how much state it holds.** Pure-noise
cells ($C_c \approx 0$) drop out — there is no state in them to reconstruct.

**Reading the numbers**

- **R² / C** — fraction of the recoverable signal captured (1.0 = everything). The score to compare observers/seeds.
- **C − R²** — what a better observer could still gain.
- **1 − C** — the part of the field that is pure noise; no model should be asked to reproduce it.

## 5. Worked example — why seeds differ with the same noise

The noise (σ = 0.05) is identical for every seed. What differs is **where the unit variance sits**: the generator
places the mode patterns at random. Cells that no pattern reaches have $\operatorname{Var}(f_c)\approx 0$, so
$C_c \approx 0$ (pure noise); cells under a pattern have signal std up to ~1.8 ≫ 0.05, so $C_c \approx 1$.

| seed | cells with signal < noise | C (ceiling) | eye R² (K=16, trained) | R² / C |
|---|---|---|---|---|
| 0 | 15% | 0.821 | 0.816 | 0.994 |
| 1 | 23% | 0.738 | 0.731 | 0.991 |
| 2 | 9% | 0.879 | 0.879 | 0.999 |
| 3 | **50%** | **0.491** | **0.483** | **0.983** |
| 4 | 15% | 0.820 | 0.821 | 1.000 |

(Training `[0,8000)` arm, `.tmps/runs/obs_t8000_seed*`, last snapshot.)

Seed 3: the patterns landed in the bottom half, the top half is pure noise. Bottom half R²_c ≈ 1, top half ≈ 0 →
average 0.48. The per-cell R² map of the eye matches the per-cell ceiling map at corr **0.999**
(`.tmps/k_sweep/cell_maps.png`). The state is reconstructed; only the noise is not.

## 6. Why even untrained / random eyes reach the ceiling (current testbed)

Every channel is linear in the field, and the state is a sum of patterns with time-varying amplitudes:

$$ s_i(t) = \sum_j \langle \text{mask}_i, \varphi_j\rangle\, a_j(t) + \text{(noise)} . $$

K channels = K linear equations in the independent amplitudes. The number of independent time signals is

$$ D = 2\,(M-1) - 4 $$

(each fluctuating mode appears in SSH as $a_j$ and in SST as its AR1-filtered response — a different series; the
constant stationary mode contributes 0 because the decoder intercept absorbs it; 2 SSH-only + 2 SST-only private
modes count once). Verified as the exact rank of the noise-free field: **14 / 34 / 74 / 154** for M = 10/20/40/80.

If K ≥ D and the masks overlap the patterns in different ways, the decoder solves for all amplitudes and rebuilds
every cell — almost regardless of where the masks sit. Current testbed: K = 16 ≥ D = 14, hence

- untrained eyes already sit at R²/C ≈ 0.99 at optimizer step 1 (flat line in `stability.png`);
- white-noise masks reach 0.97, 16 single random pixels 0.73–0.94;
- more channels cannot help (K = 64 adds ≤ 0.002).

Only when **K < D** does the placement of the eye matter, and there training helps (M = 40, D = 74: trained K=16
0.76 vs untrained 0.71). Consequence: R²/C tests *sufficiency* only when eyes are scarce; it never tests
*meaningfulness* (families, stability).

## 7. Caveats

- C is estimated from **training-slice** variances while R² is scored on **validation**, so R²/C carries ≈ ±0.005
  error and can land slightly above 1.
- The ceiling needs σ. In the synthetic testbed σ is known (`obs_noise`), and $R^2_{\text{state}}$ could also be
  measured directly against the noise-free field (`generate_field` with `obs_noise=0`, same seed → identical modes).
  For real ocean data f is unknown, so the ratio form with an estimated observation-error variance is the one that
  transfers.
- "Balanced" weighting (every cell equal) is what makes C seed-dependent. The energy-weighted R² (cells weighted by
  variance) is ≈ 0.99 here and nearly insensitive to empty cells, but it is dominated by the largest patterns.

## 8. How to compute

```bash
# stability + R²/ceiling over training snapshots (needs train.snap_every > 0)
python -m src.probes.observer_stability --runs '.tmps/runs/obs_t8000_seed*'
#   -> stability.json: noise_ceiling per run, curve[*].r2_val and r2_val_norm (= R²/C)
```

Formulas in code: `oos_recon_r2` (sections 2) and `ceiling = mean(1 - obs_noise**2 / Y[:t_fit].var(0))` (section 3)
in `src/probes/observer_stability.py`.
