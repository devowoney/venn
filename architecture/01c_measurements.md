# Measurements — every number, with the run that produced it

All runs: K=16, seed 0, 5000 steps unless stated. Run dirs are under `.tmps/runs/` (gitignored, so
they do NOT travel with the branch — the numbers are recorded here instead).

## Testbed generations

| | rev2 (`signed_patterns`, D-021) | rev3 (`stationary_constant`, D-025) |
|---|---|---|
| mode 0 amplitude | OU τ=200, **standardized to unit variance** | **constant**, `mean 1.000 / std 0.000e+00` |
| field time-mean vs mode-0 pattern | — | **corr 1.000**, rms 0.818 (field std 1.0) |
| labeller vs truth, seeds 0–4 | 9/10 (mode 0 → cyclic, trend share 0.63 — correct under D-025) | **10/10**, population 1/3/6 |

## K sweep — the family population does not respond to K (F-12)

Runs `20260831_073232 / 073350 / 073440 / 073532 / 073623`, rev2 field. Read as slow/cyclic/fast.

| K | old broken readout | re-scored | mode-0 max\|corr\| |
|---|---|---|---|
| 4 | 0/4/0 | 3/0/1 | 0.92 |
| 8 | 0/8/0 | 4/2/2 | 0.62 |
| 16 | 0/13/3 | 12/0/4 | 0.93 |
| 32 | 0/30/2 | 22/3/7 | 0.82 |
| 64 | — | 36/10/18 | — |

## Ladder development on the rev2 field (D-024)

| | baseline `20260831_091454` | +band/line `20260831_091525` | +`L_mem` `20260831_091908` |
|---|---|---|---|
| population (slow/cyc/fast) | 10/1/5 | 6/1/9 | 5/1/10 |
| role obedience | n/a | 0.75 | 0.69 |
| slow rung vs true mode 0 | (unassigned) | corr 0.50 to a *chaotic* mode, τ_e 36 | **corr 0.92, τ_e 87** (truth 106) |
| worst hidden-mode max\|corr\| | 0.25 (m6, m9) | 0.65 | 0.61 |
| effective rank / 16 | 9.23 | 10.08 | 9.56 |
| balanced recon R² | 0.9891 | 0.9891 | 0.9891 |

Also: `lambda_slow=0` ablation gave obedience 0.69 / population 7/0/9 — slightly worse, so mean
slowness stays at 1.0 as a weak tiebreak (consistent with F-7).

Fast rungs obeyed near-perfectly with honest `gap1` 0.10–0.13 against the truth's chaotic 0.07–0.20;
the baseline's "chaotic" channels sat at 0.02–0.03, i.e. not actually chaotic.

## rev3 field — the stationary observer (D-025, F-13)

| | baseline `20260901_080438` | ladder + `L_level` `20260901_080506` | + de-aligned sizes `20260901_080655` |
|---|---|---|---|
| population (s/c/ch) | 0/14/2 | **2/5/9** | 1/6/9 |
| population, D-026 rule | 0/11/5 | **2/5/9** | — |
| role obedience | n/a | **1.00** | 0.94 |
| flattest `amp_ratio` | **0.778** (still moving) | **0.023** | 0.047 |
| flat channels (<0.05) | 0 | 2 | 1 |
| flat-rung footprint | — | 2310–2590 cells | 1313–1318 cells |
| mask vs φ₀ (diagnostic) | −0.048 | −0.276 | −0.272 |
| effective rank / 16 | 8.57 | 9.61 | 9.30 |
| balanced recon R² | 0.7763 | 0.7765 | — |

Design target for K=16 at 1/3/6 proportion: 1.6 / 4.8 / 9.6.

## Structure shares, ladder run `20260901_080506` (D-026)

| ch | role | amp_r | trend | osc | resid | label |
|---|---|---|---|---|---|---|
| 0 | slow | 0.049 | 0.36 | 0.23 | 0.41 | stationary *(flat gate)* |
| 1 | slow | 0.023 | 0.03 | 0.89 | 0.08 | stationary *(flat gate)* |
| 2 | cyclic | 1.147 | 0.00 | 0.87 | 0.13 | cyclic |
| 3 | cyclic | 3.221 | 0.00 | 0.84 | 0.16 | cyclic |
| 4 | cyclic | 0.938 | 0.01 | 0.84 | 0.15 | cyclic |
| 5 | cyclic | 0.906 | 0.00 | 0.93 | 0.07 | cyclic |
| 6 | cyclic | 0.970 | 0.04 | 0.70 | 0.26 | cyclic |
| 7 | fast | 1.076 | 0.09 | 0.07 | 0.84 | chaotic |
| 8 | fast | 2.647 | 0.01 | 0.38 | 0.61 | chaotic |
| 9 | fast | 0.903 | 0.00 | **0.41** | 0.59 | chaotic |
| 10 | fast | 1.298 | 0.09 | 0.07 | 0.84 | chaotic |
| 11 | fast | 0.691 | 0.01 | **0.32** | 0.67 | chaotic |
| 12 | fast | 2.027 | 0.00 | 0.41 | 0.59 | chaotic |
| 13 | fast | 3.895 | 0.00 | 0.38 | 0.62 | chaotic |
| 14 | fast | 0.640 | 0.01 | 0.42 | 0.56 | chaotic |
| 15 | fast | 1.492 | 0.21 | 0.15 | 0.64 | chaotic |

Truth reference: cyclic modes osc 0.93/0.94/0.94; chaotic modes osc 0.07–0.34, residual 0.65–0.83.

**Stationary observer, ch1:** `amp_ratio` 0.023 (FLAT); 3 hidden modes reach its footprint;
net fluctuation / independent addition = **1.01** → incoherent addition, a quiet footprint rather
than a cancelling combination.

## Gradient matching (D-019 convention — match GRADIENTS, not loss values)

Measured at init, K=16, seed 0, against `|g_white|` as the reference:

| term | \|g\| at init | weight |
|---|---|---|
| `l_white` (reference) | 3.9e-3 … 5.1e-3 | 1.0 |
| `L_band` | 7.8e-4 | 5.0 |
| `L_line` | 1.0e-3 | 4.0 |
| `L_mem` | 9.5e-3 | 0.4 |
| `L_level` | 4.0e-2 | 0.13 |

`L_level` is the steepest term in the loss by far (slow rungs start at r ≈ 0.09–0.10 against a 0.95
target), hence the small weight.

## Reference calibration of the labeller statistics (truth modes, seed 0)

| | line_frac | τ_e | ACF trough | rebound |
|---|---|---|---|---|
| 3 cyclic modes | 0.94 / 0.95 / 0.94 | 12 / 27 / 57 | −0.92…−0.98 | +0.85…+0.97 |
| old OU "stationary" | 0.67 | 106 | −0.41 | −0.35 |
| 6 chaotic modes | 0.10–0.46 | 3–7 | −0.13…−0.63 | −0.03…+0.58 |

Baseline channels the user flagged as cyclic (rev2 run `20260831_091454`): ch2 trough −0.83 /
rebound **+0.74** (genuinely oscillatory); ch4 −0.15/−0.04, ch5 −0.31/−0.05, ch6 −0.40/−0.31,
ch10 −0.61/−0.47 (no rebound). Superseded as a classification basis by D-026, but kept because it
documents why raw-series statistics mislead: they measure the residue, not the structure.

## D-027 shape term — 5 seeds per arm (runs `.tmps/runs/<arm>_seed{0..4}`, rev3 field)

| arm | config | ambiguous ch | fast struct max | distinct cyclic periods | flat ch | obedience |
|---|---|---|---|---|---|---|
| `base` | `shape_objective: line` | 6.8 | 0.44 | 2.4 | 0.8 | 0.91 |
| `struct` | structure, all rungs, λ 3.5 | 3.0 | 0.31 | 1.0 | 0.6 | 0.91 |
| `flat98` | line, `flat_target 0.98` | 5.8 | 0.44 | — | 1.8 | 0.99 |
| `comb` | structure all, λ 3.5, flat 0.98 | 2.2 | 0.31 | — | 1.8 | 0.99 |
| `lam10` | structure all, λ 10, flat 0.98 | 0.0 | 0.30 | 1 (seed 0) | 2.0 | 1.00 |
| `fastonly` | **default**: structure fast, λ 2.3 | 1.2 | 0.31 | 1.4 | 0.8 | 0.93 |

Cyclic rung periods (ch2–6), baseline → fast-only: s0 286/143/61/61/61 → 61×5; s1 286/143/61/61/143 →
61×5; s2 286/143/143/143/286 → same; s3 143×5 → 286/286/143/143/143; s4 286/143/61/61/61 → 61×5.
FFT periods are 2000/k: 286 = the period-300 cycle, 143 = 140, 61 = 60.

Collective, seed 0: balanced recon R² base 0.7765 / comb 0.7764 / lam10 0.7765 / fastonly 0.7765;
effective rank 9.61 / 9.28 / 9.49 / 9.44; slowness_spread 88.3 / 16.0 / 17.1 / 20.1.
Gradients at init (K=16, seed 0): |g_white| 5.1e-3; |g_struct| 1.5e-3 (all rungs), 2.3e-3 (fast only).
