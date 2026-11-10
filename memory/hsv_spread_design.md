# HSV spread — design record (module 2 uncertainty from adjoint sensitivity)

> Started 2026-11-04/05 on branch `history-forecastor`. A record of the design discussion with the user, so the work
> can resume after a break. Status lines are updated as each question is confirmed. Decisions that become settled
> are copied into `decisions.md` (D-entry + SOP 04) before any code (Golden Rule).

## 0. Why this exists

- D-042 / F-37: the forecaster's scenarios were made by a learned noise generator trained by fair CRPS. Members became
  coherent rollouts and the width followed the families, but the width was only "learned from training errors": ~1.4x
  too narrow beyond lead 4 (spread/error 0.6-0.8), a generalization gap. Re-learning the generator on the 250-step
  held-out slice memorized it (rejected).
- User idea (2026-11-04): "the hippocampus = forecaster is a differentiable latent-space emulator. We can get the
  sensitivity of the loss (adjoint), so generate the spread with this sensitivity."
- **First objective (user, 2026-11-05):** the sensitivity is introduced to make the uncertainty (spread) of the latent
  forecaster. Build this step first. Online DA (Part B below) is a later direction, theory only for now.

## 1. How the idea was refined (in order)

1. Two sensitivities: the LOSS adjoint (needs truth -> training only) and the FORECAST adjoint (Jacobian, no truth ->
   usable at forecast time). Combined: the loss adjoint / 4D-Var side gives the SHAPE of initial-condition errors (B),
   the forecast adjoint gives their GROWTH (singular vectors).
2. User choices: **combined adjoint, nonlinear rollouts, perturb the LAST state z_t** (first state of the forecast
   window); the history before it stays fixed.
3. Terms (user, DA background): B = background-error covariance (not "C"); members from **Hessian singular vectors
   (HSV)**. The Hessian of the 4D-Var cost is the inverse ANALYSIS-error covariance A^-1 = B^-1 + Σ Lᵀ Hᵀ R^-1 H L; it
   is the initial norm of the HSV. With only z_t perturbed, the window at the analysis time is the state itself.
4. Observations: in the latent-only version they would be the observer's outputs (H = I), nothing random. Random /
   perturbed observations appear only in an OSSE or an EDA.
5. Member = the WHOLE nonlinear rollout started from z_t + δ_m (the perturbed z_t is only the start). The tangent-linear
   model chooses directions; the full forecaster propagates them, so members can saturate and split regimes.
   Rollout vs the current direct-lead form: the direct form is trained toward the conditional mean, which forgets the
   initial condition at long lead for chaos -> a perturbation may SHRINK where it should grow. To check on one seed.
6. **User: DA on the RAW state, perturbation only for the corresponding signals.** The observer is linear
   (`S = <mask, field>`, `src/models/encoder.py:102`, `src/models/child_encoder.py:55`; masks in `artifacts.npz`
   `masks [K,2,64,64]`, N = 8192). The forecaster sees the field only through E, so:
   - B, R are defined on the raw field; the computation only needs their projections B_z = E B_x Eᵀ, R_z = E R Eᵀ
     (K x K), computed ONCE (K applications of the B_x operator, linear in N). B_x is a rule (variance + spatial
     correlation), never an N x N matrix.
   - A raw perturbation outside what the eyes see (null space of E) cannot change the forecast; the HSV therefore
     lie in "the corresponding signals" automatically.
   - Per-launch cost stays K-dimensional. The latent shortcut would break only for a nonlinear observer or a
     raw-space B that changes at every launch.
   - Raw map of a member's start: minimum-norm lift δx = B_x Eᵀ B_z^-1 δz; members decode to the field by D.

## 2. Part B — theory: bridging the emulator to online DA (later work, not in the first step)

State-space form (x raw state, z latent, H_t history = the emulator's memory):

```
z_t     = E x_t + c                         observer (linear masks + fixed shift/scale)
z_{t+1} = F(H_t) + ε_t ,  ε ~ N(0, Q_z)     emulator step (rollout = append and repeat), model error
y_t     = 𝓗 x_t + r_t ,   r ~ N(0, R)       real observations in raw space
x_t     = D z_t + d + x_t^⊥                 decoder + what no eye sees
y_t     = G z_t + g + ν_t ,  G = 𝓗D, g = 𝓗d,  R̃ = R + 𝓗 P^⊥ 𝓗ᵀ     (representation error, P^⊥ = cov x^⊥)
```

Two observation routes: (a) encode observations z^o = E y (gridded full coverage, R_z = E R Eᵀ); (b) decode the state
into observation space (general, sparse obs).

Analysis (3D-Var / Kalman, K-dimensional):

```
J(z) = ½ (z − z^b)ᵀ B_z^-1 (z − z^b) + ½ (y − Gz − g)ᵀ R̃^-1 (y − Gz − g)
z^a  = z^b + K (y − G z^b − g),   K = B_z Gᵀ (G B_z Gᵀ + R̃)^-1
A_z  = (B_z^-1 + Gᵀ R̃^-1 G)^-1 = (∇²J)^-1        <- the SAME A_z is the HSV initial norm (the bridge)
```

4D-Var through the emulator (window [t−L, t], control = state at window start):

```
J(δz_0) = ½ δz_0ᵀ B_z^-1 δz_0 + ½ Σ_j d_jᵀ R̃^-1 d_j ,   d_j = y_{t−L+j} − G F^(j)(z_0^b + δz_0) − g
∇J      = B_z^-1 δz_0 − Σ_j L_jᵀ Gᵀ R̃^-1 d_j ,          L_j = ∂F^(j)/∂z_0  (adjoint = backprop)
weak constraint: + Σ_j ε_jᵀ Q_z^-1 ε_j
```

Memory subtlety (new vs classical DA): the emulator state is the whole history. Filter update replaces only z_t
(a "kink" the emulator never saw); smoother update corrects every state in the window (dynamically consistent memory).

Cycling and B:

```
z^b_{t+1} = F(H^a_t)
static   B_z = E B_x Eᵀ                 EKF    B_{t+1} = L_1 A_t L_1ᵀ + Q_z  (affordable: K x K)
ensemble B_ens = X Xᵀ/(m−1)             hybrid B = β_s² B_z + β_e² B_ens     (hybrid 4DEnVar)
HSV:  L_{a*}ᵀ P L_{a*} v_i = λ_i A_z^-1 v_i ,  δz_m = α Σ_i ξ_{m,i} v_i ,  members = rollout from z^a_t + δz_m
```

Estimation from training data: Q_z = cov(z_{t+1} − F(H_t)) on held-out data (truth known); Desroziers relations
E[(h(z^a) − h(z^b)) d^bᵀ] ≈ G B_z Gᵀ and E[d^a d^bᵀ] ≈ R̃ (iterate to a fixed point).

Brain reading: hippocampus predicts z^b, the eye observes y, the analysis increment K(y − h(z^b)) is the prediction
error weighted by trust (B vs R̃) = predictive coding; online DA = the emulator perceiving.

Suggested online-DA directions (later): (1) OSSE in the synthetic testbed (synthetic obs y = 𝓗 x_true + noise),
latent 3D-Var + static B_z, cycled; (2) distribution shift: the emulator is trained on perfect encodings but will read
analyses -> noisy-history training or DA-in-the-loop training; (3) latent 4D-Var with the emulator adjoint, strong ->
weak constraint, smoother memory update; (4) hybrid EnVar with HSV members + EDA; (5) observing-system design from the
adjoint sensitivity (which cells / eyes matter; feedback to the observer).

## 3. Part A — first step: HSV spread on the latent forecaster. Question-by-question confirmation

| # | question | status |
|---|---|---|
| Q1 | What uncertainty do the members represent: (a) initial-condition only (HSV + rollout) or (b) + stochastic emulator error ε ~ N(0, s·Q_z) per rollout step? Suggested: (a) first, measure, add (b) only if long leads stay narrow. | **(a) initial-condition only** (user 2026-11-05) |
| Q2 | The raw-space B_x: form (variance + correlation length?) and how it is estimated (4D-Var increments + Desroziers / NMC lagged differences / EDA). | **Prior B_z alone first** (user 2026-11-05): B_x = Σ C Σ, assumed analysis-error rule; σ_v(cell) = std of the field's one-step change x_{t+1} − x_t, Gaussian correlation length ℓ from that change's spatial correlation, same ℓ for SSH/SST, no cross term; B_z = E B_x Eᵀ; overall size left to α (CRPS, Q6). The loss-adjoint 4D-Var refinement of B_z (increments + Desroziers) = the NEXT step. |
| Q3+Q4 | (merged, user) Observation at the start / R, and therefore the initial norm A_z. | **(i) no observation term: A_z = B_z** (user 2026-11-05) — strictly B-norm singular vectors (= HSV when observations at t add little). A synthetic R_z = E R_x Eᵀ (white generator noise) was not taken: it would impose "small eyes uncertain" whatever B_x says. The R_z term enters with real observations (OSSE / online DA, Part B), same code. |
| Q5 | Tangent-linear model: rollout vs direct form (check on one seed), optimization lead a*, final norm (latent or field energy via D). | **Confirmed** (user 2026-11-05): 5a rollout (perturb z_t, recompute token t, feed lead-1 back 64 times with a key/value cache), with a ONE-SEED CHECK shown first (rollout vs direct RMSE by lead/family; perturbation growth rollout vs direct); chunked rollout (direct heads in blocks, e.g. 8 leads) as fallback if it drifts. 5b a* = 16 (members still rolled out to 64; rollout step = lead step; a* only chooses directions, TL valid ~1 Lyapunov time ≈ 22). 5c final norm = latent, each channel in its own training-std units. |
| Q6 | Members: number of singular vectors n, ± pairs, residual draw from A_z, one size α fitted by CRPS on held-out [7750, 8000). | **Confirmed** (user 2026-11-05): 6a dynamics = a DETERMINISTIC TWIN trained on the layer-2 observers (`ensemble.enable=false`, MSE, fit [0,7750), runs `twin_seed*`), not the CRPS-trained base heads. 6b n = 8 B_z-normalized SVs at a* = 16; δ_m = α Σ ξ_{m,i} v_i, ξ ~ N(0,1); ± pairs; 32 members (16 pairs); ONE α by CRPS on held-out [7750,8000) (log grid ~20 values, all channels/leads pooled); SVs only, no residual draw (added later if short leads are narrow). |
| Q7 | Evaluation: same readout as F-37 (spread/error per family and lead, roughness, per-launch spread vs error, fans) + latent growth rates σ_i(a) vs chaos Lyapunov time, error projection on evolved SVs vs chance. | **Confirmed** (user 2026-11-05): F-37 readout (5 seeds, 1937 launches, 32 members; spread, RMSE % of size, spread/error, roughness, per-launch spread vs error at h16, fans, accuracy of twin and member mean) + (1) growth spectrum σ_i(a) vs Lyapunov ~22, (2) family loading of v_i and evolved U_i, (3) share of the real error at a* in span(U_1..8) vs random 8-dim (≈ 8/45). Order: D-043 + SOP 04 → rollout code + ONE-SEED CHECK shown to the user (decide rollout vs chunked) → B_z, SVs, members, full readout. |

Answers are filled in below as they come.

## 4. Build log

### 2026-11-05 — records, twin, rollout code, one-seed check (Q5a)
- D-043 + SOP 04 "Spread from Hessian singular vectors" written first.
- Twin `twin_seed0..4` (`.tmps/runs_hf/launch_twin.sh`): RMSE % of size h1/h8/h64 cyclic 3.0/5.3/8.9, chaotic
  3.4/19.1/42.6 (≈ D-042), causality 0.0. Eval `.tmps/eval_hf/twin/`.
- `src/models/hsv_spread.py` `CachedRollout`: key/value-cached rollout, `chunk` = direct leads taken per feedback.
  Exact: teacher-forced = full causal pass (7e-7); 2048 launches x 64 steps in < 2 s.
- Field rebuilt from the observer `.hydra` config: S reproduced to 1e-5.
- One-seed check `src/probes/hsv_rollout_check.py` (twin_seed1) -> `.tmps/eval_hf/hsv/rollout_check/`:
  - pure rollout (chunk 1) UNUSABLE: cyclic RMSE h64 52.9 % vs direct 9.0; chaotic worse than persistence from ~h20;
    a unit start perturbation explodes in EVERY family (h64: chaotic 233x, stationary 160x, cyclic 70x) = rollout
    instability, not physics.
  - chunked: c8 / c16 / c32 h64 RMSE cyclic 26.0 / 18.8 / 15.0, chaotic 48.7 / 38.7 / 35.5 (direct 9.0 / 31.2);
    growth h64 chaotic 22.9 / 9.0 / 4.8, stationary 12.3 / 5.3 / 3.5, cyclic 8.0 / 5.5 / 4.3.
  - direct: response rises to ~3 (chaotic), ~4-5 (cyclic), ~2-3 (stationary) and decays after ~h40 (mean damping).
  - in NO form does a random start perturbation grow on chaotic channels only: family selectivity, if any, must
    come from B_z and the SV selection.
- Found while preparing B_x (Q2): the 1-step field change is white generator noise for 30 % (SSH) / 71 % (SST) of its
  variance (`obs_noise` 0.05); the rest is spatially smooth (correlation ≈ 0.70 x Gaussian, ℓ ≈ 5 cells). Question for
  the user: amend Q2 (remove the white nugget, Hollingsworth–Lönnberg style).
- **Q5a-final (user 2026-11-05): CHUNK 16.** Leads 1-16 = the direct answer (the SVs at a* = 16 see the clean
  forecaster); members are fed back at leads 16, 32, 48 so a start error can regrow past the direct form's damping.
  Pure rollout rejected (instability); chunk = how many direct leads are kept per question, feedback = appending the
  forecaster's own answers to the history as if observed and asking again from the new end.
- **Q2-amend (user 2026-11-06): B_x from the SMOOTH part of the one-step change, the cell-by-cell noise removed**
  (Hollingsworth–Lönnberg): correlation C(r) = exp(−r²/2ℓ²) fitted on distances r ≥ 1 (ℓ ≈ 5 cells, shared SSH/SST);
  per-cell size σ_v²(cell) = var(Δx_v at the cell) − nugget_v (nugget = cov(0) − fit extrapolated to r = 0), clipped
  at 0. The white part = observation noise (R, excluded in this step).

### 2026-11-06 — first HSV spread test (twin_seed1, coarse α grid) -> `.tmps/eval_hf/hsv/spread_test/`
- Code: `src/models/hsv_spread.py` (field_change_stats, apply_bx, latent_b, tangent_linear, singular_vectors,
  hsv_members), `src/probes/eval_hsv_spread.py`; `eval_history_forecaster.py` gained `eval_one(..., ensemble_fn)` and
  `report(R, Q, out)` (summary/figures for ensembles made outside the model).
- B_x: ℓ = 6.6 cells, smooth share of the 1-step change 0.52 (SSH) / 0.28 (SST). α = 1 (coarse grid: B at its natural
  one-step size).
- Works: family ordering emerges (h16 spread % of size: stationary 0.2, cyclic 8.6, noise 9.8, chaotic 15.9); the
  evolved patterns load on chaotic channels (53 % vs 39 % chance); 55 % of the real lead-16 error lies in the span of
  the 8 evolved patterns (chance 16 %).
- Not yet honest: chaotic spread/error 3.0 (h1) -> 1.0 (h8) -> 0.67 (h64); cyclic 1.2-1.4; stationary 0.4. Growth
  σ_1 ≈ 2 at a* = 16 (slower than the real error growth). Members smoother than the truth (roughness 0.2-0.7: no
  forecaster noise, Q1 a). Chunk-16 member mean costs accuracy at long lead (cyclic h64 16.4 vs direct 9.0).
- B mismatch found: start patterns load 73 % on STATIONARY children. Prior B_z std (model units) vs the exact
  projection of the same smooth 1-step change (cov(Δz) − E·nugget·Eᵀ): cyclic 0.094/0.105, chaotic 0.163/0.174,
  noise 0.116/0.117 (rule fine), stationary 0.663/≈0 (rule wrong: a stationary child's real change is pure noise;
  its mask was shaped so the smooth movement cancels inside it, which an isotropic distance rule cannot know).
  The exact projection alone is not usable as is (negative eigenvalues: one global nugget over-subtracts).
- **B-fix (user 2026-11-10): option (a).** Each eye's start-error SIZE = what that eye really shows in its smooth
  one-step change, var(Δz_i) − (E·nugget·Eᵀ)_ii clipped at 0 (≈ 0 for stationary children, within ~10 % of the rule
  for the others); the CORRELATIONS between eyes still come from the B_x rule: B_z ← D B_z D, D_ii = sqrt(measured /
  rule). Then all 5 seeds with the fine α grid (25 values).

### 2026-11-10 — 5 seeds -> F-38 (`.tmps/eval_hf/hsv/spread/`)
SVs point at the real errors (49-58 % vs 15-21 % chance), α = 1 (4/5 seeds), family ordering emerges; but SV growth
~2x by h16 vs real error ~8x -> too wide early (chaotic 4.0 at h1), honest h8-h16, narrow at h64 (0.65); cyclic too
wide; members' mean worse than the direct forecast. Reading: the error growth is mostly the emulator's own error (Q1 b).
Next (user decides): add emulator error (Q1 b), and/or refine B by the loss-adjoint 4D-Var (2b).

### 2026-11-10 — HSV paused (user: "let us stop here for hsv")
State at pause: F-38. Open next steps if resumed: emulator error (Q1 b), loss-adjoint 4D-Var refinement of B (2b), online DA (Part B).
