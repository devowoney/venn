# Conditional field forecast p(x_{t+1:t+l} | z_{t-k:t}) — design record

> Started 2026-11-12 on branch `history-forecastor`. Record of the design discussion with the user; status lines are
> updated as each question is confirmed. Settled decisions go to `decisions.md` (D-entry + SOP) before any code.
> Flow matching (D-044) is set aside for this step (user 2026-11-12).

## 0. Why this exists

User (2026-11-12): "build a novel method to estimate z_{t-k:t} -> p(x_{t+1:t+l} | z_{t-k:t}). Main idea is still an ML
module which predicts z_{t+1:t+l} from the history. But the question is how can we get physical distribution and
probability."

What the earlier spread attempts taught (the method must answer each):
- F-37 (learned CRPS generator): width learned on the trajectory it is scored on -> generalization gap, ~1.4x too narrow
  beyond h4.
- F-38 (HSV): start errors grow ~2x by h16 while real errors grow ~8x -> the error is mostly the EMULATOR'S OWN error,
  not amplified start error.
- F-39 (per-eye flow): honest-ish, but one eye at a time (no joint structure across eyes, no field).
- Module 3: x = D z + d + x^perp; x^perp (what no eye sees + decoding loss) has never had a distribution.

## 1. Proposal (assistant, 2026-11-12) — "retrieved error scenarios + physical lift"

Factorization (the field depends on the history only through the future latent states, plus the unseen part):

```
p(x_{1:l} | Z) = ∫ p(x_{1:l} | z_{1:l}) p(z_{1:l} | Z) dz
```

Latent factor — deterministic twin F (best mean, exists) + a LIBRARY of its own out-of-sample error trajectories:
- cross-fitting over training time (e.g. 4 folds of [0, 8000)): twin trained on 3 folds, launched on the 4th;
  store (key h_s = twin's last hidden token, e_s = z_{s+1:s+l} - F(Z_s) [l x C], r_s = field residual trajectory
  x_{s+1:s+l} - D z_{s+1:s+l} - d).
- at forecast time: weights w_s ∝ exp(-|h(Z) - h_s|^2 / τ) (attention over the memory of past mistakes = config (1)
  retrieval collapsed with (2) attention, W8);
  p(z_{1:l} | Z) = Σ_s w_s N(F(Z) + e_s, σ^2 I); members: draw s ~ w, z_m = F(Z) + e_s.
- every member's error is a REALIZED trajectory: coherent across leads and across eyes, includes emulator error
  (F-38), honest because measured off-sample (F-37), situation-dependent through h.

Physical factor — the lift to the field:
- x_m = D z_m + d + r_s (the same retrieval also gives a realized unseen-field trajectory), or the Gaussian
  version x | z ~ N(D z + d, P^perp), P^perp = low-rank (PCA) + diagonal cov of the decoder residual.
- density: mixture of low-rank Gaussians -> exact log p(x_true) (Woodbury), CRPS per cell, event probabilities.

Ruler of "physical": the testbed generator is known (constant / cyclic / Lorenz / OU + obs noise), so an ORACLE
posterior predictive (particle filter on the generator's hidden state given the observed z history) is computable:
compare our p to the true conditional p, not only to the single realized truth.

## 2. Questions (one at a time)

| # | question | status |
|---|---|---|
| Q1 | (superseded by H1/H2 below) Source of the latent spread: (a) retrieved out-of-sample ERROR trajectories added to the twin's forecast (recommended), (b) retrieved ANALOG futures (pure analog: z_t + realized increments), (c) a parametric error model (Gaussian / Q_z rollout). | replaced by user H1/H2 (2026-11-12) |
| Q2 | Retrieval key and weights: twin hidden token vs raw history window; τ, σ fit by held-out log score. | later |
| Q3 | Physical lift: realized unseen-field residual r_s vs Gaussian P^perp. | later |
| Q4 | Ruler: oracle posterior predictive from the generator + realized-truth scores (RMSE by lead, spread/error, CRPS, log score, Brier of events). | later |

## 3. User hypotheses (2026-11-12) — replace Q1

- **H1.** The highly compressing observer suppresses small signals / small errors. The pure 1-D history of an eye is the
  data-driven true latent; the predictor's forecast is the PRIOR.
- **H2.** The compressed uncertainty propagates as Gaussian noise in time: compressed uncertainty -> noise, real time
  step -> flow-matching noise-integration step. With stationary / cyclic / chaotic dynamics in the data, the predictor
  can be trained to give the physical spread (bigger displacement for dynamic eyes). Flow matching re-enters here.

Assistant reading: H1+H2 = the Mori-Zwanzig split of a resolved variable, z_{t+1} = drift(history) + noise(unresolved);
the generative path = the forecast path, i.e. a stochastic interpolant / bridge from z_t to z_{t+1} chained over leads
(prior art: DYffusion, Cachay et al. 2023; stochastic-interpolant forecasting, Chen et al. 2024; stochastic closures,
Chorin & Stinis, Lu-Lin-Chorin). Spread in physics = noise x instability: chaotic spread should come mostly from
the drift AMPLIFYING small noise (F-38: twin's amplification ~2x by h16 vs real error ~8x = the risk). Rollout
instability of lead-1 feedback (5a check) is the second risk.

Proposed first step (T1, measurement, no new model): on held-out data, the twin's one-step residual
e_{t+1} = z_{t+1} - F_1(Z_t) per eye: (i) white in time? (ii) Gaussian? (iii) size ordered by family? (iv) correlated
across eyes? If yes -> H2 holds at one step. Status: approved by the user 2026-11-12, run -> F-41 (below).

### T1 result (2026-11-12) -> F-41, `.tmps/eval_hf/one_step/` (`summary.txt`, `one_step.png`), probe `src/probes/one_step_residual.py`
- Gaussian: yes (excess kurtosis ~0 in every family; chaotic slightly heavy left tail).
- Size: stationary 0.3 % of size << cyclic 2.5 / chaotic 2.7 / noise 2.1. As a share of the true step: chaotic 0.32
  (the drift explains most of a chaotic step), the others ~0.7 (their step is mostly unpredictable).
- White? stationary yes; cyclic and noise lag-1 autocorr -0.30 (MA(1) signature) = WHITE OBSERVATION NOISE on the
  observed state (it enters the target z_{t+1} and, through z_t, the forecast). This noise does NOT propagate:
  stationary error is flat in lead (mse(a)/(a mse(1)) = 1/a exactly).
- Growth: chaotic error amplified ~4.5x beyond a random walk at h16 (the drift does carry "noise x instability");
  cyclic/noise ~0.3-0.5 (mix of non-propagating obs noise and propagating error).
- Joint: participation ratio ~10 effective noises for 38-52 eyes; strong blocks inside cyclic and chaotic = the same
  hidden mode seen by several eyes. The noise must be JOINT across eyes (a per-eye 1-D flow, D-044, cannot hold it).
- |error| lag-1 autocorr +0.10-0.13: mild state-dependent size (supports a history-dependent noise level).
- Reading for H2: two noises, not one -> state-space form: hidden latent with propagating process noise
  (integration step = real time step) + white observation noise on what the eye reports.

## 4. User hypothesis H3 (2026-11-12)

- **H3.** The predictor needs attention, to see anomalies / extreme events in the history. Flow matching runs ON the
  attention model's prediction: the attention model makes a different prediction as a function of the attention
  point or the observing length; apply the flow-matching velocity idea to it.

Assistant check (math): with the Gaussian path y_s = (1-s) eps + s y, the marginal velocity is
v(y_s, s | Z) = (E[y | y_s, Z] - y_s)/(1-s) = Σ_j w_j (y_j - y_s)/(1-s),
w_j ∝ p(j | Z) exp(-|y_s - s y_j|^2 / (2 (1-s)^2)).
With candidate futures y_j = what followed past situations j in the history, this IS softmax attention:
query (y_s, s), keys s*y_j (+ history keys for p(j|Z)), values y_j, temperature (1-s)^2 annealed by flow time.
- "attention point" = the random variable: each attended past situation gives one conditional velocity (one
  scenario); the marginal over attention points is the forecast distribution. CORRECT as stated.
- "observing length" (sight) = the set of candidates, not a random variable: predictions from different sights are
  not equally probable alternatives (a shorter sight is just less informed). Sight decides WHICH extremes can enter
  (an extreme outside the sight has zero weight); its probability is its attention weight.
- Risks: (1) the closed form memorizes (members = past futures, the pure analog option) -> a learned network with
  explicit cross-attention to history windows is needed to generalize; (2) unlike the rejected "memory" ensemble
  (perturbed attention memory, spread collapsed 0.04-0.09) the FM regression target pins the weights, no knob the loss
  can switch off; (3) 8000 steps hold few extremes and few long chaotic analogs -> one-step candidates (H2's real
  time step) are well populated, long leads come from chaining.
Status: proposed to the user.

## 5. User refinement (2026-11-12) — prior-centred flow, multi-step

- **Not necessarily one time step:** a flow may cover a window of several real steps.
- **"Non-supervised" flow:** the noise added in flow matching is Gaussian and the PREDICTION is the high-probability
  point -> the flow's source is a Gaussian centred on the attention predictor's forecast; the flow learns only how the
  real future departs from it (no target is given for the spread).

Assistant formalization: source x0 = F(Z) + Σ_a^{1/2} ε (prior, Σ_a = the predictor's joint error covariance per
lead, ~10 joint directions + diagonal, F-41), target x1 = real window z_{t+1:t+l}; path x_s = (1-s) x0 + s x1;
velocity net v(x_s, s | attention over history) regresses x1 - x0. Where the prior is right, v ≈ 0. Exact density
of the truth by the ODE likelihood.

Checks run (2026-11-12, twin_seed0..4, script `$CLAUDE_JOB_DIR/tmp/lead_gauss.py`, `gap.py`, not kept):
- Is the prediction the high-probability point? Per-eye marginal errors never bimodal (bimodality coeff 0.25-0.38,
  0 % of eyes > 0.555 at any lead) -> yes, the prediction sits at the peak. But NOT Gaussian on chaotic mid leads:
  excess kurtosis h1/h4/h8/h16/h32/h64 chaotic +0.1/+3.0/+4.4/+1.6/0.0/-0.1, noise 0/+0.6/+1.5/+0.5/-0.2/-0.1,
  stationary/cyclic ~0 -> sharp peak + rare large misses (regime flips) = what the flow must add to the Gaussian.
  (Pooled over launches: could also be a Gaussian whose width depends on the history.)
- Generalization gap of the twin, validation RMSE / training RMSE, h1/h4/h8/h16/h32/h64: stationary 1.04-1.10,
  cyclic 1.03/1.09/1.22/1.43/1.59/1.33, chaotic 1.02/1.11/1.49/1.96/1.90/1.45, noise 1.02/1.10/1.31/1.59/1.87/1.67.
  -> a flow trained on in-sample predictor errors would be ~2x too narrow on chaotic h16 (= F-37 again). Needs
  OUT-OF-FOLD predictions (cross-fitting) for the flow's training pairs. Lead 1 is honest (1.02-1.04): short windows
  are less exposed, longer windows more.

## 6. Decisions after §5
- Cross-fitting: **yes** (user 2026-11-12) -> D-045.
- Window: **(a) one 64-lead window, no chaining** (user 2026-11-12, "let's see what it gives"). Build details in D-045.

## 7. Step-1 result (2026-11-12) -> F-42
Cross-fitting works (honest diagonal prior). The flow from a diagonal start: ~15 % too narrow, rough members, and in
probability BELOW a static joint Gaussian of the out-of-fold departures. Next step proposed to the user: start the flow
from the JOINT Gaussian N(F(Z), Σ_joint) (low-rank + diagonal), so the correlations and coherent members come for free
and the flow only learns the state-dependent / non-Gaussian departure (H2: compressed uncertainty = Gaussian noise).

## 8. Step-2 result (2026-11-12) -> F-43
Joint source: coherent members; the joint Gaussian ALONE is honest and the most probable; the flow shrinks it ~15 % and
lowers log p by ~0.1 nats/number on every launch (uniform contraction). The flow carries a per-launch width signal
(rank corr +0.3..+0.5) a fixed Gaussian lacks. Next: user decides (see the step-2 report).

## 9. User clarification H4 (2026-11-12) — what was NOT tested

User: "what I imagined is that from multi prediction came from multi scenario of attention, we apply noising to find
the most likely prediction." -> D-045 (a flow learning the departure from ONE prediction) is NOT this idea.

H4 formalized: one attention model, J attention configurations c_j (attended points / sight) -> J whole 64-lead
predictions y_j = F(Z; c_j); noise them, p_s(y) = Σ_j w_j N(y; y_j, s^2) (s = compressed uncertainty, H2); denoise by
the closed-form velocity v ∝ Σ_j π_j(y,s)(y_j - y), π_j = softmax over scenarios (= attention), annealing s down:
large s -> one mode (weighted mean), small s -> separate modes (regimes); mode probability = weight of its basin.
Open: (1) scenario generator (sight length / attention masks / top-k points); (2) weights w_j (equal / fit to the
latest observed steps / attention scores); (3) can the full-sight twin give diverse, valid scenarios, or must it be
trained with random sight/masks. Proposed first measurement: vary the twin's sight on validation, scenario spread vs error.

### H4 measurement 1 (2026-11-12) -> F-44
Sight-length scenarios of the twin: nested quality ladder (longer = better), spread/error 0.1-0.76, truth outside
all scenarios most of the time, per-launch signal weak (chaotic +0.30). Side finding: full history (> training
length) hurts cyclic/noise vs sight 2048. Next: user decides.

## 10. Attention-mask scenarios (user 2026-11-12 "go with attention-mask scenarios to see") -> D-046
User: a narrow latent spread can be acceptable (compressed signal: one small latent error <-> large raw spread).
Assistant caveat: the decoder is linear, so the latent spread/error ratio carries unchanged to the SEEN part of the
field; extra raw spread lives in the unseen part (x^perp lift). Build: mask twin (random 256-step block masks in
training, last 64 steps always read), window 2048, scenario sets LOO (hide one block) and SINGLE (read one block).

### H4 measurement 2 (2026-11-12) -> F-45
Mask scenarios collapse (spread/error 0.01-0.12): any 256-step block + last 64 steps ~ whole window. Past is redundant
on this stationary testbed; scenario disagreement = epistemic ~0, real error = unpredictable part. Mask training is a
mild regularizer (long-lead RMSE down). Next: user decides.

## 11. New concept H5 (user 2026-11-12) — deterministic twin + Gaussian latent error + flow-matching DECODER

User: "Let history forecast as deterministic twin. But the prediction has error in Gaussian distribution. Then the
decoder is noising flow matching based non-linear operator which find a correction matching prediction z(t)
distribution." (Replaces the H4 mask-spread direction; D-047 mask twin stays recorded.)

Assistant reading: p(x|Z) = ∫ p(x|z) p(z|Z) dz with p(z|Z) = N(ẑ, Σ) (twin + the honest joint Gaussian of the out-of-
fold departures, F-43) and p(x|z) = a conditional flow-matching decoder (module 3) trained on (true z_t, x_t) pairs of
the training range (time-independent, no cross-fitting needed). Physical constraint: the observer is linear
(z = E x + c), so a decoded field must re-encode to its z -> the flow's correction lives in the NULL SPACE of E (project
its velocity); field members then carry the latent spread on the seen part + a learned spread on the unseen part
(H1: what compression removes is regenerated there).
Risk: on the M=10 testbed the linear decoder is already R^2/C 0.99 and the unseen part may be white observation noise
(F-41 work: 30 % SSH / 71 % SST of the one-step change is white) -> nothing non-linear to learn; the M=40 "hole regime"
(linear R^2/C 0.76) has structured unseen modes.
Proposed D1 (measurement, no model): the linear decoder's residual x - Dz (true z), training + validation: (1) white vs
smooth share, correlation length, spectrum; (2) dependence on z (size and regression on z, z^2); (3) rank. Status:
proposed to the user.

### H5 corrected by the user (2026-11-12): "No, the flow matching decoder will find a raw field uncertainty by matching
for whole distribution of z(t)"
NOT p(x | true z) with a null-space correction (the assistant's reading above, withdrawn). The decoder transports the
WHOLE predicted latent distribution N(ẑ_{t+a}, Σ_a) onto the distribution of the raw field that really follows:
  source  x0 = D z, z ~ N(ẑ, Σ) (+ small noise where the eyes see nothing, so x0 is full-dimensional)
  target  x1 = the true raw field x_{t+a}
  flow    v(x_s, s | ẑ, a), trained on OUT-OF-FOLD forecasts (D-045 folds) paired with the true future fields
The raw-field uncertainty is learned against real outcomes (it may correct a latent spread that is wrong in field
space). Members stay coherent across leads (deterministic ODE: one latent member -> one field member per lead).
Fair baseline: linear decoder D on the joint-Gaussian latent members + Gaussian noise for the decoder residual.
Readout: field RMSE by lead, CRPS, spread/error, member energy spectrum vs truth (small scales, H1), probability.
Status: reading proposed to the user for confirmation.

## 12. Worktree cleanup (user 2026-11-12: "remove previous tests, but keep the markdown files"; scope "session tests,
keep folds")
Removed (records above still describe them; recover from this record if needed): `src/models/prior_flow.py`,
`src/train/train_prior_flow.py`, `src/probes/eval_prior_flow.py`, `config/prior_flow.yaml`,
`src/probes/one_step_residual.py`, `src/probes/sight_scenarios.py`, `src/probes/mask_scenarios.py`; the attention-mask
code (`block_mask`, `attn_mask`, `hf.mask`) in `history_forecaster.py` / `train_history_forecaster.py` / config; runs and
evals `.tmps/runs_pf`, `.tmps/eval_pf`, `.tmps/runs_hf/mask_seed*`, `.tmps/eval_hf/{one_step,sight,mask}`.
Kept: every .md record; the cross-fitting option `hf.holdout` (D-045) and the 20 fold runs `.tmps/runs_hf/fold{0..3}_seed{0..4}`
(+ `launch_folds.sh`) = the out-of-fold forecasts H5 trains on; the twin runs.
