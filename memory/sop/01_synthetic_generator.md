# SOP 01 — Synthetic SSH/SST Generator (Milestone 0)

> Golden Rule: update this SOP before changing `src/data/synthetic.py`.
> Realizes D-009 (synthetic testbed) as reframed by D-013 (multi-scale sensors, hidden truth).

## Goal

Produce a deterministic (seeded) synthetic field with **spatially heterogeneous, multi-scale
dynamics**, so that *which region/scale a mask sees* determines the temporal character it reads.
Latent modes are recorded as a **hidden answer key** for post-hoc analysis — they are NEVER fed
to the model (training is unsupervised).

## Output contract

| name | shape | dtype | notes |
|---|---|---|---|
| `field` | `[T=2000, V=2, H=64, W=64]` | float32 | var0=SSH-like, var1=SST-like; each var standardized to ~0 mean, unit var over (t,x,y) |
| `truth` | dict | — | hidden key: per-mode `{scale, family, phi[H,W], amp[T], vars, lag}` + `config` + `seed` |

Reproducible from `(seed, GenConfig)` alone. Smoke test writes to `./.tmps/` as `.npz` (ephemeral).

## Latent process  `field_v(t,x,y) = Σ_k w_{v,k} · a_k(t − lag_{v,k}) · φ_k(x,y) + noise_v`

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

## Three defects this SOP's rev2 fixes (findings F-8 → D-021)

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

## rev3: the stationary mode is CONSTANT (D-025, user ruling 2026-09-01)

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

## Determinism & config

- `numpy.random.default_rng(seed)` only (no global RNG). `GenConfig` dataclass holds all knobs
  (T, grid, counts, periods, τ lag, OU τ, obs_noise, energetic-site centers).
- Log the resolved config + seed into the returned `truth` and into `progress.md` on each run.

## Edge cases / failure modes

- **Lorenz blow-up / NaN** → integrate with small dt + subsample; standardize a_k; assert finite.
- **lag indexing t<τ** → hold the initial value (clamp index at 0), do not wrap.
- **zero-variance var** (all-noise cancels) → guard std by ε before standardizing.
- **dtype drift** → cast final `field` to float32 explicitly; assert shape/dtype at the boundary.

## Verification (Phase L.2 handshake — `src/probes/smoke.py`)

1. `field.shape == (2000,2,64,64)`, `dtype == float32`, all finite.
2. Minimal PROVISIONAL K=16 selection encoder → `S.shape == (2000,16)`, finite.
   (encoder defaults — soft-binary `sigmoid(logit/T)`, full-volume domain, normalize by Σmask —
   are PROVISIONAL plumbing, not the final design; see open points in SOP 00.)
3. One slowness loss `L=mean Σ_i (s[t+1]−s[t])²` → `L.backward()`; assert mask grads finite,
   grad-norm > 0 (gradients actually reach the masks).
4. Log per-var field stats (mean/std/min/max), S stats, L, grad-norm. Success/failure line.

## Visual verification (added 2026-08-27)

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
