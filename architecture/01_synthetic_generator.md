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

| family | count | scale | spatial φ_k | temporal a_k(t) |
|---|---|---|---|---|
| stationary | 1 | LARGE | elongated equatorial band (broad Gaussian stripe) | Ornstein–Uhlenbeck, long τ (slow drift) |
| cyclic | 3 | MEDIUM | Gaussian blobs σ≈8–12 at fixed centers | sinusoids, periods P≈[60,140,300] + small noise |
| chaotic | 6 | SMALL | tight Gaussian patches σ≈2–4 at "energetic" sites | 2× Lorenz (σ=10,ρ=28,β=8/3) → x,y,z = 6 series |

- **φ_k** normalized to max |φ|=1. **a_k** standardized to unit variance (guards Lorenz scale).
- **SSH/SST coupling:** SST (var1) shares the LARGE+MEDIUM modes with SSH (var0) but **lagged by
  τ** (`lag_{1,k}>0`) and with a spatially smoothed φ; each var also keeps some private weight on
  SMALL modes. Encoded per-mode via `w_{v,k}` and `lag_{v,k}`.
- **noise_v:** small additive Gaussian (obs noise), std = `obs_noise` × field std.

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
