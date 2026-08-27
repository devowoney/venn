# SOP 02 — v0 Encoder + Training (slowness + whitening)

> Golden Rule: update this SOP before changing `src/models/encoder.py` or `src/train/train.py`.
> Realizes the settled v0 core (CLAUDE.md): learnable selection masks, slowness loss, whitening
> constraint, online pairwise SGD. Defaults recorded in D-015; all are Hydra-tunable.

## Goal

Learn K selection masks whose K scalar channels are **slow** (temporally coherent) under a
**whitening** constraint (unit variance + decorrelation). Whitening blocks the trivial optima
(M=0 collapse, duplicate kernels); decorrelation is what drives channel diversity (D-013).

## Tensors & call order

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

## v0 defaults (D-015 — all overridable via ./config/)

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
| slowness gap | L2 | L1 |
| optimizer / regime | Adam, pair-minibatch (B random consecutive pairs) | pure single-pair online |

## Reproducibility & I/O (D-014)

- All hyperparameters in `./config/config.yaml` (Hydra). A run = its composed config.
- Run dir under `./.tmps/runs/<timestamp>/`; artifacts + figures written there.
- **ONE TensorBoard dir per run:** both `train` and `evaluate` write into `<run>/tb/`,
  separated by tag namespace (train `loss/*`,`var/*`,`energy/*`; eval `masks/*`,`S/*`,
  `collective/*`). Never a second event dir -- two per run is unreadable for a human.
- Seeds fixed (numpy generator for data, `torch.manual_seed` for model).
- Artifacts (masks, final S, metrics.json) → `.tmps/`. Promotion to `./results/` needs user OK.

## TensorBoard log schema

- train scalars: `loss/total`, `loss/slow`, `loss/white`, `loss/var_hinge`, `loss/energy`,
  `loss/recon`, `loss/size`, `energy/density_{min,max}`, `mask/count_{mean,min,max}`,
  `var/{mean,min,max}`, `slow_per_ch/{min,max}` (emergent slow↔fast spread), `grad_norm`, `temp`.
- eval (SAME `tb/` dir): `masks/trained`, `masks/init` (images), `S/ch*` (the K series),
  `collective/*`.
- Figures (`src/probes/plots.py` → `<run>/figs/`) are the PRIMARY readout, not the scalars:
  `masks.png`, `masks_on_energy.png`, `features.png`, `feature_var.png`. Always look at them.

## Edge cases / failure modes

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

## Verification

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

## Known structural limits (NOT bugs — see D-018, findings F-4/F-5)

- Decorrelation saturates at off-diag corr² ≈ 0.244 / effective rank ≈ 4 of 16 because the masks
  are NON-NEGATIVE and cannot cancel a shared large-scale mode. λ_white is irrelevant (identical
  at 1, 20, 100, 500). Largely a GENERATOR artifact: every `φ_k` is a positive bump, so 100% of
  cell pairs are positively correlated (findings F-4 caveat).
- Minimizing MEAN slowness discards the chaotic modes — FIXED by `lambda_recon` (D-019): chaotic
  ensemble R² 0.01–0.03 → 0.79–1.00. The energy floor alone does NOT fix it (0.01–0.06).
- `lambda_slow` should stay SMALL. At 10 it collapses diversity (eff rank 2.95, spread 31× → 2.3×).
- Masks shrink from the ~50%-of-domain init to a median of ~28 cells (0.3%) unless `lambda_size`
  holds them on the ladder; `L_energy` accelerates the shrinkage because mean pairwise covariance
  is maximized by a tiny coherent patch (D-020).
