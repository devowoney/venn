# SOP 00 — Pipeline Overview (A.N.T. layer A)

> Golden Rule: if logic changes, update this SOP (and the relevant 0x SOP) BEFORE changing code.
> Scope: the whole VENN emulator. Detailed per-stage SOPs live in sibling `0x_*.md` files.

## A.N.T. mapping (per LLMAIProjectInstruction.md Phase A)

- **A — Architecture** (`./architecture/`): these SOPs. Goals, tensor contracts, call order, failure modes.
- **N — Navigation**: which script runs when (this doc's call order). No heavy compute here.
- **T — Tools** (`./src/`): deterministic, testable modules. `data/ models/ train/ eval/`.
  All ephemeral I/O routes through `./tmp/`.

## Modules & call order (D-012)

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

## Kernel semantics (D-013 — read before touching the encoder)

- A mask is a **spatial sensor footprint at some scale/resolution**, NOT an orthogonal unmixer.
  Masks may OVERLAP; a scalar being a mixture of latent modes is fine.
- Masks span many scales: small energetic patch (Gulf-Stream-like jet) ↔ basin-scale footprint
  (equatorial band). Temporal family emerges from *where/what scale* a mask looks.
- **"Deep" = many channels.** Value is a large bank of diverse channels whose *collective* is a
  meaningful encoding. Success = collective encoding quality + channel diversity + emergent
  temporal spread — NOT per-kernel isolation.

## Tensor contract (Milestone 0)

| symbol | shape | dtype | notes |
|---|---|---|---|
| `field` | `[2000, 2, 64, 64]` | float32 | var0 = SSH-like, var1 = SST-like; per-var standardized |
| `S` | `[2000, K]` | float32 | K=16 for the first smoke test; K a config knob afterward |

## File-management & tooling conventions (D-014 — binding)

- **`./.tmps/`** — ALL ephemeral artifacts: script outputs, logs, cached tensors, scratch.
  Safe to delete. (Renamed from `./tmp/`; overrides the framework default.)
- **`./results/`** — curated payload (metrics tables, plots, kept checkpoints). Write **only with
  explicit user authorization**; default outputs go to `./.tmps/`.
- **`./config/`** — Hydra config tree. Every ML hyperparameter lives here; nothing hard-coded. A run
  is fully defined by its composed config (reproducibility).
- **TensorBoard** — training/eval log to TB event files via `torch.utils.tensorboard.SummaryWriter`
  (base env writes; tools env / server views). Log loss components, whitening error, per-channel
  variance (scalars) and masks + scalar series (images).

## Build order (Phase L, current)

1. `01_synthetic_generator.md` → `src/data/synthetic.py` — generate `field` + hidden truth.
2. Minimal PROVISIONAL K=16 encoder (`src/models/encoder.py`) — plumbing only, defaults flagged.
3. Smoke-test probe (`src/probes/smoke.py`) — assert shapes/dtype, one forward + one backward,
   log tensor stats. This is the Phase L.2 handshake.

Open (confirm before the REAL training build, not the smoke test): mask parameterization
(soft-binary anneal vs STE), mask domain (full-volume vs per-layer), `s_i` normalization,
whitening hard vs soft, predictor history window, concrete collective metric, deeper K.
