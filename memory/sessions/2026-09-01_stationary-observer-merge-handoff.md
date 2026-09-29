# Handoff — `worktree-stationary-observer` (2026-08-31 → 2026-09-01)

> Saved here on 2026-09-28 from `architecture/README.md` when `architecture/` was retired and merged into
> `memory/`. The file table below is updated to the new `memory/` paths; the rest is as written at merge time.

Self-contained record of this session's decisions, findings and measurements, written for the
**merge back into `main`**.

## Why these files exist here and not in `memory/`

*(Historical — as written 2026-09-01.)* `.gitignore` ignores both `memory/` and `.claude/`, so the living
memory does NOT travel with a branch. Everything needed to review and merge this work is therefore
duplicated here, in a tracked directory. `memory/findings.md`, `memory/decisions.md`, `memory/progress.md`
and `memory/sessions/2026-08-31_stationary-observer-and-timescale-ladder.md` in the MAIN checkout carry
the same content (they are shared across worktrees on disk).

**Update 2026-09-28:** `memory/` is now the tracked home of the project record; `architecture/` is retired
and its contents live at the paths below.

| file (now) | was | contents |
|---|---|---|
| `memory/decisions.md` | `01_decisions.md` → `01a_decisions.md` | D-024, D-025, D-026 — full text, with what each supersedes |
| `memory/findings.md` | `02_findings.md` → `01b_findings.md` | F-11 … F-14 — what was measured and what it invalidated |
| `memory/measurements.md` | `03_measurements.md` → `01c_measurements.md` | every number, with the run ID that produced it |
| `memory/open_and_rejected.md` | `04_open_and_rejected.md` → `01d_…` | open questions, plus alternatives tested and rejected |
| `memory/sessions/2026-08-31_stationary-observer-and-timescale-ladder.md` | `05_session_narrative.md` → `01e_…` | the story |

(The `04` row's original text read "open questions, plus alternatives already tested and rejected (do not
re-try)"; the `05` row's read "the chronological story, including dead ends and my own corrections".)

## Code changed on this branch

| path | status | what |
|---|---|---|
| `src/probes/family.py` | **new** | the family labeller: `amp_ratio` flat gate + 4-part structural decomposition. Single source of truth for a family label. |
| `src/train/spectral.py` | **new** | spectral band ladder (`band_plan`, `spectral_terms`), `level_term` (flatness), `memory_term` (ablation) |
| `src/probes/validate_labeller.py` | **new** | validates the labeller against the hidden truth over several seeds, and re-scores archived runs. Run this FIRST after any labeller change (see F-11). |
| `src/data/synthetic.py` | modified | generator rev3: `stationary_constant` — a flat, un-standardized mode 0, plus the SST bypass for constant modes |
| `src/train/train.py` | modified | ladder wiring, per-rung exemptions from `l_var`/`l_energy`/`L_band`/`L_line`, `roles` saved to artifacts |
| `src/probes/evaluate.py` | modified | imports the labeller, prints structure shares, `stationary_observer` readout, constant-mode guards, design-vs-recovery wording |
| `src/probes/plots.py` | modified | features panel scaled by RMS (was standardized, which hid flatness) + `amp_ratio` per lane |
| `config/config.yaml` | modified | `data.stationary_constant`, the whole `train.spectral` block, all weights gradient-matched |
| `architecture/01_synthetic_generator.md` (now `memory/sop/01_synthetic_generator.md`) | modified | rev3 section |
| `architecture/02_encoder_training.md` (now `memory/sop/02_encoder_training.md`) | modified | D-024/D-025/D-026 sections |

## Merge checklist

1. **Nothing is committed.** Git is the user's in this repo; the branch holds working-tree changes
   only. Commit or cherry-pick as preferred.
2. **Back-compatibility is preserved by dataclass defaults.** `GenConfig.stationary_constant`
   defaults to `False`, so every pre-2026-09-01 run regenerates its original field from its own
   saved config. The new value lives in `config/config.yaml`. `evaluate.py` depends on this.
3. **Re-scoring old runs changes their reported labels**, by design (D-025/D-026). Any figure or
   number quoted from before 2026-08-31 should be re-read with `validate_labeller.py`.
4. **`.gitignore` has unresolved merge-conflict markers** (`<<<<<<< HEAD`, `=======`,
   `>>>>>>> 7afcabf`). Pre-existing, harmless today, worth cleaning during the merge.
5. Run order to reproduce:
   ```
   conda run -n oceanai python -m src.probes.validate_labeller     # readout first
   conda run -n oceanai python -m src.train.train                  # ladder on by default
   conda run -n oceanai python -m src.probes.evaluate
   conda run -n oceanai python -m src.probes.plots
   ```

## The one-paragraph summary

The encoder was never failing to see slow structure; the READOUT could not emit the label, and the
testbed's "stationary" mode was not stationary (it was standardized to unit variance, making it a
drift). Fixing the definition end-to-end — a constant mode in the generator, a flatness-gated
structural labeller, and an `L_level` term that makes a flat channel reachable instead of forbidden —
turns a bank that captured no stationary observable at all (baseline: flattest channel `amp_ratio`
0.778) into one where a dedicated channel sits flat at `amp_ratio` 0.023 and every channel lands in
its assigned family (`role_obedience` 1.00). The deeper reframing: **K is not the lever on which
dynamical families appear** (measured flat from K=4 to K=64) — the timescale must be ASSIGNED per
channel — and **families are properties of the observables the kernels build, not of the hidden
modes**, so mode recovery is a diagnostic, not the goal.
