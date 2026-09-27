# Loss guide — what every term asks for, in plain words

Companion to `02_encoder_training.md` (the SOP, which holds the formulas' history, measurements and
decisions). This file is the **plain-language reference**: what each term asks the encoder for, why it
exists, when it is silent, and what it cannot do. Code: `src/train/train.py` and
`src/train/spectral.py`. Weights: `config/config.yaml`. State as of 2026-09-27 (D-027).

---

## 1. Vocabulary

| word | meaning |
|---|---|
| **field** | the synthetic ocean: `[T=2000, V=2 (SSH, SST), 64, 64]`, standardized to std 1 |
| **mask / kernel** | one learned map in [0, 1] over the grid (both variables). Bright = selected cells |
| **channel** `s_i(t)` | what mask *i* reads: `s_i(t) = <mask_i, field_t>`, the weighted sum of the selected cells. One number per time step. K = 16 channels |
| **observer / encoder** | the bank of K masks. It decides **where to look**; it never filters or transforms the signal |
| **count** | a mask's total weight = its footprint size in cells |
| **family** | what kind of signal a channel carries: **stationary** (does not move: constant), **cyclic** (moves with structure: an oscillation or a slow drift), **chaotic** (moves broadband: no clean rhythm) |
| **rung** | a channel's **assigned slot** on a ladder: a pre-set target the channel is asked to meet. The name comes from ladder steps — rungs are ordered, each one a step up or down in size or timescale |
| **role** | which family a rung is asked to produce. K = 16 → **2 stationary, 5 cyclic, 9 fast (chaotic)** rungs, in the 1/3/6 proportion. Channel order: ch0–1 stationary, ch2–6 cyclic, ch7–15 fast |
| **size ladder** | each channel gets a target footprint, from basin-scale (ch0, ~30% of the domain) down to a small patch (ch15, ~16 cells) |
| **band ladder** | each channel gets a target frequency band: stationary rungs slow, fast rungs fast, and the 5 cyclic rungs one **octave** each, tiling periods 320 → 24 steps (ch2 longest, ch6 shortest) |
| **octave** | a band of periods where the longest is about twice the shortest |
| **hinge** | a penalty that is exactly 0 once a target is met, and grows only on the wrong side: `relu(target − x)²`. Hinges **guide** a channel into a zone; they do not pin it to a value |
| **scale-free** | written as a ratio, so multiplying a mask by any number changes neither the loss nor its gradient. Mandatory here: every non-scale-free version made the masks shrink to zero (D-017) |
| **gradient-matched weight** | each λ is set so that, at initialization, that term's gradient is as strong as `L_white`'s. Loss *values* differ by ~300× and would mislead (D-019) |
| **baseline** | the configuration before the change being tested, run on the same seeds |
| **seed** | fixes all randomness of a run: the synthetic field itself, the initial masks, the sampled time steps. Seeds 0–4 = five different random oceans |
| **ambiguous channel** | structure share between 0.35 and 0.65, i.e. the labeller's cyclic/chaotic vote is close to a coin flip (the 0.30 margin is a convenience, not calibrated) |

---

## 2. The big picture: three kinds of question

The loss has **11 terms** (9 active by default). They answer three different questions:

1. **Is the bank usable at all?** Channels alive, on signal, distinct, covering the field, at a
   spread of sizes. → `L_slow`, `L_white`, `l_var`, `l_energy`, `l_size`, `l_recon`.
2. **Which timescale does each channel watch?** → `L_band` (where in frequency).
3. **Which family does each channel become?** → `L_level` (does it move at all?),
   `L_line` (cyclic rungs: is it one clean oscillation?), `L_struct` (fast rungs: is it broadband?).

One sentence to remember: **`L_band` decides where a channel looks in frequency, `L_line` / `L_struct`
decide what shape of signal it finds there, and `L_level` decides whether it moves at all.**

Why group 1 alone cannot produce families: `L_slow` gives every channel the same instruction and
`L_white` only makes them different in *place*, not in *dynamics*. The K sweep proved it: 4 to 64
channels, always slow-dominated (F-12). So the families are **assigned** by the ladder, and the
labeller checks afterwards whether the assignment took (`role_obedience`).

---

## 3. Group 1 — a usable bank

### `L_slow` — "change slowly"  (λ = 1.0)

```
L_slow = mean_i  var(s_i(t+1) − s_i(t)) / var(s_i)
```

- **Plain words:** how much a channel changes from one step to the next, relative to how much it
  varies overall. Small = slow. Equals `2(1 − lag-1 autocorrelation)`: 0 for perfectly smooth, 2 for
  white noise (period-300 sinusoid ≈ 0.0004, period 60 ≈ 0.011, chaotic ≈ 0.36).
- **Why a ratio:** `var(Δs)` alone is won by shrinking the mask. Dividing by `var(s)` removes scale.
- **Limits:** one instruction shared by all channels, so they all chase the same slowest content.
  Blind to a constant (it only sees fluctuation). Weak in practice: ~24× smaller gradient than
  `L_white` at init. Kept as a tiebreak; removing it measured slightly worse.
- This is Slow Feature Analysis's objective; `L_white` is its constraint.

### `L_white` — "don't copy another channel"  (λ = 1.0, the reference weight)

```
L_white = mean over pairs i≠j of corr(s_i, s_j)²
```

- **Plain words:** the average squared correlation between channels. 0 = all unrelated, 1 = all
  copies. Trained runs end ≈ 0.06 (typical |corr| ≈ 0.25).
- **Why correlation, not covariance:** covariance is won by shrinking masks; a detached whitening
  makes redundancy free. Both failed (the `soft` / `hard` ablations, D-017).
- **Limits:** makes channels different, not different *kinds*. Compares values at the same moment
  only (a delayed copy is not penalized). With non-negative masks, overlapping footprints correlate,
  so it pushes masks apart and small — one reason `l_size` exists.

### `l_var` — "don't fade to nothing"  (λ = 1.0)

```
l_var = mean_i relu(1 − std(s_i) / sqrt(count_i))²
```

- **Plain words:** a floor on each channel's fluctuation. `std/sqrt(count)` ≈ 1 for a mask over
  unrelated cells, larger for cells moving together, → 0 as the mask fades.
- **Silent** for every live channel; acts only on a dying one (VICReg's variance term).
- **Exempt:** stationary rungs (they are meant not to fluctuate).
- **Limits:** keeps a channel alive, not useful — it can sit in a quiet corner (F-6). Leans slightly
  against channels built from cancelling parts.

### `l_energy` — "look where the field moves"  (λ = 0.2)

```
e_i = var(s_i) / count_i²                      average covariance between the selected cells
l_energy = mean_i relu(1 − e_i / e_ref)²       e_ref = 0.371, the field's mean per-cell temporal variance
```

- **Plain words:** the selected cells must be energetic **and** move together, at least as much as an
  average single cell varies. Independent of footprint size.
- **Not a quiet safety net:** active through most of training; at convergence every dynamic channel
  sits right at the floor (0.93–1.28).
- **Exempt:** stationary rungs.
- **Side effect:** the cheapest way to score high is a tiny coherent patch, so it pulls masks small;
  `l_size` holds it back.

### `l_size` — "keep your assigned footprint"  (λ = 0.3)

```
l_size = mean_i relu(|log(count_i / target_i)| − log 2)²
```

- **Plain words:** each channel has a target footprint on the size ladder (ch0 ≈ 30% of the domain →
  ch15 ≈ 16 cells). Free within a factor 2 of the target, penalized outside.
- **Why:** left alone, every mask shrank to a ~28-cell patch, and the "multi-scale sensor" became 16
  small patches (D-020).

### `l_recon` — "together, cover the whole field"  (λ = 0.3)

```
X      = field at 128 random moments, each cell's time mean removed          [128, 8192]
Wd     = best linear decoder of X from [1, standardized channels], solved exactly per batch
l_recon = unexplained fraction of X  =  1 − R²
```

- **Plain words:** can the 16 channel values, combined linearly, rebuild the field's motion?
  Collective (one number for the bank), spatial, ignores time order.
- **Also penalizes redundancy:** a duplicate channel explains nothing new.
- **Values:** ≈ 0.006 in training (energy-weighted R² ≈ 0.994). The probe's "balanced" R², where
  every cell counts equally, is ≈ 0.78 — the more demanding number.
- **Limits:** blind to the static pattern (means are removed), so stationary channels earn nothing
  here.

---

## 4. Group 2 — which timescale

### `L_band` — "put your power in your assigned band"  (λ = 5.0)

```
bandfrac_i = power inside rung i's band / total power      (periodogram over 4 random 512-step windows)
L_band     = mean over dynamic rungs of relu(0.5 − bandfrac_i)²
```

- **Plain words:** at least half of a channel's fluctuation must sit in its assigned frequency band.
  Fast rungs: periods ≤ 24. Cyclic rungs: one octave each — ch2 171–512, ch3 102–256, ch4 64–128,
  ch5 39–73, ch6 23–43 steps (window-bin edges; the design range is 320 → 24).
- **Why:** this is what makes channels slow or fast. `L_slow` cannot: it asks the same of everyone.
- **Limits:** the families overlap in frequency on this testbed (cycles at 60 / 140 / 300), so a band
  alone cannot tell a slow cycle from a slow drift — that is the shape terms' job. It holds cyclic
  rungs weakly: ch6's octave contains no cycle at all, and ch4's sits just above the period-60 cycle,
  so both end on period 60 in every run. 5 cyclic rungs share 3 cycles.
- **Exempt:** stationary rungs (a constant has all its power in the mean, which the spectrum drops).

---

## 5. Group 3 — which family

### `L_level` — stationary rungs: "don't move"  (λ = 0.13)

```
r_i     = |mean(s_i)| / (|mean(s_i)| + std(s_i))       1 = perfectly constant, 0 = pure fluctuation
L_level = mean over stationary rungs of relu(0.95 − r_i)²
```

- **Plain words:** the channel's level must dwarf its fluctuation. `r ≥ 0.95` is exactly the
  labeller's flat cut `amp_ratio = std/|mean| < 0.05`.
- **Why it must exist:** a constant lives entirely in the channel's **mean**, and every other term
  works on the centred channel. Worse, `l_var` and `l_energy` actively forbid a flat channel — hence
  the exemptions.
- **Hard part:** the static pattern is spatially zero-mean, so a non-negative mask must align with
  one signed lobe of it; a random mask of the same size is nowhere near flat (amp_ratio ≈ 66).
- **Known weakness:** 0.95 sits exactly on the readout's cut, leaving no margin; flat channels per
  seed are fragile (0–2). `flat_target 0.98` measured 0.8 → 1.8 per seed — **undecided (user)**.

### `L_line` — cyclic rungs: "be one clean oscillation"  (λ = 4.0)

```
linefrac_i = power within ±2 bins of the spectral peak / total power    (512-step windows)
L_line     = sum over cyclic rungs of relu(0.85 − linefrac_i)²  /  number of dynamic rungs (14)
```

- **Plain words:** at least 85% of the power in one narrow peak. The hidden cyclic modes measure 0.94.
- **Why divided by 14 and not 5:** each cyclic rung must keep the weight it had when `L_line` also
  covered the fast rungs. Dividing by 5 made it 2.8× stronger per rung, and it then pinned the cyclic
  rungs on the period-60 cycle they start on, beating `L_band` in the first ~20 steps (seed-0
  diagnosis, 2026-09-27).
- **Limits:** it credits whichever peak is cleanest, regardless of the rung's octave. `L_band` is the
  only thing that says *which* cycle.

### `L_struct` — fast rungs: "be broadband"  (λ = 2.3)

```
spectrum of the FULL series (2000 steps), mean removed, split into:
  trend    = bins 1–3 (periods ≥ 667: fewer than 4 cycles in the record — drift)
  osc      = the strongest peak above bin 4 and its neighbours down to half its power (≥ ±2 bins)
  residual = everything else — broadband
struct_i = (trend_i + osc_i) / total_i          = the labeller's own "cyclic_share"
L_struct = mean over fast rungs of relu(struct_i − 0.30)²
```

- **Plain words:** at most 30% of a fast channel's power may be drift or one dominant rhythm; at least
  70% must be broadband. The cap is the physics' own bar: the hidden chaotic modes measure 0.17–0.35.
- **It never filters the signal.** The gradient moves the mask toward cells whose signal is broadband.
- **Why it exists:** the labeller calls a channel cyclic if `struct ≥ 0.5`. The old fast-rung cap
  (`L_line` at 0.75 on a ±2-bin window) measured a different quantity and was silent on all nine fast
  rungs while six sat at the coin-flip boundary. The objective must measure what the readout measures.
- **Why fast rungs only:** on cyclic rungs, "at least 0.85 structured" credits any clean peak, and all
  five moved to the period-60 cycle (F-16).
- **Limits:** slow power that is neither the peak nor the very slowest drift counts as "broadband", so
  fast rungs can still carry 30–40% of their power at periods ≥ 100.

### Ablation-only terms (off by default)

- **`L_mem`** (λ 0.4, `slow_objective: memory`): stationary rungs keep a long-lag autocorrelation.
  It encodes the old reading of "stationary" as a slow drift, superseded by D-025.
- **`L_line` on fast rungs** (`shape_objective: line`): the D-024 original, capping line fraction at
  0.75. Silent in practice.

---

## 6. Who is exempt from what

| term | stationary rungs (ch0–1) | cyclic rungs (ch2–6) | fast rungs (ch7–15) |
|---|---|---|---|
| `L_slow`, `L_white`, `l_size`, `l_recon` | yes | yes | yes |
| `l_var`, `l_energy` | **exempt** | yes | yes |
| `L_band` | **exempt** | yes (own octave) | yes (periods ≤ 24) |
| `L_level` | **yes** | — | — |
| `L_line` | — | **yes** | — |
| `L_struct` | — | — | **yes** |

---

## 7. Common properties

- **Every ratio is scale-free** in value and gradient: a mask cannot win by shrinking or growing.
- **Every ladder term is a one-sided hinge:** silent once satisfied, so it guides rather than pins.
- **Weights are gradient-matched at init** to `L_white` (D-019), then corrected by measurement where
  needed.
- **The objective and the readout must agree** on both the quantity and the threshold (F-13, F-15);
  otherwise a term goes quiet before the labeller is satisfied.
- **Never judge a run by one metric.** The ambiguity count looked good while all cyclic rungs sat on
  one cycle (F-16); the kernel and signal figures (`src/probes/plots.py`) showed it.
