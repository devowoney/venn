# Findings — F-11 … F-14

Numbering continues `memory/findings.md`. Read F-13 before trusting any population number in F-11
or F-12: those were measured under a definition of "stationary" the user later corrected.

---

## F-11. ⚠⚠ The readout could not say "stationary" — every population number was void

**The bug.** `evaluate.py`'s labeller called a series **cyclic** whenever its single largest FFT bin
held more than `peak_thr` (0.10) of the power. A red / memory-dominated spectrum always piles its
power in the LOWEST resolvable bin, so a slow process was reported as "a clean oscillation of period
T/2". The label "stationary" was unreachable for anything red — which is every slow series there is.

**The proof was already in every `eval.json` we had ever written.** Applied to the GENERATOR'S OWN
stationary amplitude series, the labeller returned `cyclic` (`truth[0].label`), and
`labeller_truth_accuracy` read **0.7**. It sat there at 0.7 for five sessions and neither of us
treated it as a blocker. Every `population.stationary = 0` recorded in this project (D-018, F-5, the
2026-08-29 module-2 entry) measured the probe, not the encoder.

**What the encoder had actually been doing all along:** capturing the slow mode well.
`mode_recovery.m0.max_abs_corr` = 0.92 (K=4), 0.62 (K=8), 0.93 (K=16), 0.82 (K=32), ensemble R² ≈
1.00 throughout.

**Lesson — third instance, after F-6 (figures caught the dead corners) and F-9 (`var_ratio` caught
the mean-collapse ranking):** a metric that CANNOT emit an outcome is worse than a missing metric,
because it reads as evidence against the model. **Validate a readout against known ground truth
BEFORE using it to judge a model.** `src/probes/validate_labeller.py` now makes this executable.

---

## F-12. K is NOT the lever on the family population

Re-scoring the archived K sweep (`.tmps/runs/20260831_0736*`) with a labeller that could at least
emit a slow label:

| K | old (broken) readout | re-scored |
|---|---|---|
| 4 | 0/4/0 | 3/0/1 |
| 8 | 0/8/0 | 4/2/2 |
| 16 | 0/13/3 | 12/0/4 |
| 32 | 0/30/2 | 22/3/7 |
| 64 | — | 36/10/18 |

*(Read these as slow/cyclic/fast — see F-13.)*

**The sign of the problem was the opposite of what we believed.** The bank does not under-produce
slow channels; it produces almost nothing else (12 of 16 red at K=16, ZERO clean cyclic). F-5's
mean-slowness pathology had been hiding behind the F-11 labeller.

**Mechanism.** `L_slow = mean_i var(Δs_i)/var(s_i)` is ONE objective shared by all K channels. They
compete for the same globally-slowest content, so channel K+1 buys a near-duplicate of the dominant
regime rather than a new timescale. The proportions barely move from K=4 to K=64. This confirms the
user's read that deepening the kernel bank raises slow-mode visibility somewhat but cannot reach
every dynamical family. The fix is to ASSIGN the timescale per channel (D-024).

---

## F-13. STATIONARY MEANS CONSTANT — my F-11/F-12 taxonomy was wrong, and so were its populations

**User's correction:** *"when we decided 'stationary signal', I refered that some constant signal.
Actually, your injected mode 0 in pseudo SSH-SST field, it is not stationary for me. it should be
more flat."*

**Correct, and the root cause is in the GENERATOR.** `_ou_series` ends with `_standardize`, so the
"stationary" mode was emitted at UNIT VARIANCE — exactly as much temporal energy as the sinusoids and
the Lorenz modes. Mode 0 was a τ=200 wanderer, and **no flat mode existed anywhere in the testbed**
for the encoder to find. My F-11 labeller then defined stationary as "long memory, no line", which is
a definition of slow red noise, so it labelled 10 of 16 baseline channels stationary. Those counts
describe my taxonomy, not the project's.

**Verified after generator rev3:** mode-0 amp mean 1.000 / std `0.000e+00`; the field's time-mean map
correlates **1.000** with the mode-0 pattern at rms 0.818 against a field std of 1.0 — a strong,
cleanly identifiable static component. Labeller 10/10 on seeds 0–4, population 1/3/6.

**Why the encoder could never have found it.** See D-025: every loss term uses the centered channel,
and `l_var` + `l_energy` actively penalize a flat channel.

**Measured on the rev3 field (K=16, 5000 steps, seed 0):**

| | baseline `20260901_080438` | ladder + `L_level` `20260901_080506` |
|---|---|---|
| population (s/c/ch) | 0/14/2 | **2/5/9** |
| role obedience | n/a | **1.00** |
| flattest channel `amp_ratio` | 0.778 — *still moving* | **0.023** (second at 0.049) |
| flat channels (< 0.05) | **0** | **2** |
| balanced recon R² | 0.7763 | 0.7765 |

The baseline provably CANNOT produce a stationary observer — nothing in it gets near flat. (rev3
scores 0.78 balanced recon where rev2 scored 0.99: the new field has a large static component that
anomaly-based decoding cannot explain, and it hits both runs equally. Not a regression.)

**A figure was lying, again (cf. F-6).** `plots.py` standardized every channel before plotting —
dividing out the std, the very quantity that defines flatness. A channel fluctuating at 2% of its
level was drawn as dynamic as a sinusoid, which is how the ladder run first read as "ch0/ch1 are
oscillating". Now scaled by RMS about zero with `amp_ratio` printed per lane. **Check what a plot
normalizes by before trusting it.**

---

## F-14. Classify on GLOBAL STRUCTURE; and the observer is not a mode-recovery device

**User:** *"we should classifier with there global strcutre not their small fluctuation and drifts"*
and *"Hidden mode is just hidden mode. […] the Observer should decide where we should see where we
should focus on. That becomes learned kernel. if the kernel captures stationary signal, it could be a
single mode […] But, it also can be a persist phenomen or the combination of multi signals which
shows barely varying value."*

**(a) Structure, not residue.** My ACF-recurrence test read the RAW series, where fast noise and a
wandering baseline dominate the autocorrelation — so it answered a question about the residue and
called visibly oscillating channels non-cyclic. Replaced by the 4-part decomposition (D-026).

Re-scored post-hoc — a label is a readout, so no retraining is needed to change one:

| | truth accuracy | population |
|---|---|---|
| rev3 truth modes | **10/10** | 1/3/6 |
| rev2 truth modes | 9/10 — the "miss" is the old OU mode 0 now labelled **cyclic** (trend share 0.63), i.e. D-025 working as ruled | — |
| rev3 ladder `20260901_080506` | — | 2/5/9, role obedience **1.00** |
| rev3 baseline `20260901_080438` | — | 0/11/5 |

**On the channels the user named:** ch2–6 come out **cyclic** (osc share 0.70–0.93), agreeing with
him. ch9 and ch11 stay **chaotic** but with osc shares **0.41** and **0.32** against residual 0.59
and 0.67 — real oscillatory character, broadband still dominant. That is the honest reading of "they
have cyclic characteristics but are not 100% cyclic"; forcing them over would contradict the
Lorenz-derived mode (m6) they track and would cost truth accuracy.

**(b) The family belongs to the OBSERVABLE.** This retired a metric I had introduced the same day: I
had been reading the flat channel's |corr| 0.27 with the injected φ₀ as a weakness, and had even run
an experiment to "fix" it. Under the correct frame it is a diagnostic. See D-026 for the full
scoring consequences.

**New diagnostic answering the "or a combination" case.** Project every hidden mode onto the flat
channel's footprint and compare the net fluctuation against the independent-addition baseline
`sqrt(Σ sd_k²)`:

```
< 0.7   destructive interference beyond chance   -> a cancelling COMBINATION
~ 1.0   incoherent addition                      -> a quiet footprint / persistent phenomenon
> 1.3   the contributions reinforce
```

Ladder run ch1: 3 hidden modes reach the footprint, index **1.01** → this stationary observer is the
*persistent-region* case, not cancellation.

**Mistake caught inside this metric.** My first version normalized by the plain SUM of the parts and
read 0.634, which looks like partial cancellation — but three independent parts already give 0.577 of
the sum, so 0.634 is almost exactly what chance predicts. **The wrong baseline would have
manufactured a finding.**

---

## F-15. The fast rungs were ambiguous because no loss term opposed it

Run `20260901_080506`: 7 of 16 channels had structure share in 0.35–0.65, all fast rungs (0.33–0.44).
`line_cap` (±2-bin line fraction ≤ 0.75) was silent on all nine fast rungs (0.15–0.45). The hidden
chaotic modes score 0.17–0.35 and two fast channels (ch7, ch10) reached 0.16, so broadband channels
were achievable. **The objective and the readout must measure the same QUANTITY, not only agree on a
threshold** — the F-13 lesson in a harder form.

Stationary side, same run: a random mask of the same 2590-cell footprint gives amp_ratio ≈ 66, the
learned one 0.023–0.049. Flatness is a real achievement, not free averaging (my hypothesis, refuted).
It is hard because the static component is spatially zero-mean (time-mean map rms 0.795, global mean
0.0000), so a non-negative mask must align with one signed lobe. It is also fragile across seeds:
baseline flat channels per seed 2/0/1/0/1 — `flat_target 0.95` sits exactly on the 0.05 cut.

## F-16. A good ambiguity count hid a collapse of the cyclic timescales

`L_struct` on all rungs scored 3.0 ambiguous channels (baseline 6.8), but the kernel/signal figures
showed every cyclic rung on the same cycle: dominant periods 61/61/61/61/61 on seed 0 vs the
baseline's 286/143/61/61/61. `trend+osc` credits any clean peak, and period 60 is the cleanest line.
The `slowness_spread` drop (88 → 16) flagged earlier was this collapse. Fast-only (D-027) cuts
ambiguity to 1.2 but still averages only 1.4 distinct periods (baseline 2.4); seed 0 still collapses.
Cause open. **A metric about one failure mode says nothing about the others — look at the figure.**
