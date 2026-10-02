"""Post-hoc readout of a trained VENN run (SOP 02 "Verification"; D-013 collective metric).

Reads a Hydra run dir (default: the newest under ./.tmps/runs/), regenerates the exact field from
that run's saved config, and reports what the K channels actually became:

  per channel  : mask footprint size / binarity / spatial scale, normalized lag-1 slowness,
                 decorrelation time, spectral peak, heuristic family label, best-matching
                 hidden mode (post-hoc only -- the answer key is never used for training)
  collective   : family population vs the 1/3/6 target, channel redundancy (effective rank),
                 mask overlap, and LINEAR RECONSTRUCTION R^2 of the full field from the K
                 channels -- the concrete "collective encoding quality" number D-013 asks for,
                 reported against the untrained (init) masks as a baseline.

Writes eval.json + TensorBoard images/series into the run dir. Base env only writes TB event
files; viewing is the tools env's job (D-003).

Run:  conda run -n oceanai python -m src.probes.evaluate [--run .tmps/runs/<ts>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.data.synthetic import GenConfig, generate_field, normalize_input
from src.probes.family import FAMILIES, amp_ratio, label_family, series_stats

# NOTE: `series_stats` / `label_family` used to live here. They moved to src/probes/family.py after
# finding F-11: the old rule called any series whose largest FFT bin held >peak_thr of the power
# "cyclic", which a red spectrum always satisfies at its lowest bin -- so the readout could not emit
# the label "stationary" AT ALL, not even for the generator's own stationary mode. Everything that
# reads a family label must go through that one module now, so the two cannot drift again.


# --------------------------------------------------------------------------- series descriptors
def _std(a: np.ndarray) -> np.ndarray:
    """Standardize along time (axis 0 of a [T] or [T,K] array)."""
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


# ----------------------------------------------------------------------------- mask descriptors
def mask_stats(m: np.ndarray) -> dict:
    """Descriptors of ONE mask [V,H,W] in [0,1]: size, binarity, spatial extent."""
    tot = float(m.sum())
    binarity = float((4.0 * m * (1.0 - m)).mean())         # 0 = hard binary, 1 = all 0.5
    w = m.sum(0)                                           # [H,W] collapse the variable axis
    s = w.sum() + 1e-12
    H, W = w.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = float((w * yy).sum() / s), float((w * xx).sum() / s)
    ry = float(np.sqrt((w * (yy - cy) ** 2).sum() / s))    # radius of gyration = footprint scale
    rx = float(np.sqrt((w * (xx - cx) ** 2).sum() / s))
    return dict(count=tot, frac=tot / m.size, binarity=binarity,
                cy=cy, cx=cx, r_gyr=float(np.sqrt(ry * rx)))


# --------------------------------------------------------------------------------- collective
def _r2(Y: torch.Tensor, X: torch.Tensor) -> float:
    """Pooled R^2 of the least-squares fit Y ~ X (X must already carry an intercept column)."""
    beta = torch.linalg.lstsq(X, Y).solution
    sse = ((Y - X @ beta) ** 2).sum()
    sst = ((Y - Y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - sse / sst)


def recon_r2(field: np.ndarray, S: np.ndarray, device: str = "cpu") -> tuple[float, float]:
    """Linear-decode R^2 of the FIELD from the K channels -- energy-weighted and balanced.

    `raw` weights each cell by its variance, so it is dominated by the single large-scale mode
    (a basin-wide band carries most of the field's energy); `bal` standardizes every cell first,
    so a small energetic patch counts as much as the band. `bal` is the discriminating number.
    """
    T = field.shape[0]
    Y = torch.from_numpy(field.reshape(T, -1)).to(device).double()
    X = torch.from_numpy(np.c_[np.ones(T), _std(S)]).to(device).double()
    Yb = (Y - Y.mean(0, keepdim=True)) / (Y.std(0, keepdim=True) + 1e-12)
    return _r2(Y, X), _r2(Yb, X)


def mode_r2(amp: np.ndarray, S: np.ndarray, device: str = "cpu") -> list[float]:
    """Per hidden mode: R^2 of that mode's amplitude regressed on ALL K channels jointly.

    Stronger than max |corr| -- it asks whether the ENSEMBLE encodes the mode (D-013: quality is
    a property of the collection), not whether any single channel isolated it.
    """
    T = S.shape[0]
    X = torch.from_numpy(np.c_[np.ones(T), _std(S)]).to(device).double()
    out = []
    for a in amp:
        if a.std() < 1e-12:
            # A CONSTANT mode has zero variance: `_std` would divide by ~0, and a regression with an
            # intercept "explains" any constant perfectly, so R^2 is degenerate either way. Report
            # None and judge the stationary mode by `stationary_capture` instead (D-025).
            out.append(None)
        else:
            out.append(_r2(torch.from_numpy(_std(a)[:, None]).to(device).double(), X))
    return out


def eff_rank(S: np.ndarray) -> float:
    """Effective rank of the channel set = exp(entropy of the correlation eigenspectrum)."""
    C = np.corrcoef(_std(S).T)
    ev = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_pairwise(masks: np.ndarray) -> tuple[float, float]:
    """(mean |cosine| between mask vectors, mean IoU of the 0.5-thresholded masks)."""
    K = masks.shape[0]
    M = masks.reshape(K, -1)
    n = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    cos = n @ n.T
    B = M > 0.5
    inter = (B.astype(np.float32) @ B.astype(np.float32).T)
    size = B.sum(1)
    union = size[:, None] + size[None, :] - inter
    iu = np.triu_indices(K, 1)
    return float(np.abs(cos)[iu].mean()), float((inter / np.maximum(union, 1))[iu].mean())


# --------------------------------------------------------------------------------------- main
_ROLE2FAM = dict(slow="stationary", cyclic="cyclic", fast="chaotic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest .tmps/runs/*)")
    ap.add_argument("--line-thr", type=float, default=None,
                    help="line_frac above which a channel counts as cyclic (default family.py)")
    ap.add_argument("--tau-thr", type=int, default=None,
                    help="tau_e above which a non-cyclic channel counts as stationary")
    args = ap.parse_args()
    lab_kw = {k: v for k, v in dict(line_thr=args.line_thr, tau_thr=args.tau_thr).items()
              if v is not None}

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run = args.run or sorted(glob.glob(os.path.join(repo, ".tmps/runs/*")))[-1]
    run = os.path.abspath(run)
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    S, masks, masks_init = art["S"], art["masks"], art["masks_init"]
    # per-channel ASSIGNED role from the D-024 band ladder; absent in pre-ladder runs
    roles = [str(r) for r in art["roles"]] if "roles" in art.files else []
    amp, fams = art["truth_amp"], list(art["families"])
    K, T = S.shape[1], S.shape[0]
    field, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)),
                              seed=int(cfg.seed))

    print(f"[eval] run={os.path.relpath(run, repo)}  K={K} T={T} temp_final={float(art['temp_final']):.3f}")

    # --- reference: what the HIDDEN modes look like under the same descriptors ----------------
    print("\n  hidden ground-truth modes (answer key -- never seen by training)")
    print(f"  {'mode':>4} {'family':>10} {'amp_r':>9} {'trend':>6} {'osc':>5} {'resid':>6} "
          f"{'label':>10}")
    truth_rows = []
    for i in range(amp.shape[0]):
        st = series_stats(_std(amp[i]))
        # flatness must be read off the RAW amplitude: `_std` divides by the std, which is exactly
        # the quantity that makes a constant mode constant (D-025).
        ar = amp_ratio(amp[i])
        lab = label_family(st, amp_ratio=ar, **lab_kw)
        truth_rows.append(dict(mode=i, family=fams[i], label=lab, amp_ratio=ar, **st))
        print(f"  {i:>4} {fams[i]:>10} {ar:9.3f} {st['trend_share']:6.2f} {st['osc_share']:5.2f} "
              f"{st['residual_share']:6.2f} {lab:>10}")
    lab_ok = sum(r["label"] == r["family"] for r in truth_rows)
    print(f"  -> heuristic labeller reproduces {lab_ok}/{len(truth_rows)} known families")

    # --- per channel --------------------------------------------------------------------------
    Sz = _std(S)
    # A constant truth mode has ZERO variance, so `_std` would divide by ~0 and turn it into noise;
    # correlating a channel against that noise is meaningless. Flag those modes and report them via
    # amp_ratio / the time-mean check instead of a correlation (D-025).
    const_mode = amp.std(axis=1) < 1e-12                   # [n_modes]
    A = _std(amp.T)                                        # [T,n_modes]
    A[:, const_mode] = 0.0                                 # -> corr 0 rather than corr(noise)
    corr = (Sz.T @ A) / T                                  # [K,n_modes] post-hoc only
    print("\n  learned channels")
    print(f"  {'ch':>3} {'role':>7} {'amp_r':>7} {'trend':>6} {'osc':>5} {'resid':>6} "
          f"{'label':>10} {'cells':>7} {'r_gyr':>6} {'best mode':>18}")
    rows = []
    for i in range(K):
        st = series_stats(Sz[:, i])
        ms = mask_stats(masks[i])
        ar = amp_ratio(S[:, i])              # RAW channel: its static level vs its fluctuation
        lab = label_family(st, amp_ratio=ar, **lab_kw)
        j = int(np.argmax(np.abs(corr[i])))
        role = roles[i] if i < len(roles) else "-"
        rows.append(dict(ch=i, role=role, label=lab, best_mode=j, best_corr=float(corr[i, j]),
                         best_mode_family=fams[j], amp_ratio=ar, **st, **ms))
        print(f"  {i:>3} {role:>7} {ar:7.3f} {st['trend_share']:6.2f} {st['osc_share']:5.2f} "
              f"{st['residual_share']:6.2f} "
              f"{lab:>10} {ms['count']:7.0f} "
              f"{ms['r_gyr']:6.1f} "
              f"{'m%d(%s)' % (j, fams[j][:4]):>12}{np.abs(corr[i, j]):6.2f}")

    # --- collective ----------------------------------------------------------------------------
    pop = {f: sum(r["label"] == f for r in rows) for f in FAMILIES}
    cos_ov, iou_ov = mean_pairwise(masks)
    r2, r2b = recon_r2(field, S, device)
    # the untrained eye must see the SAME input as the trained one (SOP 02 input_norm)
    field_in = normalize_input(field, cfg.train.get("input_norm", "none"), int(art["t_fit"]) if "t_fit" in art.files else None)
    S_init = np.einsum("tvhw,kvhw->tk", field_in, masks_init)
    r2_i, r2b_i = recon_r2(field, S_init, device)
    mr, mr_init = mode_r2(amp, S, device), mode_r2(amp, S_init, device)
    mode_recovery = {f"m{j}({fams[j]})": dict(max_abs_corr=float(np.abs(corr[:, j]).max()),
                                              ensemble_r2=mr[j], ensemble_r2_init=mr_init[j])
                     for j in range(amp.shape[0])}
    # --- did any channel become a STATIONARY OBSERVER? (D-025, criterion revised by D-026) ----
    # THE CRITERION IS FLATNESS, AND NOTHING ELSE. The observer's job is to decide where to look,
    # not to recover the hidden modes (user, 2026-09-01): a flat channel may be a single static
    # mode, a persistent phenomenon, or a COMBINATION of varying signals whose sum barely moves --
    # all three are legitimate stationary observables. So `mask_align_with_pattern` below is a
    # DIAGNOSTIC ("what did it happen to sit on"), explicitly not a success measure; an earlier
    # version of this probe treated its low value as a failure, which was the wrong frame.
    stat_capture = None
    i_flat = int(np.argmin([r["amp_ratio"] for r in rows]))
    flat_ok = bool(rows[i_flat]["amp_ratio"] < 0.05)
    align = None
    if const_mode.any():
        phi0 = art["truth_phi"][int(np.argmax(const_mode))]
        align = float(np.corrcoef(masks[i_flat][0].ravel(), phi0.ravel())[0, 1])
    # HOW the flat channel got flat: is it sitting on something quiet, or cancelling several
    # varying signals? Contribution of hidden mode k to this channel is amp_k(t)*<mask, phi_k>
    # (SSH layer; the field's global rescaling cancels out of the ratio).
    proj = masks[i_flat][0].reshape(1, -1) @ art["truth_phi"].reshape(amp.shape[0], -1).T   # [1,n]
    contrib = amp * proj.ravel()[:, None]                       # [n_modes, T]
    sd_each = contrib.std(axis=1)
    net = float(contrib.sum(axis=0).std())
    # Normalize against the INDEPENDENCE baseline, not against the sum of the parts. Uncorrelated
    # modes already add to sqrt(sum sd^2), which for 3 equal parts is 0.58 of the plain sum -- so a
    # "net/sum = 0.63" reading looks like cancellation while being exactly what chance predicts.
    # cancel < 1 is destructive interference BEYOND chance; ~1 is incoherent addition; > 1 means the
    # contributions reinforce.
    indep = float(np.sqrt((sd_each ** 2).sum()))
    cancel = float(net / (indep + 1e-12))
    stat_capture = dict(best_channel=i_flat, best_channel_role=rows[i_flat]["role"],
                        amp_ratio=rows[i_flat]["amp_ratio"], flat_enough=flat_ok,
                        n_flat_channels=int(sum(r["amp_ratio"] < 0.05 for r in rows)),
                        cancellation_vs_independent=cancel,
                        net_over_sum_of_parts=float(net / (sd_each.sum() + 1e-12)),
                        n_modes_contributing=int((sd_each > 0.05 * sd_each.max()).sum()),
                        mask_align_with_pattern=align)   # DIAGNOSTIC ONLY -- see comment above
    print(f"\n  stationary observer  : ch{i_flat} ({rows[i_flat]['role']} rung)  "
          f"amp_ratio {rows[i_flat]['amp_ratio']:.4f} "
          f"({'FLAT' if flat_ok else 'still moving'})")
    print(f"    how it got flat    : {stat_capture['n_modes_contributing']} hidden modes reach its "
          f"footprint; net fluctuation / independent-addition = {cancel:.2f} "
          f"({'CANCELLATION beyond chance' if cancel < 0.7 else
             'incoherent addition, i.e. a quiet footprint' if cancel < 1.3 else
             'the modes REINFORCE'})"
          + (f" | mask-vs-pattern corr {align:+.3f} (diagnostic)" if align is not None else ""))

    coll = dict(population=pop, stationary_capture=stat_capture, target_population=dict(stationary=1, cyclic=3, chaotic=6),
                eff_rank=eff_rank(S), eff_rank_init=eff_rank(S_init),
                mask_cos_overlap=cos_ov, mask_iou_overlap=iou_ov,
                recon_r2=r2, recon_r2_init=r2_i, recon_r2_bal=r2b, recon_r2_bal_init=r2b_i,
                slowness_spread=float(max(r["gap1"] for r in rows) / max(1e-12, min(r["gap1"] for r in rows))),
                mode_recovery=mode_recovery,
                labeller_truth_accuracy=lab_ok / len(truth_rows),
                # D-024: the ladder ASSIGNS roles; this says whether the channels obeyed. A rung
                # maps to a family as slow->stationary, cyclic->cyclic, fast->chaotic.
                role_obedience=(sum(r["label"] == _ROLE2FAM.get(r["role"], "") for r in rows) / K
                                if roles else None))

    print(f"\n  population           : {pop}   (design target 1/3/6 in proportion -- what we ASK "
          f"the bank for, not a mode-recovery score)")
    if roles:
        print(f"  role obedience       : {coll['role_obedience']:.2f}  "
              f"(channels whose label matches their assigned rung)")
    print(f"  slowness spread      : {coll['slowness_spread']:.1f}x  "
          f"(gap1 {min(r['gap1'] for r in rows):.4f} .. {max(r['gap1'] for r in rows):.4f})")
    print(f"  effective rank       : {coll['eff_rank']:.2f} / {K}   (init {coll['eff_rank_init']:.2f})")
    print(f"  mask overlap         : cos {cos_ov:.3f}   IoU {iou_ov:.3f}")
    print(f"  RECON R^2 energy-wtd : {r2:.4f}   (untrained masks {r2_i:.4f})")
    print(f"  RECON R^2 balanced   : {r2b:.4f}   (untrained masks {r2b_i:.4f})   <- the real one")
    print(f"  {'hidden mode':>16} {'maxcorr':>8} {'ens R2':>8} {'(init)':>8}   "
          f"<- DIAGNOSTIC: full recovery is not the goal (D-026)")
    for k, v in mode_recovery.items():
        # a constant mode has no ensemble R^2 to report (see mode_r2); it is judged by
        # `stationary_capture` above, so print a dash rather than a fake number
        def _f(x):
            return f"{x:8.2f}" if x is not None else f"{'--':>8}"
        print(f"  {k:>16} {v['max_abs_corr']:8.2f} {_f(v['ensemble_r2'])} "
              f"{_f(v['ensemble_r2_init'])}")

    # --- artifacts: eval.json + TensorBoard ----------------------------------------------------
    with open(os.path.join(run, "eval.json"), "w") as f:
        json.dump(dict(run=run, channels=rows, truth=truth_rows, collective=coll), f, indent=2)

    # ONE TensorBoard dir per run: eval writes into the SAME `tb/` the training loop used, so a
    # run is a single entry in TensorBoard instead of two that have to be read side by side.
    # Tag namespaces keep them apart (train: loss/*, var/*; eval: masks/*, S/*, collective/*).
    writer = SummaryWriter(log_dir=os.path.join(run, "tb"))
    mv = masks.max(axis=1)                                 # [K,H,W] max over the variable axis
    writer.add_images("masks/trained", mv[:, None], 0)
    writer.add_images("masks/init", masks_init.max(axis=1)[:, None], 0)
    for i in range(K):                                     # the K scalar lines (CLAUDE.md readout)
        for t in range(0, T, max(1, T // 1000)):
            writer.add_scalar(f"S/ch{i:02d}", float(Sz[t, i]), t)
    for f, v in dict(recon_r2=r2, recon_r2_init=r2_i, recon_r2_bal=r2b, recon_r2_bal_init=r2b_i,
                     eff_rank=coll["eff_rank"], mask_iou=iou_ov,
                     n_stationary=pop["stationary"], n_cyclic=pop["cyclic"],
                     n_chaotic=pop["chaotic"]).items():
        writer.add_scalar(f"collective/{f}", float(v), 0)
    writer.close()
    print(f"\n[eval] wrote {os.path.relpath(os.path.join(run, 'eval.json'), repo)}"
          f" + TensorBoard events into {os.path.relpath(os.path.join(run, 'tb'), repo)}/")


if __name__ == "__main__":
    main()
