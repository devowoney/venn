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

from src.data.synthetic import GenConfig, generate_field

FAMILIES = ("stationary", "cyclic", "chaotic")


# --------------------------------------------------------------------------- series descriptors
def _std(a: np.ndarray) -> np.ndarray:
    """Standardize along time (axis 0 of a [T] or [T,K] array)."""
    return (a - a.mean(0)) / (a.std(0) + 1e-12)


def series_stats(x: np.ndarray, max_lag: int = 400) -> dict:
    """Descriptors of ONE standardized series [T]: slowness, decorrelation time, spectral peak."""
    T = len(x)
    gap1 = float(np.var(np.diff(x)))                       # = 2(1-rho1) for unit variance
    xf = np.fft.rfft(x - x.mean())
    p = np.abs(xf) ** 2
    p[0] = 0.0
    kpk = int(np.argmax(p))
    peak_frac = float(p[kpk] / (p.sum() + 1e-12))          # spectral concentration -> cyclicity
    period = float(T / kpk) if kpk > 0 else float("inf")
    # first lag where the autocorrelation drops below 1/e (via FFT autocorrelation)
    ac = np.fft.irfft(np.abs(np.fft.rfft(x - x.mean(), 2 * T)) ** 2)[:max_lag]
    ac = ac / (ac[0] + 1e-12)
    below = np.where(ac < np.exp(-1.0))[0]
    tau = int(below[0]) if len(below) else max_lag
    return dict(gap1=gap1, rho1=float(1.0 - gap1 / 2.0), tau_e=tau,
                period=period, peak_frac=peak_frac)


def label_family(st: dict, slow_thr: float, peak_thr: float) -> str:
    """Heuristic, post-hoc, reversible family label (D-004: classifier is a labeller, not a gate).

    cyclic     -> spectral power concentrated in one line (a clean oscillation)
    stationary -> not cyclic, and slow: small lag-1 gap / long decorrelation time
    chaotic    -> everything else (broadband and fast)
    """
    if st["peak_frac"] > peak_thr and np.isfinite(st["period"]):
        return "cyclic"
    return "stationary" if st["gap1"] < slow_thr else "chaotic"


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
    return [_r2(torch.from_numpy(_std(a)[:, None]).to(device).double(), X) for a in amp]


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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: newest .tmps/runs/*)")
    ap.add_argument("--slow-thr", type=float, default=0.03,
                    help="lag-1 gap variance below which a channel counts as stationary")
    ap.add_argument("--peak-thr", type=float, default=0.10,
                    help="spectral peak fraction above which a channel counts as cyclic")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run = args.run or sorted(glob.glob(os.path.join(repo, ".tmps/runs/*")))[-1]
    run = os.path.abspath(run)
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    S, masks, masks_init = art["S"], art["masks"], art["masks_init"]
    amp, fams = art["truth_amp"], list(art["families"])
    K, T = S.shape[1], S.shape[0]
    field, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)),
                              seed=int(cfg.seed))

    print(f"[eval] run={os.path.relpath(run, repo)}  K={K} T={T} temp_final={float(art['temp_final']):.3f}")

    # --- reference: what the HIDDEN modes look like under the same descriptors ----------------
    print("\n  hidden ground-truth modes (answer key -- never seen by training)")
    print(f"  {'mode':>4} {'family':>10} {'gap1':>7} {'tau_e':>6} {'peak%':>6} {'period':>8} {'label':>10}")
    truth_rows = []
    for i in range(amp.shape[0]):
        st = series_stats(_std(amp[i]))
        lab = label_family(st, args.slow_thr, args.peak_thr)
        truth_rows.append(dict(mode=i, family=fams[i], label=lab, **st))
        print(f"  {i:>4} {fams[i]:>10} {st['gap1']:7.4f} {st['tau_e']:6d} "
              f"{100*st['peak_frac']:6.1f} {st['period']:8.1f} {lab:>10}")
    lab_ok = sum(r["label"] == r["family"] for r in truth_rows)
    print(f"  -> heuristic labeller reproduces {lab_ok}/{len(truth_rows)} known families")

    # --- per channel --------------------------------------------------------------------------
    Sz = _std(S)
    A = _std(amp.T)                                        # [T,n_modes]
    corr = (Sz.T @ A) / T                                  # [K,n_modes] post-hoc only
    print("\n  learned channels")
    print(f"  {'ch':>3} {'gap1':>7} {'tau_e':>6} {'peak%':>6} {'period':>8} {'label':>10} "
          f"{'cells':>7} {'frac':>5} {'bin':>5} {'r_gyr':>6} {'best mode':>18}")
    rows = []
    for i in range(K):
        st = series_stats(Sz[:, i])
        ms = mask_stats(masks[i])
        lab = label_family(st, args.slow_thr, args.peak_thr)
        j = int(np.argmax(np.abs(corr[i])))
        rows.append(dict(ch=i, label=lab, best_mode=j, best_corr=float(corr[i, j]),
                         best_mode_family=fams[j], **st, **ms))
        print(f"  {i:>3} {st['gap1']:7.4f} {st['tau_e']:6d} {100*st['peak_frac']:6.1f} "
              f"{st['period']:8.1f} {lab:>10} {ms['count']:7.0f} {ms['frac']:5.2f} "
              f"{ms['binarity']:5.2f} {ms['r_gyr']:6.1f} "
              f"{'m%d(%s)' % (j, fams[j][:4]):>12}{np.abs(corr[i, j]):6.2f}")

    # --- collective ----------------------------------------------------------------------------
    pop = {f: sum(r["label"] == f for r in rows) for f in FAMILIES}
    cos_ov, iou_ov = mean_pairwise(masks)
    r2, r2b = recon_r2(field, S, device)
    S_init = np.einsum("tvhw,kvhw->tk", field, masks_init)
    r2_i, r2b_i = recon_r2(field, S_init, device)
    mr, mr_init = mode_r2(amp, S, device), mode_r2(amp, S_init, device)
    mode_recovery = {f"m{j}({fams[j]})": dict(max_abs_corr=float(np.abs(corr[:, j]).max()),
                                              ensemble_r2=mr[j], ensemble_r2_init=mr_init[j])
                     for j in range(amp.shape[0])}
    coll = dict(population=pop, target_population=dict(stationary=1, cyclic=3, chaotic=6),
                eff_rank=eff_rank(S), eff_rank_init=eff_rank(S_init),
                mask_cos_overlap=cos_ov, mask_iou_overlap=iou_ov,
                recon_r2=r2, recon_r2_init=r2_i, recon_r2_bal=r2b, recon_r2_bal_init=r2b_i,
                slowness_spread=float(max(r["gap1"] for r in rows) / max(1e-12, min(r["gap1"] for r in rows))),
                mode_recovery=mode_recovery,
                labeller_truth_accuracy=lab_ok / len(truth_rows))

    print(f"\n  population           : {pop}   (target 1/3/6 in proportion)")
    print(f"  slowness spread      : {coll['slowness_spread']:.1f}x  "
          f"(gap1 {min(r['gap1'] for r in rows):.4f} .. {max(r['gap1'] for r in rows):.4f})")
    print(f"  effective rank       : {coll['eff_rank']:.2f} / {K}   (init {coll['eff_rank_init']:.2f})")
    print(f"  mask overlap         : cos {cos_ov:.3f}   IoU {iou_ov:.3f}")
    print(f"  RECON R^2 energy-wtd : {r2:.4f}   (untrained masks {r2_i:.4f})")
    print(f"  RECON R^2 balanced   : {r2b:.4f}   (untrained masks {r2b_i:.4f})   <- the real one")
    print(f"  {'hidden mode':>16} {'maxcorr':>8} {'ens R2':>8} {'(init)':>8}")
    for k, v in mode_recovery.items():
        print(f"  {k:>16} {v['max_abs_corr']:8.2f} {v['ensemble_r2']:8.2f} "
              f"{v['ensemble_r2_init']:8.2f}")

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
