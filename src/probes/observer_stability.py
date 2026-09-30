"""Is the eye-lobe (observer) STABLE over training time, and does it see enough to rebuild the state?

North Star (user, 2026-09-30): a stable observer (over training time) that makes sufficient, meaningful
signal to reconstruct the original states. This probe scores both halves of that sentence on the mask
snapshots written by `train.snap_every`:

  stability      : per-channel IoU of the binarized mask (m > 0.5 = "activated pixel") against the FINAL
                   mask, and against the previous snapshot. An eye that has settled reads ~1 on both.
  sufficiency    : balanced linear-decode R^2 of the field from the K channels, decoder fitted on the
                   training slice [0, t_fit) and scored on the unseen validation slice [t_val, T).
                   Out-of-sample on purpose: evaluate.py's recon R^2 is fitted and scored on the whole
                   record, so it cannot tell a sensor from a memorized fit.

Writes stability.json + stability.png into --out (default .tmps/observer_stability/).

Run:  conda run -n oceanai python -m src.probes.observer_stability \
          --runs '.tmps/runs/obs_t2000_seed*' '.tmps/runs/obs_t8000_seed*'
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import GenConfig, generate_field  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel IoU of two binarized mask banks [K,V,H,W] -> [K]."""
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    inter, union = (a & b).sum(1), (a | b).sum(1)
    return inter / np.maximum(union, 1)


def oos_recon_r2(Y: torch.Tensor, S: torch.Tensor, t_fit: int, t_val: int) -> tuple[float, float]:
    """Balanced decode R^2 of the field Y [T,N] from S [T,K]: fitted on [0,t_fit), scored on [t_val,T).

    Every statistic (cell standardization, channel standardization, decoder weights) comes from the
    training slice only, so the validation score is a genuine forecast-free reconstruction test.
    Returns (train R^2, validation R^2).
    """
    mu_y, sd_y = Y[:t_fit].mean(0), Y[:t_fit].std(0) + 1e-12      # "balanced": every cell counts equally
    Yb = (Y - mu_y) / sd_y
    mu_s, sd_s = S[:t_fit].mean(0), S[:t_fit].std(0) + 1e-12
    X = torch.cat([torch.ones(len(S), 1, dtype=S.dtype, device=S.device), (S - mu_s) / sd_s], dim=1)
    beta = torch.linalg.lstsq(X[:t_fit], Yb[:t_fit]).solution         # [K+1, N] linear decoder

    def r2(sl: slice) -> float:
        y, yh = Yb[sl], X[sl] @ beta
        return float(1.0 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    return r2(slice(0, t_fit)), r2(slice(t_val, len(Y)))


def score_run(run: str, t_val: int, device: str) -> dict:
    """Stability + sufficiency curves for one run dir that carries `masks_snap`."""
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    field_np, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)), seed=cfg.seed)
    T = field_np.shape[0]
    t_fit = int(art["t_fit"])
    assert t_fit <= t_val < T, f"{run}: training slice {t_fit} overlaps validation start {t_val}"

    field = torch.from_numpy(field_np).to(device)                      # [T,V,H,W]
    Y = field.reshape(T, -1).double()                                  # [T,N]
    snaps = art["masks_snap"].astype(np.float32)                       # [n,K,V,H,W]
    # noise ceiling: the best balanced R^2 ANY observer can reach, because the generator's iid obs noise is
    # unpredictable. Per cell: 1 - noise_var / cell_var, averaged over cells like the balanced R^2 itself.
    # Raw R^2 differs by seed mostly through how much of the grid is signal-free; R^2 / ceiling does not.
    ceiling = float((1.0 - float(cfg.data.obs_noise) ** 2 / Y[:t_fit].var(0)).mean())
    on = snaps > 0.5                                                   # activated pixels
    rows = []
    for j, m in enumerate(snaps):
        with torch.no_grad():                                          # s_i = <mask_i, field> (norm: none)
            S = torch.einsum("tvhw,kvhw->tk", field, torch.from_numpy(m).to(device)).double()
        r2_tr, r2_va = oos_recon_r2(Y, S, t_fit, t_val)
        rows.append(dict(step=int(art["snap_steps"][j]),
                         iou_final=float(iou(on[j], on[-1]).mean()),
                         iou_prev=float(iou(on[j], on[j - 1]).mean()) if j else float("nan"),
                         iou_final_min=float(iou(on[j], on[-1]).min()),
                         active=float(on[j].sum() / on[j].shape[0]),  # mean activated pixels per channel
                         r2_train=r2_tr, r2_val=r2_va, r2_val_norm=r2_va / ceiling))
    return dict(run=os.path.relpath(run), seed=int(cfg.seed), t_fit=t_fit, t_val=t_val, T=T,
                noise_ceiling=ceiling, curve=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run-dir globs (each needs masks_snap)")
    ap.add_argument("--t-val", type=int, default=8000, help="validation starts here, scored to T")
    ap.add_argument("--out", default=".tmps/observer_stability")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = sorted({r for g in args.runs for r in glob.glob(g) if os.path.isdir(r)})
    res = [score_run(r, args.t_val, device) for r in runs]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stability.json"), "w") as f:
        json.dump(res, f, indent=1)

    # --- figure: one colour per training length, thin = seed, thick = seed mean --------------------
    arms = sorted({r["t_fit"] for r in res})
    cols = dict(zip(arms, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]))
    panels = [("iou_final", "IoU of activated pixels vs FINAL mask"),
              ("iou_prev", "IoU vs previous snapshot (250 steps earlier)"),
              ("r2_val_norm", f"recon R$^2$ / noise ceiling on validation [{args.t_val},T)")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (key, title) in zip(axs, panels):
        for t in arms:
            sub = [r for r in res if r["t_fit"] == t]
            x = np.array([c["step"] for c in sub[0]["curve"]])
            ys = np.array([[c[key] for c in r["curve"]] for r in sub])
            for y in ys:
                ax.plot(x, y, color=cols[t], lw=0.7, alpha=0.35)
            ax.plot(x, np.nanmean(ys, 0), color=cols[t], lw=2.2, label=f"train [0,{t})  n={len(sub)}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=9)
    axs[2].axhline(1.0, color="#52514e", lw=0.8, ls=":")                # 1 = everything recoverable captured
    axs[2].set_ylim(0.9, 1.01)
    fig.suptitle("Eye-lobe stability and sufficiency vs training length (same T=10000 system per seed)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "stability.png"), dpi=130)

    # --- console summary at the last snapshot. (A "settle step" vs the FINAL mask was dropped: IoU vs final
    # reaches 1 by construction, so it reports when training stopped, not when the eye stopped moving.)
    print(f"{'run':>34} {'iou_final@50%':>13} {'iou_prev@end':>12} {'R2 tr':>6} {'R2 val':>6} {'ceil':>6} "
          f"{'R2/ceil':>7}")
    for r in res:
        c = r["curve"]
        mid = c[len(c) // 2]["iou_final"]
        print(f"{r['run']:>34} {mid:13.3f} {c[-1]['iou_prev']:12.3f} {c[-1]['r2_train']:6.3f} "
              f"{c[-1]['r2_val']:6.3f} {r['noise_ceiling']:6.3f} {c[-1]['r2_val_norm']:7.3f}")
    print(f"[stability] wrote {args.out}/stability.json + stability.png")


if __name__ == "__main__":
    main()
