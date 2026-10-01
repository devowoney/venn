"""Are SSH and SST coupled inside the eye, and can SST be read from SSH alone?

Question (user, 2026-10-01): "Are the two dynamics correlated in the architecture? Can we investigate the
variation of SST from SSH observation?"

Two measurements on trained eyes (run dirs with `artifacts.npz`; masks span the full (V,H,W) volume):

  A. coupling in the eye
     Every channel is linear in the field, so it splits EXACTLY into an SSH part and an SST part:
         s_i = <mask_i[SSH], SSH> + <mask_i[SST], SST> = s_i^SSH + s_i^SST.
     Reported per channel: share of the mask on SSH, share of the channel's variance carried by each part,
     and the lagged correlation between the two parts (SST is an AR1 response to SSH, so its part should LAG).

  B. SST from SSH observation
     Balanced linear decode of the SST field (every cell standardized, like the reconstruction score) from:
       ssh_field_now   all 4096 SSH cells at t (via top PCs)          -> upper bound for an instantaneous view
       ssh_field_hist  same + lagged copies                           -> upper bound with memory
       ssh_eye_now     the SSH parts s^SSH of the K channels at t     -> what the eye sees of SSH
       ssh_eye_hist    same + lagged copies
       full_eye_now    the full channels (they also SEE SST)          -> reference, not an SSH-only answer
     Decoders are fitted on [0, t_fit) and scored on the unseen [t_fit, T).

     Ceiling: the generator's SST = shared modes (an AR1 low-pass of the SSH amplitudes) + SST-private modes +
     iid noise. Nothing in SSH can explain the private modes or the noise, so the best any SSH-based decoder
     can reach is the variance share of the SHARED part ("SSH-explainable ceiling"). The truth is rebuilt from
     the hidden answer key with the generator's own rules, and also gives a per-mode readout: which SST mode
     amplitude each decoder recovers.

Writes ssh_to_sst.json + one figure per run + a summary figure into --out (default .tmps/ssh_sst/).

Run:  conda run -n oceanai python -m src.probes.ssh_to_sst --runs '.tmps/runs/obs_t8000_seed*'
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
from omegaconf import OmegaConf  # noqa: E402

from src.data.synthetic import GenConfig, _ar1_response, _gaussian_blur, generate_field  # noqa: E402

SSH_C, SST_C, THIRD_C, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86"
LAGS = (0, 1, 2, 4, 8, 16, 32, 64)        # generic memory, covers several SST response times (tau = 20)
N_PC = 64                                  # SSH field compressed to its top PCs before the lagged decode
DECODERS = ("ssh_field_now", "ssh_field_hist", "ssh_eye_now", "ssh_eye_hist", "full_eye_now")


# ---------------------------------------------------------------------------------------------------------
# hidden truth of the SST variable, split by mode
# ---------------------------------------------------------------------------------------------------------
def sst_truth(field_sst: np.ndarray, truth: dict, cfg: GenConfig) -> dict:
    """Rebuild the SST state mode by mode, exactly as `generate_field` builds it, in the field's units.

    Returns per-mode SST amplitude series [M,T], per-mode SST maps [M,H,W] (already divided by the
    variable's standardization std), the `only` flags, and the shared / private / noise parts of the field.
    """
    modes = truth["modes"]
    T = field_sst.shape[0]
    amps, maps = [], []
    for m in modes:
        a, shared = m["amp"].astype(np.float64), m["family"] in ("stationary", "cyclic")
        w = cfg.sst_shared_w if shared else cfg.sst_chaotic_w
        if np.std(a) < 1e-12:
            a1 = a
        elif cfg.sst_tau > 0:
            a1 = _ar1_response(a, cfg.sst_tau)
        else:
            a1 = a[np.clip(np.arange(T) - (cfg.sst_lag if shared else 0), 0, T - 1)]
        amps.append(w * a1 if m["only"] != "ssh" else np.zeros(T))
        maps.append(_gaussian_blur(m["phi"].astype(np.float64), cfg.sst_blur_sigma) if shared
                    else m["phi"].astype(np.float64))
    A, P = np.stack(amps), np.stack(maps)                        # [M,T], [M,H,W]
    raw = np.einsum("mt,mhw->thw", A, P)
    mu, sd = raw.mean(), raw.std() + 1e-8                        # the generator's per-variable standardization
    P = P / sd
    only = np.array([m["only"] for m in modes])
    comp = lambda sel: np.einsum("mt,mhw->thw", A[sel], P[sel])  # noqa: E731
    shared_part = comp(only == "both")                           # constant offsets vanish in per-cell stats
    private_part = comp(only == "sst")
    state = raw / sd - mu / sd
    return dict(amp=A, maps=P, only=only, shared=shared_part, private=private_part,
                noise=field_sst - state, families=np.array([m["family"] for m in modes]))


# ---------------------------------------------------------------------------------------------------------
# decoding helpers
# ---------------------------------------------------------------------------------------------------------
def lagged(X: np.ndarray, lags=LAGS) -> np.ndarray:
    """Stack causal lagged copies [T, F*len(lags)]; the first max(lag) rows repeat the first sample."""
    T = len(X)
    return np.concatenate([X[np.clip(np.arange(T) - L, 0, T - 1)] for L in lags], axis=1)


def ridge_decode(X: np.ndarray, Y: np.ndarray, t_fit: int, t0: int, ridge: float = 1e-3):
    """Fit Y ~ [1, X] on [t0, t_fit) with standardized features, predict all T. Stats from training only."""
    mu, sd = X[t0:t_fit].mean(0), X[t0:t_fit].std(0) + 1e-12
    Xs = np.concatenate([np.ones((len(X), 1)), (X - mu) / sd], axis=1)
    A = Xs[t0:t_fit]
    G = A.T @ A
    G[1:, 1:] += ridge * (t_fit - t0) * np.eye(G.shape[0] - 1)  # ridge relative to the sample count
    beta = np.linalg.solve(G, A.T @ Y[t0:t_fit])
    return Xs @ beta


def balanced_r2(Yb: np.ndarray, Yh: np.ndarray, sl: slice) -> float:
    """Pooled R^2 over cells that were standardized with training stats (each cell weighs the same)."""
    y, yh = Yb[sl], Yh[sl]
    return float(1.0 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def series_r2(y: np.ndarray, yh: np.ndarray) -> float:
    v = y.var()
    return float("nan") if v < 1e-10 else float(1.0 - ((y - yh) ** 2).mean() / v)


def lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 60):
    """corr(a(t), b(t+L)) for L in [-max_lag, max_lag]; L > 0 means b LAGS a."""
    a, b = (a - a.mean()) / (a.std() + 1e-12), (b - b.mean()) / (b.std() + 1e-12)
    Ls = np.arange(-max_lag, max_lag + 1)
    c = np.array([np.mean(a[max(0, -L):len(a) - max(0, L)] * b[max(0, L):len(b) - max(0, -L)]) for L in Ls])
    return Ls, c


# ---------------------------------------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------------------------------------
def analyse_run(run: str, out: str) -> dict:
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    cfg_h = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    gcfg = GenConfig(**OmegaConf.to_container(cfg_h.data, resolve=True))
    field, truth = generate_field(gcfg, seed=cfg_h.seed)
    field = field.astype(np.float64)
    T, V, H, W = field.shape
    t_fit, t0 = int(art["t_fit"]), max(LAGS)                     # t0: skip rows whose lags are padded
    val = slice(t_fit, T)
    masks = art["masks"].astype(np.float64)                      # [K,V,H,W]
    roles = [str(r) for r in art["roles"]]
    K = len(masks)

    ssh, sst = field[:, 0].reshape(T, -1), field[:, 1].reshape(T, -1)
    s_ssh = ssh @ masks[:, 0].reshape(K, -1).T                   # [T,K]  SSH part of every channel
    s_sst = sst @ masks[:, 1].reshape(K, -1).T                   # [T,K]  SST part
    S = s_ssh + s_sst                                             # == the encoder's own output
    assert np.allclose(S[:, :], art["S"].astype(np.float64), atol=1e-2 * np.abs(S).max()), "S split mismatch"

    # ---- A. coupling inside the eye ------------------------------------------------------------------
    mass_ssh = masks[:, 0].sum((1, 2)) / masks.sum((1, 2, 3))
    v_ssh, v_sst = s_ssh[:t_fit].var(0), s_sst[:t_fit].var(0)
    var_share_ssh = v_ssh / (v_ssh + v_sst + 1e-12)
    corr0, peak_corr, peak_lag = [], [], []
    for i in range(K):
        Ls, c = lag_corr(s_ssh[:t_fit, i], s_sst[:t_fit, i])
        corr0.append(float(c[Ls == 0][0]))
        j = int(np.argmax(np.abs(c)))
        peak_corr.append(float(c[j]))
        peak_lag.append(int(Ls[j]))

    # ---- B. SST from SSH ------------------------------------------------------------------------------
    st = sst_truth(field[:, 1], truth, gcfg)
    mu_y, sd_y = sst[:t_fit].mean(0), sst[:t_fit].std(0) + 1e-12
    Yb = (sst - mu_y) / sd_y
    # ceilings, balanced like the score: explainable variance share per cell, averaged over cells
    shared = st["shared"].reshape(T, -1)
    ceil_ssh = float(np.mean(shared[val].var(0) / sst[val].var(0)))
    ceil_noise = float(np.mean(1.0 - st["noise"].reshape(T, -1)[val].var(0) / sst[val].var(0)))

    # SSH field PCs from the training slice (the decoder may use anything SSH holds)
    ssh_c = ssh - ssh[:t_fit].mean(0)
    _, _, Vt = np.linalg.svd(ssh_c[:t_fit], full_matrices=False)
    pcs = ssh_c @ Vt[:N_PC].T
    feats = dict(ssh_field_now=pcs, ssh_field_hist=lagged(pcs[:, :32]),
                 ssh_eye_now=s_ssh, ssh_eye_hist=lagged(s_ssh), full_eye_now=S)
    preds, r2 = {}, {}
    for name in DECODERS:
        preds[name] = ridge_decode(feats[name], Yb, t_fit, t0)
        r2[name] = balanced_r2(Yb, preds[name], val)

    # per-mode readout: how much of each mode's SST amplitude each decoder recovers (validation)
    per_mode = {}
    for name in DECODERS:
        rows = []
        for k in range(len(st["amp"])):
            a = st["amp"][k]
            if a.std() < 1e-10:
                rows.append(float("nan"))
                continue
            ah = ridge_decode(feats[name], a[:, None], t_fit, t0)[:, 0]
            rows.append(series_r2(a[val], ah[val]))
        per_mode[name] = rows

    res = dict(run=os.path.basename(run), seed=int(cfg_h.seed), K=K, roles=roles,
               mass_share_ssh=mass_ssh.tolist(), var_share_ssh=var_share_ssh.tolist(),
               corr_parts_lag0=corr0, corr_parts_peak=peak_corr, peak_lag=peak_lag,
               ceiling_ssh_explainable=ceil_ssh, ceiling_noise=ceil_noise, r2=r2,
               r2_over_ceiling={k: v / ceil_ssh for k, v in r2.items()},
               modes=[dict(family=str(f), only=str(o)) for f, o in zip(st["families"], st["only"])],
               per_mode_r2=per_mode)
    plot_run(res, field, st, preds, Yb, sd_y, mu_y, t_fit, out)
    return res


# ---------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------
LABEL = dict(ssh_field_now="SSH field\nnow", ssh_field_hist="SSH field\n+ history", ssh_eye_now="eye, SSH part\nnow",
             ssh_eye_hist="eye, SSH part\n+ history", full_eye_now="full eye\n(sees SST)")


def plot_run(res, field, st, preds, Yb, sd_y, mu_y, t_fit, out):
    T, _, H, W = field.shape
    K = res["K"]
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.55, wspace=0.3)

    # A1: how each eye divides between the variables
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(K)
    ax.bar(x - 0.2, res["mass_share_ssh"], 0.38, color=SSH_C, label="mask mass on SSH")
    ax.bar(x + 0.2, res["var_share_ssh"], 0.38, color=THIRD_C, label="channel variance from SSH")
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_xticks(x, [f"{i}\n{r[:4]}" for i, r in enumerate(res["roles"])], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share on SSH (1 = SSH only, 0 = SST only)")
    ax.set_title("A. Each eye mixes both variables", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # A2: correlation between the SSH part and the SST part of each eye, with the lag of the peak
    ax = fig.add_subplot(gs[0, 2:])
    ax.bar(x, res["corr_parts_peak"], 0.6, color=SST_C, label="peak lagged corr")
    ax.scatter(x, res["corr_parts_lag0"], color="k", s=14, zorder=3, label="corr at lag 0")
    for i, L in enumerate(res["peak_lag"]):
        ax.text(i, 1.03, f"{L:+d}", ha="center", fontsize=7)
    ax.set_xticks(x, [str(i) for i in range(K)], fontsize=7)
    ax.set_ylim(-1, 1.15)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("A. corr(SSH part, SST part) per eye; number = lag of peak (>0: SST lags)", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # B1: decoder ladder vs ceilings
    ax = fig.add_subplot(gs[1, :2])
    vals = [res["r2"][d] for d in DECODERS]
    cols = [SSH_C, SSH_C, THIRD_C, THIRD_C, GREY]
    ax.bar(range(len(DECODERS)), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(res["ceiling_ssh_explainable"], color=SST_C, lw=1.5, ls="--",
               label=f"SSH-explainable ceiling {res['ceiling_ssh_explainable']:.2f}")
    ax.axhline(res["ceiling_noise"], color="k", lw=1, ls=":", label=f"noise ceiling {res['ceiling_noise']:.2f}")
    ax.set_xticks(range(len(DECODERS)), [LABEL[d] for d in DECODERS], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("SST balanced R² (validation)")
    ax.set_title("B. Rebuilding the SST field from SSH", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # B2: per-mode recovery
    ax = fig.add_subplot(gs[1, 2:])
    M = np.array([res["per_mode_r2"][d] for d in DECODERS])        # [D,M]
    im = ax.imshow(np.clip(M, 0, 1), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for (d, k), v in np.ndenumerate(M):
        ax.text(k, d, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="k" if np.isnan(v) or v < 0.6 else "w")
    ax.set_yticks(range(len(DECODERS)), [LABEL[d].replace("\n", " ") for d in DECODERS], fontsize=8)
    ax.set_xticks(range(M.shape[1]), [f"{m['family'][:4]}\n{m['only']}" for m in res["modes"]], fontsize=7)
    ax.set_title("B. SST amplitude of each hidden mode recovered (R²; 'sst' = SST-private)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.025)

    # B3: one validation snapshot + one cell time series
    t = t_fit + 700
    truth_map = Yb[t].reshape(H, W)
    vmax = np.abs(truth_map).max()
    for j, (img, title) in enumerate([(truth_map, "SST truth (std units)"),
                                      (preds["ssh_field_hist"][t].reshape(H, W), "from SSH field + history"),
                                      (preds["ssh_eye_hist"][t].reshape(H, W), "from eye SSH part + history")]):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(f"{title}, t={t}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    ax = fig.add_subplot(gs[2, 3])
    c = int(np.argmax(st["shared"].reshape(T, -1)[t_fit:].var(0)))   # the cell SSH should explain best
    tt = np.arange(t_fit, t_fit + 600)
    ax.plot(tt, Yb[tt, c], color="k", lw=1.5, label="SST truth")
    ax.plot(tt, preds["ssh_field_now"][tt, c], color=SSH_C, lw=1, ls="--", label="SSH field now")
    ax.plot(tt, preds["ssh_field_hist"][tt, c], color=SSH_C, lw=1.5, label="SSH field + history")
    ax.plot(tt, preds["ssh_eye_hist"][tt, c], color=THIRD_C, lw=1.5, label="eye SSH part + history")
    ax.set_title(f"SST at the most SSH-explainable cell ({c // W},{c % W})", fontsize=9)
    ax.set_xlabel("time step (validation)")
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"{res['run']}: SSH ↔ SST inside the eye, and SST from SSH observation", fontsize=13)
    fig.savefig(os.path.join(out, f"ssh_to_sst_{res['run']}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_res, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    D = len(DECODERS)
    for s, r in enumerate(all_res):
        ax.plot(range(D), [r["r2_over_ceiling"][d] for d in DECODERS], "o-", color=SSH_C, alpha=0.35 + 0.13 * s,
                lw=1.5, ms=6, label=f"seed {r['seed']}")
    ax.axhline(1, color=SST_C, ls="--", lw=1.5, label="SSH-explainable ceiling")
    ax.set_xticks(range(D), [LABEL[d] for d in DECODERS], fontsize=8)
    ax.set_ylabel("SST R² / SSH-explainable ceiling")
    ax.set_title("SST from SSH, all seeds (above 1 only possible if the decoder sees SST)", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1]
    fams = {"stationary": "o", "cyclic": "s", "chaotic": "^"}
    for d, col, off in [("ssh_field_now", SSH_C, -0.15), ("ssh_field_hist", SSH_C, -0.05),
                        ("ssh_eye_now", THIRD_C, 0.05), ("ssh_eye_hist", THIRD_C, 0.15)]:
        for r in all_res:
            for m, v in zip(r["modes"], r["per_mode_r2"][d]):
                if np.isnan(v):
                    continue
                g = {"both": 0, "ssh": 1, "sst": 2}[m["only"]] + (0.5 if m["family"] == "chaotic" else 0)
                ax.scatter(g + off, v, marker=fams[m["family"]], color=col, s=22,
                           alpha=1.0 if "hist" in d else 0.4, edgecolors="none")
    ax.set_xticks([0, 0.5, 2, 2.5], ["shared\ncyclic", "shared\nchaotic", "SST-private\ncyclic", "SST-private\nchaotic"])
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("R² of the mode's SST amplitude")
    ax.set_title("Per hidden mode (blue: SSH field, aqua: eye SSH part; faint = now, solid = + history)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ssh_to_sst_summary.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run-dir globs (need artifacts.npz with masks)")
    ap.add_argument("--out", default=".tmps/ssh_sst")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(r for g in a.runs for r in glob.glob(g) if os.path.isdir(r))
    all_res = []
    for r in runs:
        res = analyse_run(r, a.out)
        all_res.append(res)
        print(f"{res['run']}: ceiling {res['ceiling_ssh_explainable']:.3f} | "
              + " ".join(f"{d}={res['r2'][d]:.3f}" for d in DECODERS), flush=True)
    plot_summary(all_res, a.out)
    with open(os.path.join(a.out, "ssh_to_sst.json"), "w") as f:
        json.dump(all_res, f, indent=1)


if __name__ == "__main__":
    main()
