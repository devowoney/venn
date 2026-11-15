"""Show the ESTIMATED DISTRIBUTIONS of a forecast, not only their scores (D-053 runs): p(z(t+a)) per channel and the
decoded p(x(t+a)) at single cells, for one validation launch, and p(x) at one cell through time.

  latent.png  one launch: density of z_i for 2 channels per family -- fixed Gaussian N(zhat, sigma^2), the network
              (Gaussian, the decoder's prior) and the network with family shapes -- with zhat and the true z marked
  field.png   one launch: truth, member mean, 5-95 % width map of the flow members with 3 marked cells (widest, middle,
              narrowest); per cell the distribution of the flow members vs the linear decoder's members, truth marked
  time.png    the widest cell through consecutive validation launches: flow 5-95 % and 25-75 % bands, linear 5-95 %,
              truth
Run:  python -m src.probes.plot_prediction_distribution --run .tmps/runs_fd/fdB_l32_net \
          --fn .tmps/eval_fn/seed0_lead32 --out .tmps/eval_fd/fdB_l32_net/distribution
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.models.family_net import GRID, DramaticSampler, FamilyNet, launch_features   # noqa: E402
from src.models.family_prior import ETA                    # noqa: E402
from src.models.flow_decoder import FlowDecoder            # noqa: E402
from src.models.frontal_decoder import FrontalDecoder      # noqa: E402
from src.train.fit_frontal_decoder import load_eye         # noqa: E402

COL = dict(fixed="#eb6834", net="#1baf7a", family="#2a78d6", flow="#2a78d6", linear="#eb6834", truth="#1b1b1a",
           grey="#8c8b86")
FAMS = ("stationary", "cyclic", "chaotic", "noise")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="decoder run trained with fd.prior=net")
    ap.add_argument("--fn", required=True, help="family-network dir of the same lead (gauss_net.pt, family.pt)")
    ap.add_argument("--hf_dir", default="/home/sysadmin/jlee/venn/.claude/worktrees/history-forecastor/.tmps/runs_hf")
    ap.add_argument("--obs_seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=256)
    ap.add_argument("--time_launches", type=int, default=150)
    ap.add_argument("--time_members", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dramatic", action="store_true", help="draw from the exaggerated family prior (D-054)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    torch.manual_seed(0)

    pr = dict(np.load(os.path.join(args.run, "prior.npz"), allow_pickle=True))
    npz = np.load(os.path.join(args.run, "netprior.npz"))
    fam = np.load(os.path.join(args.hf_dir, f"twin_seed{args.obs_seed}", "norm.npz"))["fam"]
    z, zhat, a, t_tr = pr["z"], pr["zhat"], int(pr["lead"]), int(pr["t_tr"])
    T, K = z.shape
    sig = np.sqrt(np.diag(pr["Sigma"]))
    ck = torch.load(os.path.join(args.run, "model.pt"), map_location=dev, weights_only=False)
    flow = FlowDecoder(ck["K"], V=ck["V"], ch=tuple(ck["ch"]), emb=ck["emb"], sigma0=ck["sigma0"],
                       source=ck.get("source", "launch"), zmap=ck.get("zmap", 0)).to(dev).eval()
    flow.load_state_dict(ck["state"])
    dk = torch.load(os.path.join(args.run, "decoder.pt"), map_location=dev, weights_only=False)
    lin = FrontalDecoder(dk["K"], dk["lags"], dk["field_shape"], dk["ridge"]).to(dev)
    lin.load_state_dict(dk["state"])
    res_sd = dk["res_var"].to(dev).sqrt()
    _, Y, shape, _, _ = load_eye(ck["encoder_run"], dev)
    X = Y.float().reshape(len(Y), *shape)
    mu, sd = (torch.tensor(pr[k], dtype=torch.float32, device=dev) for k in ("mu", "sd"))
    val = np.arange(t_tr - a, T - a)
    t0 = int(val[len(val) // 2])                                   # the same launch as the eval snapshot
    m_, w_, LR, s_ = (torch.tensor(npz[k], dtype=torch.float32, device=dev) for k in ("m", "w", "chol_R", "sigma"))
    zh_t = torch.tensor(zhat, dtype=torch.float32, device=dev)

    dram = DramaticSampler(args.run, args.hf_dir, args.obs_seed, val, dev) if args.dramatic else None

    def latent_members(t: np.ndarray, M: int) -> torch.Tensor:
        """Joint draws of the decoder's p(z) (network, or the exaggerated prior) for launches t -> [B, M, K]."""
        if dram is not None:
            return dram.sample(t, M)
        eta = torch.randn(len(t), M, K, device=dev) @ LR.T
        return zh_t[t][:, None] + s_[None, None] * (m_[t][:, None] + w_[t][:, None] * eta)

    def decode(zs: torch.Tensor, chunk: int = 512):
        B, M, _ = zs.shape
        flat = zs.reshape(B * M, K)
        fl = torch.cat([flow.sample(None, flat[i:i + chunk], steps=16, shape=tuple(shape))
                        for i in range(0, len(flat), chunk)]).view(B, M, *shape)
        li = (lin(zs * sd + mu) + res_sd.view(*shape) * torch.randn(B, M, *shape, device=dev))
        return fl, li

    # --- 1. latent densities, one launch -----------------------------------------------------------------------------
    common, feats, s, _, _ = launch_features(z, zhat, sig, fam, a, 7750)
    g = GRID.numpy()
    dens = {}
    for name in ("gauss_net", "family"):
        c = torch.load(os.path.join(args.fn, f"{name}.pt"), map_location=dev, weights_only=False)
        net = FamilyNet(fam, c["feat_dims"], c["shapes"]).to(dev)
        net.load_state_dict(c["state"])
        tt = np.array([t0])
        cdf = net.cdf(torch.tensor(np.nan_to_num(common[tt]), dtype=torch.float32, device=dev),
                      {f: torch.tensor(np.nan_to_num(v[tt]), dtype=torch.float32, device=dev) for f, v in feats.items()},
                      torch.tensor(s[tt], dtype=torch.float32, device=dev))[0].cpu().numpy()
        dens[name] = np.gradient(cdf, g, axis=-1)                   # density of the scaled error, [K, G]
    picks = [i for f in FAMS for i in np.flatnonzero(fam == f)[:2]]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for k, i in enumerate(picks):
        A = ax.flat[k]
        x = zhat[t0, i] + sig[i] * g                               # back to canonical z units
        A.plot(x, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi) / sig[i], color=COL["fixed"], ls="--", lw=1.5,
               label="fixed Gaussian")
        A.plot(x, dens["gauss_net"][i] / sig[i], color=COL["net"], lw=2, label="network (decoder's prior)")
        A.plot(x, dens["family"][i] / sig[i], color=COL["family"], lw=1.5, label="network, family shape")
        if dram is not None:                                       # density of the exaggerated prior from its table
            # density from the quantile table: probability between neighbouring knots over their distance, drawn as a
            # step curve so flat tops, soft edges and tails all show
            q = dram.tab[dram.row[t0], i].cpu().numpy() + zhat[t0, i]
            from scipy.special import ndtr
            dp = np.diff(ndtr(ETA)) / np.maximum(np.diff(q), 1e-9)
            A.stairs(dp, q, color="#a8401b", lw=2.2, label="exaggerated (decoded)")
        A.axvline(zhat[t0, i], color=COL["grey"], ls=":", lw=1.2, label="forecast ẑ")
        A.axvline(z[t0 + a, i], color=COL["truth"], lw=1.5, label="true z")
        A.set_xlim(zhat[t0, i] - 4 * sig[i], zhat[t0, i] + 4 * sig[i])
        A.set_title(f"{fam[i]} channel {int(pr['channels'][i])}", fontsize=10)
        A.set_xlabel(f"z(t+{a})  (canonical units)")
        A.spines[["top", "right"]].set_visible(False)
    ax.flat[0].set_ylabel("density")
    ax.flat[4].set_ylabel("density")
    ax.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"estimated p(z(t+{a})) for one validation launch (t = {t0})"
                 + (" -- decoder fed the EXAGGERATED prior" if dram is not None else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "latent.png"), dpi=120)
    plt.close(fig)

    # --- 2. field distribution, one launch -----------------------------------------------------------------------------
    fl, li = decode(latent_members(np.array([t0]), args.members))
    fl, li = fl[0, :, 0].cpu().numpy(), li[0, :, 0].cpu().numpy()   # SSH members [M, H, W]
    truth = X[t0 + a, 0].cpu().numpy()
    q05, q95 = np.quantile(fl, [0.05, 0.95], axis=0)
    width = q95 - q05
    order = np.argsort(width.ravel())
    cells = [np.unravel_index(order[j], width.shape) for j in (len(order) - 1, len(order) // 2, 0)]
    labels = ("widest", "middle", "narrowest")
    fig = plt.figure(figsize=(17, 8.5))
    lim = float(np.abs(truth).max())
    for c, (img, title, cm, lo, hi) in enumerate(((truth, f"truth SSH x(t+{a})", "RdBu_r", -lim, lim),
                                                   (fl.mean(0), "flow member mean", "RdBu_r", -lim, lim),
                                                   (width, "flow 5-95 % width", "Blues", 0, width.max()))):
        A = fig.add_subplot(2, 3, c + 1)
        im = A.imshow(img, cmap=cm, vmin=lo, vmax=hi, origin="lower")
        for (yy, xx), lab in zip(cells, labels):
            A.plot(xx, yy, "o", mfc="none", mec="#1b1b1a", ms=10, mew=1.5)
            A.text(xx + 2, yy + 2, lab, fontsize=8)
        A.set_xticks([])
        A.set_yticks([])
        A.set_title(title, fontsize=10)
        fig.colorbar(im, ax=A, fraction=0.046)
    for c, ((yy, xx), lab) in enumerate(zip(cells, labels)):
        A = fig.add_subplot(2, 3, 4 + c)
        lo_, hi_ = min(fl[:, yy, xx].min(), li[:, yy, xx].min(), truth[yy, xx]), \
            max(fl[:, yy, xx].max(), li[:, yy, xx].max(), truth[yy, xx])
        bins = np.linspace(lo_ - 0.05, hi_ + 0.05, 40)
        A.hist(fl[:, yy, xx], bins=bins, density=True, color=COL["flow"], alpha=0.55, label="flow members")
        A.hist(li[:, yy, xx], bins=bins, density=True, histtype="step", color=COL["linear"], lw=1.8,
               label="linear members")
        A.axvline(truth[yy, xx], color=COL["truth"], lw=2, label="truth")
        A.set_title(f"p(x) at the {lab} cell ({yy}, {xx})", fontsize=10)
        A.set_xlabel("SSH (field units)")
        A.legend(fontsize=7, frameon=False)
        A.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"estimated p(x(t+{a})) for one validation launch (t = {t0}), {args.members} members", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "field.png"), dpi=110)
    plt.close(fig)

    # --- 3. the widest cell through time ---------------------------------------------------------------------------
    yy, xx = cells[0]
    tt = val[len(val) // 2 - args.time_launches // 2: len(val) // 2 + args.time_launches // 2]
    Q = {"flow": [], "linear": []}
    for i0 in range(0, len(tt), 16):
        t = tt[i0:i0 + 16]
        f_, l_ = decode(latent_members(t, args.time_members))
        Q["flow"].append(np.quantile(f_[:, :, 0, yy, xx].cpu().numpy(), [0.05, 0.25, 0.5, 0.75, 0.95], axis=1))
        Q["linear"].append(np.quantile(l_[:, :, 0, yy, xx].cpu().numpy(), [0.05, 0.5, 0.95], axis=1))
    Q = {k: np.concatenate(v, 1) for k, v in Q.items()}
    tru = X[tt + a, 0, yy, xx].cpu().numpy()
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.fill_between(tt + a, Q["flow"][0], Q["flow"][4], color=COL["flow"], alpha=0.2, lw=0, label="flow 5-95 %")
    ax.fill_between(tt + a, Q["flow"][1], Q["flow"][3], color=COL["flow"], alpha=0.35, lw=0, label="flow 25-75 %")
    ax.plot(tt + a, Q["flow"][2], color=COL["flow"], lw=1, label="flow median")
    ax.plot(tt + a, Q["linear"][0], color=COL["linear"], lw=0.9, ls="--", label="linear 5-95 %")
    ax.plot(tt + a, Q["linear"][2], color=COL["linear"], lw=0.9, ls="--")
    ax.plot(tt + a, tru, color=COL["truth"], lw=1.3, label="truth")
    inside = np.mean((tru >= Q["flow"][0]) & (tru <= Q["flow"][4]))
    ax.set_title(f"SSH at cell ({yy}, {xx}), lead {a}: estimated distribution over {len(tt)} launches "
                 f"(truth inside flow 5-95 %: {100 * inside:.0f} %)", fontsize=10)
    ax.set_xlabel("target time step")
    ax.legend(fontsize=8, frameon=False, ncol=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "time.png"), dpi=120)
    plt.close(fig)
    print(f"[dist] lead {a}, launch {t0}: cells {cells}; truth inside flow 5-95 % at the widest cell {100 * inside:.0f} %")


if __name__ == "__main__":
    main()
