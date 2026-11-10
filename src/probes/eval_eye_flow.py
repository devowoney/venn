"""Evaluate the per-eye flow forecaster on the VALIDATION set (D-044, SOP 04 "Per-eye flow forecaster").

Every launch t in validation whose 64 targets lie in [t_tr, T), every live eye: the eye's own last 256 values go in,
32 futures come out. The members are then read exactly like any other forecaster's ensemble (`score` + `report` of
`eval_history_forecaster.py`: RMSE of the members' mean by lead, spread, spread/error, CRPS, roughness, per-launch
spread vs error, fans). Added here, the PROBABILITY itself:
  - NLL of the truth under the flow (exact change of variables, nats per lead, model z units), per family;
    reference = a diagonal Gaussian with the members' own per-lead mean and std (what modelling the leads jointly buys);
  - rank histograms of the truth among the members at leads 1 / 16 / 64 (flat = calibrated, U = too narrow,
    hump = too wide).

Run:  python -m src.probes.eval_eye_flow --runs .tmps/runs_ef/flow_seed0 --out .tmps/eval_ef/flow
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.models.eye_flow import EyeFlow                    # noqa: E402
from src.probes.eval_history_forecaster import COL, FAMS, report, score   # noqa: E402


def load_flow(run: str, device: str):
    """The trained flow (best held-out checkpoint), its normalization and the observer artifacts it read."""
    ck = torch.load(os.path.join(run, "model.pt"), map_location=device, weights_only=False)
    c = ck["cfg"]
    model = EyeFlow(c["hist_len"], c["leads"], c["hist_patch"], c["lead_patch"], c["d"], c["enc_layers"],
                    c["dec_layers"], c["heads"], c["dropout"], c.get("target_scale", "none")).to(device)
    state = ck["state"]
    if "lead_scale" not in state:                                      # step-1 runs, saved before the target scale
        state = {**state, "lead_scale": torch.ones(c["leads"])}
    model.load_state_dict(state)
    model.eval()
    nm = np.load(os.path.join(run, "norm.npz"))
    art = np.load(os.path.join(ck["encoder_run"], "artifacts.npz"), allow_pickle=True)
    return model, nm, art, c, ck["step"]


def members_all(model: EyeFlow, z: torch.Tensor, L: np.ndarray, M: int, steps: int, chunk: int = 1024,
                seed: int = 0) -> np.ndarray:
    """[N,M,A,K] futures (model units, absolute = z_t + increments) of every eye at every launch in L."""
    T, K = z.shape
    A, H = model.A, model.L
    out = np.zeros((len(L), M, A, K), dtype=np.float32)
    torch.manual_seed(seed)
    Lt = torch.as_tensor(L, device=z.device)
    for k in range(K):
        for i in range(0, len(L), chunk):
            t = Lt[i:i + chunk]
            hist = z[t[:, None] + torch.arange(-H + 1, 1, device=z.device), k]   # the eye's own last H values
            y = model.sample(hist, M, steps)                           # [n,M,A] increments
            out[i:i + chunk, :, :, k] = (hist[:, -1:, None] + y).cpu().numpy()
    return out


def causality(model: EyeFlow, z: torch.Tensor, t_cut: int, M: int, steps: int) -> float:
    """Max |change| of the members launched at t < t_cut when every state from t_cut on is replaced by noise
    (same member noise both times). The window ends at t, so this must be exactly 0."""
    t = torch.arange(t_cut - 64, t_cut, device=z.device)
    zp = z.clone()
    zp[t_cut:] = 5.0 * torch.randn_like(zp[t_cut:])
    off = torch.arange(-model.L + 1, 1, device=z.device)
    torch.manual_seed(1)
    a = model.sample(z[t[:, None] + off, 0], M, steps)
    torch.manual_seed(1)
    b = model.sample(zp[t[:, None] + off, 0], M, steps)
    return float((a - b).abs().max())


def density(model: EyeFlow, z: torch.Tensor, L: np.ndarray, ens: np.ndarray, steps: int, probes: int,
            chunk: int = 512):
    """NLL of the true future per (launch, eye), nats per lead, for the flow and the members' diagonal Gaussian."""
    T, K = z.shape
    A, H = model.A, model.L
    nll_f = np.zeros((len(L), K))
    nll_g = np.zeros((len(L), K))
    Lt = torch.as_tensor(L, device=z.device)
    off_h = torch.arange(-H + 1, 1, device=z.device)
    off_f = torch.arange(1, A + 1, device=z.device)
    for k in range(K):
        for i in range(0, len(L), chunk):
            t = Lt[i:i + chunk]
            hist = z[t[:, None] + off_h, k]
            y = z[t[:, None] + off_f, k] - hist[:, -1:]
            nll_f[i:i + chunk, k] = (-model.log_prob(hist, y, steps, probes) / A).cpu().numpy()
    truth = z.cpu().numpy()[L[:, None] + np.arange(1, A + 1)[None, :]]   # [N,A,K]
    mu, sd = ens.mean(1), ens.std(1, ddof=1) + 1e-6                     # [N,A,K]
    nll_g = (0.5 * ((truth - mu) / sd) ** 2 + np.log(sd) + 0.5 * math.log(2 * math.pi)).mean(1)
    return nll_f, nll_g


def ranks(ens: np.ndarray, truth: np.ndarray, leads=(1, 16, 64)) -> dict:
    """Rank of the truth among the M members (0..M) at a few leads -> {lead: [N,K]}."""
    return {a: (ens[:, :, a - 1] < truth[:, None, a - 1]).sum(1) for a in leads}


def prob_figure(P: list, M: int, out: str) -> None:
    """Left: NLL per family, flow vs the members' diagonal Gaussian. Right: rank histograms at h1 / h16 / h64."""
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.6))
    ax = axs[0]
    # every seed has its own eyes (38..52 live children): each record carries its own family list
    present = [f for f in FAMS if any(f in p["fams"] for p in P)]
    xs = np.arange(len(present))
    for j, (key, lab, alpha) in enumerate([("nll_f", "flow (joint, exact)", 1.0),
                                           ("nll_g", "diag. Gaussian of the members", 0.4)]):
        v = [np.mean([p[key][:, [i for i, f in enumerate(p['fams']) if f == fam]].mean() for p in P
                      if fam in p['fams']]) for fam in present]
        ax.bar(xs + (j - 0.5) * 0.38, v, 0.38, color=[COL[f] for f in present], alpha=alpha, label=lab)
    ax.set_xticks(xs, present)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("NLL of the truth (nats per lead, z units)")
    ax.set_title("probability of what really happened (lower = better)")
    ax.legend(fontsize=8)
    for ax, a in zip(axs[1:], (1, 16, 64)):
        for fam in present:
            r = np.concatenate([p["rank"][a][:, [i for i, f in enumerate(p["fams"]) if f == fam]].ravel() for p in P])
            h = np.bincount(r, minlength=M + 1) / len(r) * (M + 1)
            ax.step(np.arange(M + 1), h, where="mid", color=COL[fam], lw=1.8, label=fam)
        ax.axhline(1, color="k", ls="--", lw=1, label="calibrated")
        ax.set_ylim(0, None)
        ax.set_title(f"rank of the truth among {M} members, lead {a}\n(flat = calibrated, U = too narrow, ∩ = too wide)")
        ax.set_xlabel("rank"), ax.set_ylabel("density / uniform")
        ax.legend(fontsize=7)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[ef] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nll_every", type=int, default=8, help="density on every n-th validation launch (cost)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R, P, lines = [], [], []
    for run in args.runs:
        model, nm, art, c, best_step = load_flow(run, args.device)
        S = art["S"][:, nm["channels"]].astype(np.float64)
        T, K = S.shape
        A, M, steps, t_tr = int(c["leads"]), int(c["members"]), int(c["ode_steps"]), int(nm["t_tr"])
        z = (S - nm["mu"]) / nm["sd"]
        zt = torch.tensor(z, dtype=torch.float32, device=args.device)
        L = np.arange(t_tr - 1, T - A)                                 # every target in validation
        ens = members_all(model, zt, L, M, steps)                      # [N,M,A,K]
        zhat = np.full((T - 1, A, K), np.nan, dtype=np.float32)        # best estimate = the members' mean ...
        zhat[L] = ens.mean(1)                                          # ... defined at the scored launches only
        caus = causality(model, zt, t_tr + 100, M, steps)
        r = score(run, nm, art, t_tr, S, z, L, ens, zhat, caus, True)
        R.append(r)
        truth = z[L[:, None] + np.arange(1, A + 1)[None, :]]
        Ln = L[::args.nll_every]
        nll_f, nll_g = density(model, zt, Ln, ens[::args.nll_every], steps, int(c["probes"]))
        fams = r["fams"]
        P.append(dict(nll_f=nll_f, nll_g=nll_g, rank=ranks(ens, truth), fams=fams))
        lines.append(f"{os.path.basename(run)}: best held-out checkpoint at step {best_step}; causality {caus:.1e}; "
                     f"density on {len(Ln)} launches x {K} eyes")
        for fam in FAMS:
            sel = [i for i, f in enumerate(fams) if f == fam]
            if sel:
                lines.append(f"  {fam:<10} NLL nats/lead: flow {nll_f[:, sel].mean():+.3f}  "
                             f"members' diag Gaussian {nll_g[:, sel].mean():+.3f}")
        del ens
    report(R, [], args.out)
    txt = "Per-eye flow forecaster (D-044): probability of the truth\n\n" + "\n".join(lines)
    open(os.path.join(args.out, "flow_summary.txt"), "w").write(txt + "\n")
    print(txt)
    # keep the probability readout so the figure can be redrawn without re-sampling
    np.savez(os.path.join(args.out, "prob.npz"), **{f"s{i}_{k}": (np.array(v) if k != "rank" else np.stack(list(v.values())))
                                                    for i, p in enumerate(P) for k, v in p.items()})
    prob_figure(P, M, os.path.join(args.out, "flow_prob.png"))


if __name__ == "__main__":
    main()
