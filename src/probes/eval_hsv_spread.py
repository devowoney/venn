"""Spread from Hessian (B-norm) singular vectors on the deterministic twin — D-043, SOP 04, memory/hsv_spread_design.md.

Per seed:
  1. B: rebuild the observer's raw field; smooth part of its one-step change on the fitting range -> B_x (rule);
     B_z = E B_x Eᵀ through the live layer-2 masks, in the forecaster's units.
  2. Singular vectors at every launch: tangent-linear of the chunk-16 rollout at lead a* = 16 (exact JVPs),
     SVD of L B_z^{1/2}; top n start patterns v_i.
  3. Members: rollouts (chunk 16) from z_t ± α Σ ξ_i v_i, 32 members; ONE α per seed = min fair CRPS on launches whose
     targets lie in the held-out [t_fit, t_tr) (never validation).
  4. Readout on validation: the standard ensemble readout (eval_history_forecaster.report) + growth spectrum σ_i(lead),
     family loading of start / evolved patterns, share of the real lead-a* error inside the evolved patterns.

Run:  python -m src.probes.eval_hsv_spread --runs .tmps/runs_hf/twin_seed0 ... --out .tmps/eval_hf/hsv/spread
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

from src.data.synthetic import GenConfig, generate_field, normalize_input   # noqa: E402
from src.models.hsv_spread import (CachedRollout, field_change_stats, hsv_members, latent_b,
                                   measured_eye_scale, singular_vectors, sqrt_psd, tangent_linear)   # noqa: E402
from src.probes.eval_history_forecaster import COL, FAMS, eval_one, load_run, report   # noqa: E402
from src.train.train_history_forecaster import fair_crps   # noqa: E402


def observer_field(encoder_run: str, art) -> np.ndarray:
    """The raw field exactly as the observer saw it (same data config, seed and input normalization)."""
    cfg = OmegaConf.load(os.path.join(encoder_run, ".hydra", "config.yaml"))
    x, _ = generate_field(GenConfig(**OmegaConf.to_container(cfg.data, resolve=True)), seed=int(cfg.seed))
    return normalize_input(x, cfg.train.get("input_norm", "none"), int(art["t_fit"]))


def svs_for(cr, t0: torch.Tensor, Bh: torch.Tensor, a_star: int, chunk: int, n: int, batch: int = 128):
    """Singular vectors for many launches, in batches. Returns v [N,K,n], sigma [N,n], U [N,K,n]."""
    out = [singular_vectors(tangent_linear(cr, c, a_star, chunk), Bh, n) for c in torch.split(t0, batch)]
    return tuple(torch.cat([o[i] for o in out]) for i in range(3))


def members_for(cr, t0, v, alpha, xi, steps, chunk, batch: int = 64) -> torch.Tensor:
    return torch.cat([hsv_members(cr, c, vc, alpha, xi, steps, chunk)
                      for c, vc in zip(torch.split(t0, batch), torch.split(v, batch))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="deterministic twin runs (D-043 Q6a)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_sv", type=int, default=8)
    ap.add_argument("--a_star", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--alpha_grid", default="-1.5,1.5,25", help="log10 lo, log10 hi, count")
    ap.add_argument("--diag_launches", type=int, default=128, help="launches for the growth spectrum")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    lo, hi, ng = args.alpha_grid.split(",")
    grid = np.logspace(float(lo), float(hi), int(ng))
    spec_leads = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    R, D, lines = [], [], []

    for run in args.runs:
        ck = torch.load(os.path.join(run, "model.pt"), map_location="cpu", weights_only=False)
        nm = np.load(os.path.join(run, "norm.npz"))
        art = np.load(os.path.join(ck["encoder_run"], "artifacts.npz"), allow_pickle=True)
        t_fit, t_tr = int(nm["t_fit"]), int(nm["t_tr"])
        ch = nm["channels"]
        fams = [str(f) for f in nm["fam"]]
        K = len(ch)

        # --- 1. B ---------------------------------------------------------------------------------------------------
        x = observer_field(ck["encoder_run"], art)
        S_chk = np.einsum("tvhw,kvhw->tk", x[:200].astype(np.float64), art["masks"][ch].astype(np.float64))
        rebuild_err = float(np.abs(S_chk - art["S"][:200, ch]).max() / art["S"][:, ch].std())
        sigma, ell, nugget, a = field_change_stats(x[:t_fit])
        del x
        Bz = latent_b(art["masks"][ch], nm["sd"], sigma, ell, dev)
        rule_std = Bz.diagonal().sqrt().cpu().numpy()
        z_fit = (art["S"][:t_fit, ch].astype(np.float64) - nm["mu"]) / nm["sd"]
        Bz, meas_std = measured_eye_scale(Bz, z_fit, art["masks"][ch], nm["sd"], nugget)   # B-fix (a)
        Bh = sqrt_psd(Bz)
        lam = torch.linalg.eigvalsh(Bz).flip(0).cpu().numpy()
        bdiag = Bz.diagonal().cpu().numpy()

        # --- the dynamics and its rollouts -----------------------------------------------------------------------------
        model, _, _, _ = load_run(run, dev)
        S = art["S"][:, ch].astype(np.float64)
        z = torch.tensor((S - nm["mu"]) / nm["sd"], dtype=torch.float32, device=dev)
        T, A = z.shape[0], model.A
        cr = CachedRollout(model, z)
        g = torch.Generator(device=dev).manual_seed(0)
        xi = torch.randn(args.members // 2, args.n_sv, device=dev, generator=g)   # one mixing for every launch and α

        # --- 3a. α on the held-out slice ---------------------------------------------------------------------------------
        Lh = torch.arange(t_fit - 1, t_tr - 1, device=dev)
        idx = Lh[:, None] + torch.arange(1, A + 1, device=dev)[None]
        ok = (idx < t_tr).float()                                       # [Nh,A] targets inside the held-out slice
        yh = z[idx.clamp_max(T - 1)]
        vh, _, _ = svs_for(cr, Lh, Bh, args.a_star, args.chunk, args.n_sv)
        crps_a = []
        with torch.no_grad():
            for al in grid:
                e = members_for(cr, Lh, vh, float(al), xi, A, args.chunk)            # [Nh,M,A,K]
                c = fair_crps(e.permute(0, 2, 3, 1), yh)                             # [Nh,A,K]
                crps_a.append(float((c * ok[..., None]).sum() / (ok.sum() * K)))
        alpha = float(grid[int(np.argmin(crps_a))])
        edge = int(np.argmin(crps_a)) in (0, len(grid) - 1)

        # --- 2 + 3b. validation members, with the singular-vector diagnostics kept aside ------------------------------
        diag = {}

        def ensemble_fn(model_, nm_, art_, zt, L):
            t0 = torch.as_tensor(L, device=dev)
            v, sv, U = svs_for(cr, t0, Bh, args.a_star, args.chunk, args.n_sv)
            with torch.no_grad():
                ens = members_for(cr, t0, v, alpha, xi, A, args.chunk)
                center = cr.rollout(t0, z[t0], args.a_star, chunk=args.chunk)[0][:, -1]   # unperturbed, lead a*
            err = z[t0 + args.a_star] - center                                          # real error at a*
            proj = (torch.einsum("nki,nk->ni", U, err) ** 2).sum(1) / (err ** 2).sum(1)
            fam_idx = {f: [i for i, ff in enumerate(fams) if ff == f] for f in FAMS}
            vn = v ** 2 / (v ** 2).sum(1, keepdim=True)                                 # [N,K,n] share per channel
            un = U ** 2                                                                  # unit columns already
            diag.update(proj=float(proj.mean()), proj_chance=args.n_sv / K,
                        load_v={f: float(vn[:, c].sum(1).mean()) for f, c in fam_idx.items() if c},
                        load_u={f: float(un[:, c].sum(1).mean()) for f, c in fam_idx.items() if c},
                        chance={f: len(c) / K for f, c in fam_idx.items() if c},
                        sigma_astar=sv.mean(0).cpu().numpy())
            return ens.cpu().numpy()

        r = eval_one(run, dev, args.members, ensemble_fn=ensemble_fn)

        # --- 4. growth spectrum σ_i(lead) on a subset of validation launches -------------------------------------------
        Ls = torch.as_tensor(r["L"][np.linspace(0, len(r["L"]) - 1, args.diag_launches).astype(int)], device=dev)
        spec = np.stack([singular_vectors(tangent_linear(cr, Ls, a_, args.chunk), Bh, args.n_sv)[1].mean(0).cpu().numpy()
                         for a_ in spec_leads])                                  # [leads, n]
        diag.update(spec=spec, alpha=alpha, crps_a=np.array(crps_a), ell=ell, nugget=nugget, a=a,
                    rebuild_err=rebuild_err, lam=lam, bdiag=bdiag, fams=fams, edge=edge)
        R.append(r)
        D.append(diag)
        name = os.path.basename(run)
        lines += [f"{name}: field rebuild err {rebuild_err:.1e} | B_x: ell {ell:.2f} cells, smooth share a "
                  f"{np.round(a, 2).tolist()}, nugget {np.round(nugget, 4).tolist()} | B_z eig top/bottom "
                  f"{lam[0]:.3g}/{lam[-1]:.3g} | alpha {alpha:.3g}{' (GRID EDGE)' if edge else ''}",
                  f"   start-error std per eye (model units), rule -> measured: " + ", ".join(
                      f"{f} {rule_std[[i for i, ff in enumerate(fams) if ff == f]].mean():.3f} -> "
                      f"{meas_std[[i for i, ff in enumerate(fams) if ff == f]].mean():.3f}"
                      for f in FAMS if f in fams),
                  f"   growth sigma_1..{args.n_sv} at a*={args.a_star}: {np.round(diag['sigma_astar'], 2).tolist()}",
                  f"   error at a* inside span(U_1..{args.n_sv}): {diag['proj']:.3f} (chance {diag['proj_chance']:.3f})",
                  "   loading (start v / evolved U / chance): " + ", ".join(
                      f"{f} {diag['load_v'][f]:.2f}/{diag['load_u'][f]:.2f}/{diag['chance'][f]:.2f}" for f in diag["chance"])]
        print("\n".join(lines[-5:]), flush=True)

    report(R, [], args.out)
    txt = "HSV spread (D-043): per-seed B, alpha and singular-vector diagnostics\n\n" + "\n".join(lines)
    open(os.path.join(args.out, "hsv_summary.txt"), "w").write(txt + "\n")
    print(txt)
    diag_figure(D, spec_leads, grid, os.path.join(args.out, "hsv_diag.png"))


def diag_figure(D: list, spec_leads: list, grid: np.ndarray, out: str) -> None:
    """Row of four: CRPS vs α (held-out), growth spectrum, family loading of the SVs, error projection vs chance."""
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.2))
    ax = axs[0]
    for i, d in enumerate(D):
        ax.plot(grid, d["crps_a"], "o-", ms=3, label=f"seed {i}: α = {d['alpha']:.3g}")
    ax.set_xscale("log")
    ax.set_title("held-out CRPS vs perturbation size α (one α per seed)")
    ax.set_xlabel("α"), ax.set_ylabel("fair CRPS (model units)"), ax.legend(fontsize=8)
    ax = axs[1]
    for i, d in enumerate(D):
        for j in range(d["spec"].shape[1]):
            ax.plot(spec_leads, d["spec"][:, j], color=plt.cm.viridis(j / d["spec"].shape[1]), lw=1.2, alpha=0.6,
                    label=f"σ_{j + 1}" if i == 0 else None)
    lt = np.array([1, 64])
    ax.plot(lt, np.exp(lt / 22.0), "k:", lw=1, label="e-folding 22 steps (chaos)")
    ax.set_yscale("log")
    ax.set_title("growth of the singular vectors vs lead (mean over launches)")
    ax.set_xlabel("lead"), ax.set_ylabel("σ_i (growth of a 1-sigma B start error)"), ax.legend(fontsize=7, ncol=2)
    ax = axs[2]
    fams = [f for f in FAMS if f in D[0]["chance"]]
    xs = np.arange(len(fams))
    for k, (key, lab, off) in enumerate([("chance", "share of channels", -0.27), ("load_v", "start patterns v", 0.0),
                                         ("load_u", "evolved patterns U", 0.27)]):
        vals = [np.mean([d[key].get(f, np.nan) for d in D]) for f in fams]
        ax.bar(xs + off, vals, 0.27, label=lab, color=[COL[f] for f in fams], alpha=[0.25, 0.6, 1.0][k])
    ax.set_xticks(xs, fams)
    ax.set_title("where the singular vectors sit (energy share per family)")
    ax.legend(fontsize=8)
    ax = axs[3]
    ax.bar(np.arange(len(D)), [d["proj"] for d in D], color="tab:red", label="real error at a* in span(U)")
    ax.plot(np.arange(len(D)), [d["proj_chance"] for d in D], "k_", ms=30, mew=2, label="chance (n/K)")
    ax.set_xticks(np.arange(len(D)), [f"seed {i}" for i in range(len(D))])
    ax.set_ylim(0, 1)
    ax.set_title("does the real error come from the singular vectors?")
    ax.legend(fontsize=8)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"[hsv] wrote {out}")


if __name__ == "__main__":
    main()
