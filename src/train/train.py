"""VENN v0 training: learn selection masks with slowness + whitening (SOP 02, D-015).

Online pair-minibatch SGD. All hyperparameters come from ./config/config.yaml (Hydra, D-014).
Artifacts + TensorBoard logs are written under ./.tmps/runs/<timestamp>/ (Hydra run dir).

Run:  conda run -n oceanai python -m src.train.train
      conda run -n oceanai python -m src.train.train train.max_steps=400 train.lambda_white=5
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

# make `src` importable even though Hydra changes the working directory to the run dir
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import hydra                                              # noqa: E402
from omegaconf import DictConfig, OmegaConf               # noqa: E402
from torch.utils.tensorboard import SummaryWriter         # noqa: E402

from src.data.synthetic import GenConfig, generate_field  # noqa: E402
from src.models.encoder import SelectionEncoder            # noqa: E402
from src.train.spectral import (band_plan, level_term, memory_term, rung_roles,  # noqa: E402
                               spectral_terms, structure_term)


def _anneal(step: int, total: int, t0: float, t1: float) -> float:
    """Geometric temperature schedule t0 -> t1 over `total` steps."""
    frac = min(1.0, step / max(1, total - 1))
    return float(t0 * (t1 / t0) ** frac)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # --- data (hidden-truth generator) ------------------------------------------------------
    gen_cfg = GenConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    field_np, truth = generate_field(gen_cfg, seed=cfg.seed)
    field = torch.from_numpy(field_np).to(device)          # [T,V,H,W]
    T, V, H, W = field.shape
    # time split (SOP 02, D-030): the observer learns only from the PAST [0, t_fit). null = full record.
    # The final S is still encoded over all of T, so module 2 can be scored on days never seen here.
    t_fit = int(cfg.train.t_train) if cfg.train.get("t_train") else T
    field_fit = field[:t_fit]                              # [t_fit,V,H,W] the only data that trains
    fams = [m["family"] for m in truth["modes"]]

    # --- model / optim ----------------------------------------------------------------------
    K = int(cfg.model.K)
    truth_amp = np.stack([m["amp"] for m in truth["modes"]])     # [n_modes,T] hidden answer key
    truth_phi = np.stack([m["phi"] for m in truth["modes"]])     # [n_modes,H,W]
    truth_scale = np.array([m["scale"] for m in truth["modes"]])
    enc = SelectionEncoder(K=K, V=V, H=H, W=W, temp=cfg.model.temp0, norm=cfg.model.norm,
                           init=cfg.model.init, init_std=cfg.model.init_std,
                           sigma_min=cfg.model.sigma_min, sigma_max=cfg.model.sigma_max,
                           signed=bool(cfg.model.signed),
                           generator=torch.Generator().manual_seed(cfg.seed)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=cfg.train.lr)
    eye = torch.eye(K, device=device)
    B = int(cfg.train.batch)
    steps = int(cfg.train.max_steps)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)  # index sampling RNG

    with torch.no_grad():                                  # untrained reference (D-013 baseline)
        masks_init = enc.masks().cpu().numpy()
        # reference energy density: the mean per-cell temporal variance of the field. A channel
        # whose var/count^2 falls below this is reading quieter-than-average cells (D-019).
        e_ref = field_fit.var(dim=0).mean().clamp_min(1e-12)
        cell_mean = field_fit.reshape(t_fit, -1).mean(dim=0, keepdim=True)  # [1,N] for L_recon
        x_ref = ((field_fit.reshape(t_fit, -1) - cell_mean) ** 2).mean()    # field energy scale
        # geometric footprint ladder: channel 0 gets the largest target, channel K-1 the smallest
        # (matches the multiscale init's sigma ladder, reversed so index order reads big -> small)
        n_cells = float(V * H * W)
        tgt_cnt = torch.logspace(np.log10(cfg.model.size_max_frac),
                                 np.log10(cfg.model.size_min_frac), K,
                                 device=device) * n_cells                    # [K] target cell counts
        log_tol = float(np.log(cfg.train.size_tol))

        # --- spectral band ladder (D-024): assign each channel a TIMESCALE, not just a footprint --
        sp = cfg.train.spectral
        use_spec = bool(sp.enable)
        if use_spec:
            roles = rung_roles(K, list(sp.pop_target))
            band_w = band_plan(K, int(sp.win_len), roles, float(sp.slow_period_min),
                               float(sp.fast_period_max), float(sp.cyclic_period_max),
                               device=device)                                     # [K,n_bins]
            is_cyc = torch.tensor([r == "cyclic" for r in roles], device=device)
            is_slow = torch.tensor([r == "slow" for r in roles], device=device)
            is_fast = torch.tensor([r == "fast" for r in roles], device=device)
            # D-025: under `slow_objective: level` the stationary rungs are asked to be CONSTANT, so
            # they must be exempted from every term that presumes a channel fluctuates -- the
            # anti-death hinge (demands unit std), the energy floor (demands temporal variance in
            # the selected cells), and both spectral hinges (scored on a spectrum whose DC bin we
            # deliberately zero). `w_dyn` is 1 for the fluctuating rungs and 0 for the flat ones.
            flat_rungs = str(sp.slow_objective) == "level"
            w_dyn = (~is_slow).float() if flat_rungs else torch.ones(K, device=device)
            # DE-ALIGN the two ladders for the flat rungs. `rung_roles` puts the slow rungs at
            # channel 0..n-1, which the D-020 size ladder pins to the LARGEST footprints (28-32% of
            # the domain). Measured consequence (run 20260901_080506): the flat rungs are genuinely
            # flat (amp_ratio 0.023/0.049) but spatially diffuse -- they read a domain-wide static
            # average, aligning with the constant mode's own pattern at only |corr| 0.28. A static
            # structure has a SIZE of its own, unrelated to its timescale, so give those rungs their
            # own footprint target instead of the biggest rung on the ladder.
            if flat_rungs and float(sp.slow_size_frac) > 0:
                tgt_cnt = torch.where(is_slow, float(sp.slow_size_frac) * n_cells, tgt_cnt)
            print(f"[train] band ladder roles: {roles} | slow_objective={sp.slow_objective}")
        else:
            roles, band_w, is_cyc, is_slow, is_fast = [], None, None, None, None
            flat_rungs, w_dyn = False, torch.ones(K, device=device)

    writer = SummaryWriter(log_dir="tb")                   # under the Hydra run dir (.tmps/runs/..)
    print(f"[train] device={device} K={K} B={B} steps={steps} "
          f"lambda_white={cfg.train.lambda_white} | families={fams}")

    for step in range(steps):
        enc.temp = _anneal(step, steps, cfg.model.temp0, cfg.model.temp1)

        idx = torch.randint(0, t_fit - 1, (B,), generator=g)   # consecutive-pair starts
        xb = field_fit[idx]                                # [B,V,H,W]
        xb1 = field_fit[idx + 1]                           # [B,V,H,W]
        s_t = enc(xb)                                       # [B,K]
        s_tp1 = enc(xb1)                                    # [B,K]

        # batch covariance of centered s_t (raw) -- used by all modes
        mu = s_t.mean(dim=0, keepdim=True)                 # [1,K]
        sc = s_t - mu                                      # [B,K]
        cov = (sc.T @ sc) / (B - 1)                        # [K,K]
        cond = 1.0

        # DIFFERENTIABLE per-channel normalization. The scale factors must NOT be detached: a
        # detached std is a constant to autograd, so scaling every mask down shrinks the loss --
        # exactly how the first `corr` attempt drove var 1e6 -> 1e-40 (D-017).
        v = torch.diagonal(cov).clamp_min(1e-8)            # [K] channel variance (differentiable)
        inv = v.rsqrt()                                    # [K]
        corr = cov * inv[:, None] * inv[None, :]           # [K,K] unit diagonal, scale-invariant
        off = corr - torch.diag(torch.diagonal(corr))
        l_white = (off ** 2).sum() / (K * (K - 1))         # mean squared off-diagonal correlation

        if cfg.train.whitening == "corr":
            # Slowness as a RAYLEIGH QUOTIENT var(ds_i)/var(s_i) plus a decorrelation penalty on
            # the correlation matrix -- i.e. SFA's objective and constraint, both written so they
            # are invariant to mask scale in VALUE AND GRADIENT. Nothing can be won by shrinking
            # or by duplicating a channel, so the K channels must spread out (D-013/D-017).
            d = s_tp1 - s_t
            gap = (d - d.mean(dim=0, keepdim=True)) * inv  # scale-free per-channel gap
            l_slow = (gap ** 2).mean(dim=0).mean()         # = mean_i var(ds_i)/var(s_i)
            loss = cfg.train.lambda_slow * l_slow + cfg.train.lambda_white * l_white
        elif cfg.train.whitening == "hard":
            # ABLATION (fails -- kept for the record, D-017): detached ZCA of the correlation
            # matrix. Scale-free, but because `inv_sqrt` carries no gradient, redundancy is free:
            # duplicate channels leave near-zero eigenvalues that the eps floor never re-inflates,
            # so the whitened gap goes to ~0 with every channel carrying the SAME signal.
            with torch.no_grad():
                evals, evecs = torch.linalg.eigh(corr)     # corr = U diag(evals) U^T
                inv_sqrt = evecs @ torch.diag(
                    evals.clamp_min(cfg.train.white_eps).rsqrt()) @ evecs.T   # R^{-1/2} [K,K]
                cond = float(evals.max() / evals.clamp_min(cfg.train.white_eps).min())
            gap = ((s_tp1 - mu.detach()) - sc) * inv @ inv_sqrt   # whitened signal gap
            l_slow = (gap ** 2).sum(dim=1).mean()
            loss = l_slow
        else:                                              # "soft" ABLATION (collapses, D-015)
            gap = s_tp1 - s_t                              # slowness on RAW s (not scale-free)
            l_slow = (gap ** 2).sum(dim=1).mean()
            loss = l_slow + cfg.train.lambda_white * ((cov - eye) ** 2).sum()

        # anti-death hinge: nothing above forbids a mask going to literally 0 (a dead channel has
        # no signal to be slow or correlated). Require each channel's count-normalized signal to
        # keep unit std -- a VICReg-style one-sided hinge, inactive for any live channel.
        sd_n = (sc / enc.soft_count().clamp_min(1e-6).sqrt()).std(dim=0)      # [K]
        # weighted so a FLAT rung is not required to fluctuate (D-025); w_dyn is all-ones otherwise
        l_var = ((torch.relu(1.0 - sd_n) ** 2) * w_dyn).sum() / w_dyn.sum().clamp_min(1e-6)
        loss = loss + cfg.train.lambda_var * l_var

        # --- energy floor (D-019): keep each kernel ON SIGNAL -----------------------------------
        # e_i = var(s_i)/count_i^2 is the mean pairwise covariance of the selected cells: free of
        # mask SIZE, so it cannot be gamed by growing or shrinking the footprint. Without this the
        # scale-free L_slow walks every kernel into the dead corners of the domain (finding F-6).
        cnt = enc.soft_count().clamp_min(1e-6)                                # [K]
        e_ch = torch.diagonal(cov) / cnt ** 2                                 # [K] energy density
        l_energy = (((torch.relu(1.0 - e_ch / e_ref) ** 2) * w_dyn).sum()
                    / w_dyn.sum().clamp_min(1e-6))       # flat rungs exempt (D-025)
        if cfg.train.lambda_energy:
            loss = loss + cfg.train.lambda_energy * l_energy

        # --- scale ladder (D-020): keep a DIVERSITY OF MASK TYPES ------------------------------
        # Left free, every mask shrinks to a tiny patch (init ~4095 cells -> median ~28), because
        # decorrelation prefers disjoint footprints AND e_i = var/count^2 is maximized by a tiny
        # coherent patch. So the "multi-scale sensor" picture (D-013) decays into 16 small patches.
        # Assign channel i a target footprint on a geometric ladder (small energetic patch ->
        # basin-scale) and penalize only OUTSIDE a tolerance factor, so it guides without pinning.
        l_size = (torch.relu((cnt / tgt_cnt).log().abs() - log_tol) ** 2).mean()
        if cfg.train.lambda_size:
            loss = loss + cfg.train.lambda_size * l_size

        # --- coverage / reconstruction (D-019): do the K channels SPAN the field? ---------------
        # Best linear decode of this batch's field from [1, s], solved in closed form and left
        # differentiable, so l_recon is exactly the unexplained variance fraction (1 - R^2). Also
        # penalizes redundancy: a duplicate channel buys no reduction, so its gradient points at
        # whatever residual is still unexplained.
        if cfg.train.lambda_recon:
            X = xb.reshape(B, -1) - cell_mean                                 # [B,N]
            # decode from STANDARDIZED channels: with raw s (variance ~1e6) the normal-equation
            # matrix has entries ~1e9, `recon_ridge` is negligible against it, and with channels
            # still correlated at ~0.99 the solve is near-singular -> garbage gradients that swamp
            # every other term at any lambda. Standardizing puts G on a unit scale (D-019).
            S1 = torch.cat([torch.ones(B, 1, device=device), sc * inv], dim=1)  # [B,K+1]
            G = S1.T @ S1 + cfg.train.recon_ridge * B * torch.eye(K + 1, device=device)
            Wd = torch.linalg.solve(G, S1.T @ X)                              # [K+1,N]
            l_recon = ((X - S1 @ Wd) ** 2).mean() / x_ref
            loss = loss + cfg.train.lambda_recon * l_recon
        else:
            l_recon = torch.zeros((), device=device)

        # --- spectral band ladder (D-024) --------------------------------------------------------
        # Needs CONTIGUOUS time, which the pair minibatch above cannot give: a lag-1 pair says
        # nothing about a period-300 cycle. Encoding the whole series is ~free here (one einsum over
        # T=2000 x 8192 cells), so we do that and slice `n_win` random windows out of it -- the
        # sampling noise that keeps the updates stochastic (D-011) comes from the window starts.
        if use_spec:
            S_all = enc(field_fit)                                            # [t_fit,K]
            L = int(sp.win_len)
            w0 = torch.randint(0, t_fit - L + 1, (int(sp.n_win),), generator=g)   # window starts
            S_win = torch.stack([S_all[s0:s0 + L] for s0 in w0.tolist()])     # [n_win,L,K]
            # WHICH SHAPE TERM decides cyclic-vs-chaotic (D-027). `structure` speaks the readout's
            # own language (trend+osc vs residual, cut at 0.5) and is the default; `line` is the
            # D-024 original, kept as an ablation -- its cap sat at a linefrac of 0.75 and was
            # measured SILENT on all nine fast rungs, which is why they drifted to the boundary.
            use_struct = str(sp.shape_objective) == "structure"
            # `struct_rungs: fast` (DEFAULT) applies L_struct to the FAST rungs only and leaves the
            # cyclic rungs on L_line. Measured (seed 0): with L_struct on the cyclic rungs too, all
            # five of them collapsed onto the period-60 cycle (baseline: 286/143/61/61/61), because
            # `trend+osc` credits ANY clean peak and period 60 is the cleanest line in the field.
            # The cap on the fast side has no such pull -- "be broadband" favours no mode.
            struct_fast_only = use_struct and str(sp.struct_rungs) == "fast"
            line_w = None
            if use_struct:
                # L_line stays on the cyclic rungs only when L_struct owns the fast ones; with
                # `struct_rungs: all` it is off everywhere (the first D-027 variant)
                line_w = (is_cyc.float() * w_dyn) if struct_fast_only else torch.zeros(K, device=device)
            l_band, l_line, sp_diag = spectral_terms(
                S_win, band_w, is_cyc, float(sp.band_target), float(sp.line_target),
                float(sp.line_cap), chan_w=w_dyn, line_w=line_w)
            loss = loss + sp.lambda_band * l_band
            if use_struct:
                cyc_for_struct = torch.zeros_like(is_cyc) if struct_fast_only else is_cyc
                l_struct, struct_sh = structure_term(S_all, cyc_for_struct, is_fast,
                                                     float(sp.struct_target), float(sp.struct_cap))
                loss = loss + sp.lambda_struct * l_struct
                if struct_fast_only:
                    loss = loss + sp.lambda_line * l_line
            else:
                l_struct, struct_sh = torch.zeros((), device=device), None
                loss = loss + sp.lambda_line * l_line
            if flat_rungs:
                # STATIONARY = CONSTANT (D-025): the slow rungs are driven by flatness, measured on
                # the full series, since "constant" is a claim about all of T.
                l_level, r_flat = level_term(S_all, is_slow, float(sp.flat_target))
                l_mem, rho_lag = torch.zeros((), device=device), None
                loss = loss + sp.lambda_level * l_level
            else:
                # `slow_objective: memory` -- the earlier reading of stationary as a slow DRIFT.
                # Kept as an ablation; the user's definition is `level` (see D-025).
                l_mem, rho_lag = memory_term(S_win, is_slow, int(sp.mem_lag), float(sp.mem_target))
                l_level, r_flat = torch.zeros((), device=device), None
                loss = loss + sp.lambda_mem * l_mem
        else:
            l_band = l_line = l_mem = l_level = l_struct = torch.zeros((), device=device)
            sp_diag, rho_lag, r_flat, struct_sh = None, None, None, None

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(enc.parameters(), float(cfg.train.grad_clip))
        opt.step()

        if step % int(cfg.train.log_every) == 0 or step == steps - 1:
            with torch.no_grad():
                var = torch.diagonal(cov)                  # per-channel RAW variance [K]
                slow_ch = (gap ** 2).mean(dim=0)           # per-channel slowness (whitened if hard)
            writer.add_scalar("loss/total", loss.item(), step)
            writer.add_scalar("loss/slow", l_slow.item(), step)
            writer.add_scalar("loss/white", l_white.item(), step)
            writer.add_scalar("loss/var_hinge", l_var.item(), step)
            writer.add_scalar("loss/energy", l_energy.item(), step)
            writer.add_scalar("loss/size", l_size.item(), step)
            writer.add_scalar("mask/count_min", cnt.min().item(), step)
            writer.add_scalar("mask/count_max", cnt.max().item(), step)
            writer.add_scalar("loss/recon", l_recon.item(), step)
            writer.add_scalar("loss/band", l_band.item(), step)
            writer.add_scalar("loss/line", l_line.item(), step)
            writer.add_scalar("loss/mem", l_mem.item(), step)
            writer.add_scalar("loss/level", l_level.item(), step)
            writer.add_scalar("loss/struct", l_struct.item(), step)
            if struct_sh is not None:
                # the readout's own cyclic/chaotic vote, live: cyclic rungs should climb toward 1,
                # fast rungs fall toward 0, and neither should linger at the 0.5 boundary (D-027)
                writer.add_scalar("struct/cyclic_min", struct_sh[is_cyc].min().item(), step)
                writer.add_scalar("struct/fast_max", struct_sh[is_fast].max().item(), step)
            if r_flat is not None:
                writer.add_scalar("band/flat_ratio_max", r_flat.max().item(), step)
            if rho_lag is not None:
                writer.add_scalar("band/rho_lag_max", rho_lag.max().item(), step)
            if sp_diag is not None:
                bf, lf = sp_diag["bandfrac"], sp_diag["linefrac"]
                writer.add_scalar("band/frac_min", bf.min().item(), step)
                writer.add_scalar("band/frac_mean", bf.mean().item(), step)
                writer.add_scalar("band/line_min", lf.min().item(), step)
                writer.add_scalar("band/line_max", lf.max().item(), step)
            writer.add_scalar("energy/density_min", (e_ch / e_ref).min().item(), step)
            writer.add_scalar("energy/density_max", (e_ch / e_ref).max().item(), step)
            writer.add_scalar("mask/count_mean", enc.soft_count().mean().item(), step)
            writer.add_scalar("var/mean", var.mean().item(), step)
            writer.add_scalar("var/min", var.min().item(), step)
            writer.add_scalar("var/max", var.max().item(), step)
            writer.add_scalar("slow_per_ch/min", slow_ch.min().item(), step)
            writer.add_scalar("slow_per_ch/max", slow_ch.max().item(), step)
            writer.add_scalar("temp", enc.temp, step)
            writer.add_scalar("cond", cond, step)
            writer.add_scalar("grad_norm", float(gnorm), step)
            print(f"  step {step:5d} | L={loss.item():.4f} slow={l_slow.item():.5f} "
                  f"offcorr2={l_white.item():.4f} energy={l_energy.item():.4f} "
                  f"recon={l_recon.item():.4f} size={l_size.item():.3f} "
                  f"band={l_band.item():.4f} line={l_line.item():.4f} "
                  f"mem={l_mem.item():.4f} level={l_level.item():.4f} "
                  f"struct={l_struct.item():.4f} | "
                  f"e/e_ref[{(e_ch/e_ref).min():.2f},{(e_ch/e_ref).max():.2f}] "
                  f"|g|={float(gnorm):.1e} temp={enc.temp:.2f}")

    # --- final artifacts (ephemeral; promotion to results/ needs user OK) --------------------
    with torch.no_grad():
        S = enc(field).cpu().numpy()                       # [T,K] full scalar series
        masks = enc.masks().cpu().numpy()                  # [K,V,H,W]
    np.savez("artifacts.npz", S=S, masks=masks, masks_init=masks_init, families=np.array(fams),
             truth_amp=truth_amp, truth_phi=truth_phi, truth_scale=truth_scale,
             temp_final=np.float32(enc.temp),
             roles=np.array(roles))       # per-channel assigned role, for the probe to check against
    metrics = dict(final_band=float(l_band.item()), final_line=float(l_line.item()),
                   final_mem=float(l_mem.item()), final_level=float(l_level.item()),
                   final_struct=float(l_struct.item()),
                   final_loss=float(loss.item()), final_slow=float(l_slow.item()),
                   final_energy=float(l_energy.item()), final_recon=float(l_recon.item()),
                   final_size=float(l_size.item()),
                   mask_count_min=float(cnt.min().item()), mask_count_max=float(cnt.max().item()),
                   final_white=float(l_white.item()), var_mean=float(var.mean().item()),
                   var_min=float(var.min().item()), var_max=float(var.max().item()),
                   slow_ch_min=float(slow_ch.min().item()), slow_ch_max=float(slow_ch.max().item()))
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    writer.close()
    print(f"[train] done. run dir: {os.getcwd()}")
    print(f"[train] metrics: {metrics}")


if __name__ == "__main__":
    main()
