"""Train the flow-matching decoder (D-048, SOP 05 "Flow-matching decoder") on the TRAINING set only.

Input : the deterministic twin `twin_seed{s}` and its cross-fitted folds `fold{k}_seed{s}` (module 2, frozen), the
        observer they read (artifacts.npz) and its field (regenerated from the observer's config + seed).
Output (Hydra run dir .tmps/runs_fd/<stamp>/):
        prior.npz  the Gaussian latent forecast: out-of-fold lead-a forecasts on [0, t_tr), the twin's validation
                   forecasts, the true z, Sigma_a and its Cholesky factor (canonical z units, see `latent_prior`)
        decoder.pt linear ridge decoder on the same live channels + its per-cell residual variance (the baseline)
        model.pt   flow decoder, EMA weights of the best held-out checkpoint;  tb/  TensorBoard
Run:  python -m src.train.train_flow_decoder fd.obs_seed=0
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import hydra                                              # noqa: E402
from omegaconf import DictConfig, OmegaConf               # noqa: E402
from torch.utils.tensorboard import SummaryWriter         # noqa: E402

from src.models.family_net import DramaticSampler, net_prior   # noqa: E402
from src.models.flow_decoder import FlowDecoder           # noqa: E402
from src.models.frontal_decoder import FrontalDecoder     # noqa: E402
from src.probes.eval_history_forecaster import load_run as load_hf   # noqa: E402
from src.train.fit_frontal_decoder import load_eye        # noqa: E402


@torch.no_grad()
def forecast_lead(run: str, S_live: torch.Tensor, lead: int, device: str) -> np.ndarray:
    """Lead-`lead` forecast [T, K] of a frozen forecaster run, in PHYSICAL latent units; row t = made at t from S_0..S_t.
    One causal pass over the record (each run has its own fitting-range mu / sd)."""
    model, nm, _, _ = load_hf(run, device)
    mu, sd = (torch.tensor(nm[k], dtype=torch.float32, device=device) for k in ("mu", "sd"))
    zhat = model(((S_live - mu) / sd)[None])[0, :, lead - 1]
    return (zhat * sd + mu).double().cpu().numpy()


def latent_prior(fd, S_all: np.ndarray, device: str) -> dict:
    """The Gaussian latent forecast z(t+a) ~ N(zhat(t), Sigma_a), in CANONICAL z units (live channels standardized on
    [0, t_tr)). zhat on the training set is OUT-OF-FOLD: fold k (targets of its block never fitted, D-045) forecasts the
    launches whose lead-a target falls in its block; on validation it is the twin's forecast. Sigma_a = second moment of
    the out-of-fold departures (bias included), shrunk toward its diagonal."""
    a, t_tr, nf = int(fd.lead), int(fd.t_tr), int(fd.folds)
    twin = os.path.join(fd.hf_dir, f"twin_seed{int(fd.obs_seed)}")
    channels = np.load(os.path.join(twin, "norm.npz"))["channels"]
    S = S_all[:, channels]
    T, K = S.shape
    mu, sd = S[:t_tr].mean(0), S[:t_tr].std(0) + 1e-8
    z = (S - mu) / sd
    St = torch.tensor(S, dtype=torch.float32, device=device)
    zhat = np.full((T, K), np.nan)
    blk = t_tr // nf
    for k in range(nf):
        f = forecast_lead(os.path.join(fd.hf_dir, f"fold{k}_seed{int(fd.obs_seed)}"), St[:t_tr], a, device)
        lo, hi = max(k * blk - a, 0), (k + 1) * blk - a                # launches whose target t+a is in the block
        zhat[lo:hi] = (f[lo:hi] - mu) / sd
    f = forecast_lead(twin, St, a, device)
    zhat[t_tr - a:T - a] = (f[t_tr - a:T - a] - mu) / sd               # validation launches: the twin
    oof = np.arange(int(fd.min_launch), t_tr - a)
    e = z[oof + a] - zhat[oof]
    C = e.T @ e / len(e)
    lam = float(fd.shrink)
    Sig = (1 - lam) * C + lam * np.diag(np.diag(C))
    print(f"[fd] prior lead {a}: {len(oof)} out-of-fold departures, RMS {np.sqrt(np.diag(C)).mean():.3f} "
          f"(z units), bias {np.abs(e.mean(0)).mean():.3f}; validation RMS "
          f"{np.sqrt(((z[t_tr:] - zhat[t_tr - a:T - a]) ** 2).mean(0)).mean():.3f}")
    return dict(channels=channels, mu=mu, sd=sd, z=z, zhat=zhat, Sigma=Sig, chol=np.linalg.cholesky(Sig),
                lead=a, t_tr=t_tr)


@hydra.main(version_base=None, config_path="../../config", config_name="flow_decoder")
def main(cfg: DictConfig) -> None:
    fd = cfg.fd
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    twin = os.path.join(fd.hf_dir, f"twin_seed{int(fd.obs_seed)}")
    enc = torch.load(os.path.join(twin, "model.pt"), map_location="cpu", weights_only=False)["encoder_run"]
    S_all, Y, shape, obs_noise, t_enc = load_eye(enc, device)
    t_tr, t_fit, a = int(fd.t_tr), int(fd.t_fit), int(fd.lead)
    assert t_tr == t_enc, (t_tr, t_enc)
    print(f"[fd] observer={enc} field={tuple(shape)} fit targets [{int(fd.min_launch) + a},{t_fit}) "
          f"held-out [{t_fit},{t_tr}) validation [{t_tr},{len(Y)})")

    # --- module 2's Gaussian forecast (frozen) -----------------------------------------------------------------------
    pr = latent_prior(fd, S_all.cpu().numpy(), device)
    np.savez("prior.npz", **pr)
    K = len(pr["channels"])

    # --- baseline: the linear ridge decoder on the same live channels + its iid per-cell residual ---------------------
    S_live = S_all[:, torch.as_tensor(pr["channels"], device=device)]
    lin = FrontalDecoder(K, [0], shape).to(device).fit(S_live, Y, t_tr)
    with torch.no_grad():
        res = Y[:t_tr].float() - lin(S_live[:t_tr].float()).reshape(t_tr, -1)
    torch.save(dict(state=lin.state_dict(), K=K, lags=[0], field_shape=list(shape), ridge=lin.ridge,
                    encoder_run=enc, t_tr=t_tr, res_var=res.var(0).cpu()), "decoder.pt")
    print(f"[fd] linear decoder residual RMS {float(res.pow(2).mean().sqrt()):.4f} (field units; obs noise {obs_noise})")

    # --- flow pairs (launch field, next field, latent condition) -----------------------------------------------------
    X = Y.float().reshape(len(Y), *shape)                             # field units [T,V,H,W]
    z = torch.tensor(pr["z"], dtype=torch.float32, device=device)
    zhat = torch.tensor(pr["zhat"], dtype=torch.float32, device=device)
    L = torch.tensor(pr["chol"], dtype=torch.float32, device=device)
    first = max(int(fd.min_launch), int(fd.first_launch))
    fit_t = torch.arange(first, t_fit - a, device=device)                     # launches; targets < t_fit
    val_t = torch.arange(t_fit - a, t_tr - a, device=device)                  # held-out targets [t_fit, t_tr)

    if str(fd.prior) in ("net", "dramatic"):
        # D-053: the launch-conditioned network's p(z) instead of the fixed Gaussian (its own centre and width per launch)
        fam = np.load(os.path.join(fd.hf_dir, f"twin_seed{int(fd.obs_seed)}", "norm.npz"))["fam"]
        sig_np = np.sqrt(np.diag(pr["Sigma"]))
        npr = net_prior(str(fd.prior_net), pr["z"], pr["zhat"], sig_np, fam, a, device)
        np.savez("netprior.npz", **npr, sigma=sig_np)
        assert first >= npr["t0"], (first, npr["t0"])
        n_m, n_w = (torch.tensor(npr[k], device=device) for k in ("m", "w"))
        n_L = torch.tensor(npr["chol_R"], dtype=torch.float32, device=device)
        n_s = torch.tensor(sig_np, dtype=torch.float32, device=device)
        print(f"[fd] prior: network {fd.prior_net} (t0 {npr['t0']}), mean width {float(n_w[fit_t].mean()):.3f}")
    if str(fd.prior) == "dramatic":
        # D-055: the exaggerated family prior of D-054 (on the network's centre / width), tables for every fit and
        # held-out launch, built once
        dram = DramaticSampler(os.getcwd(), str(fd.hf_dir), int(fd.obs_seed), np.arange(first, t_tr - a), device,
                               fn_dir=str(fd.prior_net))
        print(f"[fd] prior: EXAGGERATED family prior on {fd.prior_net}")

    def cond(t: torch.Tensor, eta: torch.Tensor | None = None) -> torch.Tensor:
        if str(fd.cond).lower() == "true":                         # Hydra reads `true` as a boolean
            return z[t + a]
        eta = torch.randn(len(t), K, device=device) if eta is None else eta
        if str(fd.prior) == "dramatic":
            return dram.sample(t, 1, eta[:, None])[:, 0]
        if str(fd.prior) == "net":
            return zhat[t] + n_s * (n_m[t] + n_w[t] * (eta @ n_L.T))
        return zhat[t] + eta @ L.T                                    # a draw of N(zhat_oof, Sigma)

    model = FlowDecoder(K, V=shape[0], ch=tuple(fd.ch), emb=int(fd.emb), sigma0=float(fd.sigma0),
                        source=str(fd.source), zmap=int(fd.zmap)).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(fd.lr), weight_decay=float(fd.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(fd.steps))
    print(f"[fd] flow decoder {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params, cond={fd.cond}, "
          f"source={fd.source}, zmap={fd.zmap}, sigma0={fd.sigma0}, {len(fit_t)} fit pairs, {len(val_t)} held-out")

    # fixed noise for the held-out score, so checkpoints are compared on the same draws
    g = torch.Generator(device=device).manual_seed(1234)
    v_eps = torch.randn(len(val_t), *shape, generator=g, device=device)
    v_s = torch.rand(len(val_t), generator=g, device=device)
    v_eta = torch.randn(len(val_t), K, generator=g, device=device)

    @torch.no_grad()
    def held_out(m) -> float:
        tot = 0.0
        for i in range(0, len(val_t), 64):
            t = val_t[i:i + 64]
            tot += float(m.loss(X[t], X[t + a], cond(t, v_eta[i:i + 64]), v_eps[i:i + 64], v_s[i:i + 64])) * len(t)
        return tot / len(val_t)

    tb = SummaryWriter("tb")
    best, hist = float("inf"), []
    for step in range(1, int(fd.steps) + 1):
        model.train()
        t = fit_t[torch.randint(len(fit_t), (int(fd.batch),), device=device)]
        loss = model.loss(X[t], X[t + a], cond(t))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(fd.grad_clip))
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - float(fd.ema))
        if step % 100 == 0:
            tb.add_scalar("loss/train", loss.item(), step)
        if step % int(fd.eval_every) == 0 or step == int(fd.steps):
            h = held_out(ema)
            tb.add_scalar("loss/held_out_ema", h, step)
            hist.append((step, loss.item(), h))
            mark = ""
            if h < best:
                best, mark = h, " *"
                torch.save(dict(state=ema.state_dict(), K=K, V=shape[0], ch=list(fd.ch), emb=int(fd.emb),
                                sigma0=float(fd.sigma0), source=str(fd.source), zmap=int(fd.zmap), cfg=OmegaConf.to_container(fd, resolve=True),
                                encoder_run=enc, step=step), "model.pt")
            print(f"[fd] step {step:6d}  train {loss.item():.5f}  held-out(EMA) {h:.5f}{mark}", flush=True)
    json.dump(dict(best_held_out=best, history=hist), open("metrics.json", "w"), indent=2)
    print(f"[fd] done. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
