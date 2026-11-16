"""Train the kernel-aware Gaussian decoder (D-056) on TRUE pairs (z(t), x(t)) of the training set.

Input : module 2's twin_seed{s} (which live channels; the observer run it read), the observer's frozen masks and its
        field (regenerated from its config + seed). z = live channels of S standardized on [0, t_tr) (canonical units,
        the same as every latent prior of module 3).
Output (Hydra run dir .tmps/runs_kd/<stamp>/): model.pt (EMA weights of the best held-out checkpoint + everything needed
        to rebuild it), metrics.json, tb/ (TensorBoard).
Run:  python -m src.train.train_kernel_decoder kd.obs_seed=0
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

from src.models.kernel_decoder import KernelDecoder       # noqa: E402
from src.train.fit_frontal_decoder import load_eye        # noqa: E402


def kernel_data(hf_dir: str, obs_seed: int, t_tr: int, device: str) -> dict:
    """Live channels, their masks, canonical z [T, k] and the field [T, V, H, W] of one observer seed."""
    twin = os.path.join(hf_dir, f"twin_seed{obs_seed}")
    enc = torch.load(os.path.join(twin, "model.pt"), map_location="cpu", weights_only=False)["encoder_run"]
    channels = np.load(os.path.join(twin, "norm.npz"))["channels"]
    S_all, Y, shape, obs_noise, _ = load_eye(enc, device)
    art = np.load(os.path.join(enc, "artifacts.npz"), allow_pickle=True)
    masks = torch.tensor(art["masks"][channels], dtype=torch.float32, device=device)
    S = S_all[:, torch.as_tensor(channels, device=device)].float()
    mu, sd = S[:t_tr].mean(0), S[:t_tr].std(0) + 1e-8
    X = Y.float().reshape(len(Y), *shape)
    return dict(enc=enc, channels=channels, masks=masks, mu=mu, sd=sd, z=(S - mu) / sd, X=X, obs_noise=obs_noise)


@hydra.main(version_base=None, config_path="../../config", config_name="kernel_decoder")
def main(cfg: DictConfig) -> None:
    kd = cfg.kd
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    d = kernel_data(str(kd.hf_dir), int(kd.obs_seed), int(kd.t_tr), device)
    z, X = d["z"], d["X"]
    model = KernelDecoder(d["masks"], d["mu"], d["sd"], ch=tuple(kd.ch), emb=int(kd.emb),
                          logvar_range=(float(kd.logvar_min), float(kd.logvar_max))).to(device)
    with torch.no_grad():                                          # the consistency the architecture guarantees
        mean, _ = model(z[:64])
        S = z[:64] * d["sd"] + d["mu"]
        rel = float((mean.flatten(1) @ model.K.T - S).abs().max() / S.abs().max())
    print(f"[kd] observer {d['enc']}: k={len(d['channels'])} live channels, field {tuple(X.shape[1:])}; "
          f"|K mu - S| / |S| = {rel:.1e}; {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(kd.lr), weight_decay=float(kd.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(kd.steps))
    fit_t = torch.arange(0, int(kd.t_fit), device=device)
    sel_t = torch.arange(int(kd.t_fit), int(kd.t_tr), device=device)

    @torch.no_grad()
    def held_out(m) -> tuple:
        nll, se = 0.0, 0.0
        for i in range(0, len(sel_t), 64):
            t = sel_t[i:i + 64]
            mean, lv = m(z[t])
            nll += float(0.5 * ((X[t] - mean) ** 2 * torch.exp(-lv) + lv).mean()) * len(t)
            se += float(((X[t] - mean) ** 2).mean()) * len(t)
        return nll / len(sel_t), (se / len(sel_t)) ** 0.5

    tb = SummaryWriter("tb")
    best, hist = float("inf"), []
    for step in range(1, int(kd.steps) + 1):
        model.train()
        t = fit_t[torch.randint(len(fit_t), (int(kd.batch),), device=device)]
        loss = model.nll(z[t], X[t])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(kd.grad_clip))
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - float(kd.ema))
        if step % 100 == 0:
            tb.add_scalar("nll/train", loss.item(), step)
        if step % int(kd.eval_every) == 0 or step == int(kd.steps):
            h, r = held_out(ema)
            tb.add_scalar("nll/held_out_ema", h, step)
            tb.add_scalar("rmse/held_out_ema", r, step)
            hist.append((step, loss.item(), h, r))
            mark = ""
            if h < best:
                best, mark = h, " *"
                torch.save(dict(state=ema.state_dict(), masks=d["masks"].cpu(), mu=d["mu"].cpu(), sd=d["sd"].cpu(),
                                channels=d["channels"], encoder_run=d["enc"], ch=list(kd.ch), emb=int(kd.emb),
                                logvar_range=(float(kd.logvar_min), float(kd.logvar_max)),
                                cfg=OmegaConf.to_container(kd, resolve=True), step=step), "model.pt")
            print(f"[kd] step {step:6d}  train nll {loss.item():.4f}  held-out nll {h:.4f}  rmse {r:.4f}{mark}",
                  flush=True)
    json.dump(dict(best_held_out_nll=best, history=hist), open("metrics.json", "w"), indent=2)
    print(f"[kd] done. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
