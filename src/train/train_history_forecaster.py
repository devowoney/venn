"""Train the history forecaster (module 2, "prefrontal cortex", SOP 04) on the TRAINING set only.

Input : a frozen eye-lobe run (artifacts.npz, S[T,K]); only S[:t_fit] ever reaches a gradient or a statistic.
Output: model.pt + norm.npz (train-only mu/sd) + TensorBoard tb/, in the Hydra run dir .tmps/runs_hf/<stamp>/.

Run:  python -m src.train.train_history_forecaster hf.encoder_run=.tmps/runs/hf_enc_seed0
      python -m src.train.train_history_forecaster hf.encoder_run=... hf.t_fit=1750      # development run
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

from src.models.history_forecaster import HistoryForecaster   # noqa: E402


def lead_mask(t0: int, T: int, t_fit: int, leads: int, min_sight: int, device) -> torch.Tensor:
    """[T,A] bool: which (position, lead) pairs enter the loss for a sequence starting at absolute time t0.

    Position j (absolute time t0+j) counts only once it has >= min_sight steps of history, and lead a only if
    its target t0+j+a is still inside the fitting range (< t_fit): no target from outside the training data.
    """
    j = torch.arange(T, device=device)[:, None]
    a = torch.arange(1, leads + 1, device=device)[None, :]
    return (j >= min_sight) & (t0 + j + a < t_fit)


def forecast_loss(model: HistoryForecaster, z: torch.Tensor, t0: int, t_fit: int, min_sight: int):
    """Mean squared forecast error over all valid (position, lead, channel) of one history z [1,T,K]."""
    T, A = z.shape[1], model.A
    zhat = model(z)                                                    # [1,T,A,K]
    # target of (position j, lead a) = z[j+a]; build it by shifting, padding past the end (masked out anyway)
    pad = torch.cat([z, z[:, -1:].expand(1, A, -1)], dim=1)            # [1,T+A,K]
    tgt = torch.stack([pad[:, a:a + T] for a in range(1, A + 1)], dim=2)   # [1,T,A,K]
    m = lead_mask(t0, T, t_fit, A, min_sight, z.device)[None, :, :, None].float()
    err = ((zhat - tgt) ** 2) * m
    return err.sum() / (m.sum() * z.shape[-1]).clamp_min(1.0)


@hydra.main(version_base=None, config_path="../../config", config_name="history_forecaster")
def main(cfg: DictConfig) -> None:
    hf = cfg.hf
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # --- data: the TRAINING set only -----------------------------------------------------------
    run = hf.encoder_run if os.path.isabs(hf.encoder_run) else os.path.join(hydra.utils.get_original_cwd(),
                                                                          hf.encoder_run)
    S = np.load(os.path.join(run, "artifacts.npz"))["S"].astype(np.float64)   # [T,K]
    T_all, K = S.shape
    t_tr = int(hf.t_tr)
    t_fit = int(hf.t_fit) if hf.t_fit else t_tr
    assert t_fit <= t_tr < T_all, (t_fit, t_tr, T_all)
    # channel std spans ~10 .. ~1000: standardize with statistics of the fitting range only
    mu, sd = S[:t_fit].mean(0), S[:t_fit].std(0) + 1e-8
    z_all = torch.tensor((S - mu) / sd, dtype=torch.float32, device=device)
    z_fit = z_all[:t_fit]                                              # the only data that trains
    np.savez("norm.npz", mu=mu, sd=sd, t_fit=t_fit, t_tr=t_tr)
    print(f"[hf] encoder={run} S={S.shape} fit=[0,{t_fit}) train=[0,{t_tr}) validation=[{t_tr},{T_all})")

    model = HistoryForecaster(K, leads=int(hf.leads), d=int(hf.d), layers=int(hf.layers),
                              heads=int(hf.heads), dropout=float(hf.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hf.lr), weight_decay=float(hf.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(hf.steps))
    writer = SummaryWriter(log_dir="tb")
    g = torch.Generator().manual_seed(cfg.seed + 1000)
    print(f"[hf] params={sum(p.numel() for p in model.parameters()):,}")

    for step in range(int(hf.steps)):
        model.train()
        loss = 0.0
        # B long histories z[s0 : t_fit] with random start: varied lengths, all ending at the fit boundary
        for s0 in torch.randint(0, int(hf.max_start), (int(hf.batch),), generator=g).tolist():
            loss = loss + forecast_loss(model, z_fit[None, s0:], s0, t_fit, int(hf.min_sight))
        loss = loss / int(hf.batch)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(hf.grad_clip))
        opt.step()
        sched.step()

        if step % int(hf.eval_every) == 0 or step == int(hf.steps) - 1:
            model.eval()
            with torch.no_grad():
                # development score: forecasts made in [t_fit - 1, t_tr), targets inside [t_fit, t_tr) only.
                # Final runs (t_fit = t_tr) have no development slice -> report the training loss only.
                msg = ""
                if t_fit < t_tr:
                    zhat = model(z_all[None, :t_tr])[0]                # [t_tr,A,K]
                    a1 = ((zhat[t_fit - 1:t_tr - 1, 0] - z_all[t_fit:t_tr]) ** 2).mean()
                    p1 = ((z_all[t_fit - 1:t_tr - 1] - z_all[t_fit:t_tr]) ** 2).mean()
                    skill1 = float(1 - a1 / p1)
                    writer.add_scalar("dev/skill_lead1", skill1, step)
                    msg = f" dev skill1={skill1:+.3f}"
            writer.add_scalar("train/loss", float(loss.detach()), step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            print(f"  step {step:5d} | loss={float(loss.detach()):.4f} |g|={float(gn):.2f}{msg}")

    torch.save(dict(state=model.state_dict(), cfg=OmegaConf.to_container(hf, resolve=True), K=K,
                    encoder_run=run), "model.pt")
    json.dump(dict(final_loss=float(loss.detach()), t_fit=t_fit, t_tr=t_tr), open("metrics.json", "w"), indent=2)
    writer.close()
    print(f"[hf] done. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
