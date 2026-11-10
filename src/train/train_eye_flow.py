"""Train the per-eye flow forecaster (D-044, SOP 04 "Per-eye flow forecaster") on the TRAINING set only.

Input : a frozen observer run (artifacts.npz, S[T,K_all]); the live eyes and their z units are the transformer
        forecaster's (same `history_channels`, same fitting-range μ/σ). Only windows whose 64 targets are < t_fit train.
Output: model.pt (best held-out checkpoint) + norm.npz + TensorBoard tb/, in the Hydra run dir .tmps/runs_ef/<stamp>/.

Run:  python -m src.train.train_eye_flow ef.encoder_run=<feature-observer>/.tmps/runs/obs_t8000_l2rw_seed0
"""
from __future__ import annotations

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

from src.models.eye_flow import EyeFlow                   # noqa: E402
from src.train.train_history_forecaster import history_channels   # noqa: E402


def windows(z: torch.Tensor, eyes: torch.Tensor, t: torch.Tensor, L: int, A: int):
    """(history [B,L], increments [B,A]) of eye `eyes[b]` launched at time `t[b]` (z [T,K], model units).

    history = z[t−L+1 .. t] of that eye; increments = z[t+1 .. t+A] − z[t]. Nothing else of the record is read.
    """
    off_h = torch.arange(-L + 1, 1, device=z.device)
    off_f = torch.arange(1, A + 1, device=z.device)
    hist = z[t[:, None] + off_h, eyes[:, None]]
    fut = z[t[:, None] + off_f, eyes[:, None]] - hist[:, -1:]
    return hist, fut


@hydra.main(version_base=None, config_path="../../config", config_name="eye_flow")
def main(cfg: DictConfig) -> None:
    ef = cfg.ef
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    run = ef.encoder_run if os.path.isabs(ef.encoder_run) else os.path.join(hydra.utils.get_original_cwd(),
                                                                          ef.encoder_run)
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    t_tr, t_fit, L, A = int(ef.t_tr), int(ef.t_fit), int(ef.hist_len), int(ef.leads)
    # the same eyes and families as the transformer forecaster; the families are kept for the READOUT only --
    # the model never sees them
    channels, fam = history_channels(art, t_fit, str(ef.channels), "auto")
    S = art["S"][:, channels].astype(np.float64)
    T, K = S.shape
    assert t_fit < t_tr < T, (t_fit, t_tr, T)
    mu, sd = S[:t_fit].mean(0), S[:t_fit].std(0) + 1e-8                # fitting range only
    z = torch.tensor((S - mu) / sd, dtype=torch.float32, device=device)
    np.savez("norm.npz", mu=mu, sd=sd, sd_std=sd, t_fit=t_fit, t_tr=t_tr, channels=channels, fam=np.array(fam))
    print(f"[ef] observer={run} S={S.shape} fit=[0,{t_fit}) held-out=[{t_fit},{t_tr}) validation=[{t_tr},{T}); "
          f"{K} eyes: " + " / ".join(f"{f} {fam.count(f)}" for f in ("stationary", "cyclic", "chaotic", "noise")))

    # launches: training = every t with a full history and all A targets < t_fit;
    # held-out = targets inside [t_fit, t_tr) (launch t_fit−1 .. t_tr−A−1)
    t_train = torch.arange(L - 1, t_fit - A, device=device)
    t_held = torch.arange(t_fit - 1, t_tr - A, device=device)
    # held-out batch with FIXED noise and τ: the same question at every check, so the curve is comparable
    hk, ht = torch.meshgrid(torch.arange(K, device=device), t_held, indexing="ij")
    h_hist, h_fut = windows(z, hk.flatten(), ht.flatten(), L, A)
    g = torch.Generator(device=device).manual_seed(123)
    h_eps = torch.randn(h_fut.shape, generator=g, device=device)
    h_tau = torch.rand(h_fut.shape[0], generator=g, device=device)
    print(f"[ef] training windows {K} x {len(t_train)} = {K * len(t_train):,}; held-out windows {h_fut.shape[0]:,}")

    model = EyeFlow(L, A, int(ef.hist_patch), int(ef.lead_patch), int(ef.d), int(ef.enc_layers),
                    int(ef.dec_layers), int(ef.heads), float(ef.dropout),
                    str(ef.get("target_scale", "none"))).to(device)
    if model.target_scale != "none":
        # g_a = RMS over ALL training windows (every eye, every launch) of the target in units of the window's own
        # scale (s_h for history_lead, 1 for lead): after division the target is ~1 at every lead, on average
        acc = torch.zeros(A, device=device)
        for k in range(K):
            hist, fut = windows(z, torch.full_like(t_train, k), t_train, L, A)
            unit = model.history_step(hist) if model.target_scale == "history_lead" else 1.0
            acc += (fut / unit).pow(2).sum(0)
        model.lead_scale.copy_((acc / (K * len(t_train))).sqrt())
        g_a = model.lead_scale.cpu().numpy()
        print(f"[ef] target_scale={model.target_scale}: g_a h1/h4/h16/h64 = "
              f"{g_a[0]:.3f}/{g_a[3]:.3f}/{g_a[15]:.3f}/{g_a[-1]:.3f}")
    opt =torch.optim.AdamW(model.parameters(), lr=float(ef.lr), weight_decay=float(ef.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(ef.steps))
    writer = SummaryWriter(log_dir="tb")
    print(f"[ef] params={sum(p.numel() for p in model.parameters()):,}")

    best, best_step, curve = float("inf"), -1, []
    for step in range(int(ef.steps)):
        model.train()
        B = int(ef.batch)
        eyes = torch.randint(0, K, (B,), device=device)
        t = t_train[torch.randint(0, len(t_train), (B,), device=device)]
        loss = model.loss(*windows(z, eyes, t, L, A))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(ef.grad_clip))
        opt.step()
        sched.step()

        if step % int(ef.eval_every) == 0 or step == int(ef.steps) - 1:
            model.eval()
            with torch.no_grad():
                held = float(np.mean([float(model.loss(h_hist[i:i + 4096], h_fut[i:i + 4096], h_eps[i:i + 4096],
                                                       h_tau[i:i + 4096])) for i in range(0, len(h_fut), 4096)]))
            loss = loss.detach()
            writer.add_scalar("train/loss", float(loss), step)
            writer.add_scalar("heldout/flow_loss", held, step)
            writer.add_scalar("train/grad_norm", float(gn), step)
            curve.append((step, float(loss), held))
            tag = ""
            if held < best:                                             # keep the best held-out checkpoint
                best, best_step, tag = held, step, " *"
                torch.save(dict(state=model.state_dict(), cfg=OmegaConf.to_container(ef, resolve=True), K=K,
                                encoder_run=run, step=step), "model.pt")
            print(f"  step {step:6d} | loss={float(loss):.4f} held-out={held:.4f} |g|={float(gn):.2f}{tag}", flush=True)

    json.dump(dict(best_heldout=best, best_step=best_step, curve=curve), open("metrics.json", "w"), indent=2)
    writer.close()
    print(f"[ef] done. best held-out flow loss {best:.4f} at step {best_step}. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
