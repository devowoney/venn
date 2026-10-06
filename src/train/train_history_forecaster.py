"""Train the history forecaster (module 2, "prefrontal cortex" = latent emulator, SOP 04) on the TRAINING set only.

Input : a frozen eye-hippocampus run (artifacts.npz, S[T,K]); only S[:t_fit] ever reaches a gradient or a statistic.
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
from src.probes.family import amp_ratio, label_family, series_stats   # noqa: E402


def lead_mask(t0: int, T: int, t_fit: int, leads: int, min_sight: int, device) -> torch.Tensor:
    """[T,A] bool: which (position, lead) pairs enter the loss for a sequence starting at absolute time t0.

    Position j (absolute time t0+j) counts only once it has >= min_sight steps of history, and lead a only if
    its target t0+j+a is still inside the fitting range (< t_fit): no target from outside the training data.
    """
    j = torch.arange(T, device=device)[:, None]
    a = torch.arange(1, leads + 1, device=device)[None, :]
    return (j >= min_sight) & (t0 + j + a < t_fit)


def fair_crps(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair (unbiased for finite M) ensemble CRPS, elementwise. x [...,M] members, y [...] truth -> [...].

    CRPS = E|X - y| - E|X - X'|/2. The spread term is computed by SORTING the members:
    sum_{i<j} |x_j - x_i| = sum_k x_(k) (2k - M - 1), which is O(M log M) instead of O(M^2) and differentiable.
    """
    M = x.shape[-1]
    skill = (x - y[..., None]).abs().mean(-1)
    xs, _ = torch.sort(x, dim=-1)
    k = torch.arange(1, M + 1, device=x.device, dtype=x.dtype)
    spread = (xs * (2 * k - M - 1)).sum(-1) / (M * (M - 1))           # = sum_{i!=j}|xi-xj| / (2M(M-1))
    return skill - spread


def forecast_loss(model: HistoryForecaster, z: torch.Tensor, t0: int, t_fit: int, min_sight: int,
                  members: int = 0, lambda_crps: float = 1.0, input_noise: float = 0.0):
    """Loss of one history z [1,T,K] over all valid (position, lead).

    Deterministic model (members = 0): mean squared error on every channel.
    Ensemble model: MSE of the base forecast on the NON-chaotic channels + lambda_crps x fair CRPS of the members on
    the CHAOTIC channels (SOP 04) -- the chaotic channels are asked for an honest spread, not for the mean.
    """
    T, A = z.shape[1], model.A
    # history corruption (anti-memorization): the model READS a noisy history but is scored against the clean
    # future, so it cannot recognize and replay the one training trajectory it has seen (F-21 point 5)
    z_in = z + input_noise * torch.randn_like(z) if input_noise > 0 else z
    # target of (position j, lead a) = z[j+a]; build it by shifting, padding past the end (masked out anyway)
    pad = torch.cat([z, z[:, -1:].expand(1, A, -1)], dim=1)            # [1,T+A,K]
    tgt = torch.stack([pad[:, a:a + T] for a in range(1, A + 1)], dim=2)   # [1,T,A,K]
    m = lead_mask(t0, T, t_fit, A, min_sight, z.device)[None, :, :, None].float()   # [1,T,A,1]
    if members == 0:
        err = ((model(z_in) - tgt) ** 2) * m
        return err.sum() / (m.sum() * z.shape[-1]).clamp_min(1.0)
    ens = model(z_in, members=members)                                 # [1,T,M,A,K]
    c = model.chaos                                                    # [K]
    # non-chaotic channels: MSE of EVERY member (= MSE of the mean + member variance). In mode "multi_scenario" the members
    # are identical there, so this is the old base-forecast MSE; in mode "memory" it lets a reliable channel spread
    # only where that lowers its error (SOP 04, user choice "CRPS chaotic + MSE others")
    sq = ((ens - tgt[:, :, None]) ** 2).mean(2)                        # [1,T,A,K] averaged over members
    mse = (sq * m * (1 - c)).sum() / (m.sum() * (1 - c).sum()).clamp_min(1.0)
    crps = fair_crps(ens.permute(0, 1, 3, 4, 2), tgt)                  # [1,T,A,K]
    crps = (crps * m * c).sum() / (m.sum() * c.sum()).clamp_min(1.0)
    return mse + lambda_crps * crps


@torch.no_grad()
def calibrate_spread(model: HistoryForecaster, z_all: torch.Tensor, t_fit: int, t_tr: int, members: int = 32,
                     grid=np.linspace(1.0, 6.0, 26)) -> np.ndarray:
    """Per-lead spread inflation s_a on the held-out end of the training set (SOP 04 "spread calibration").

    Launches whose targets lie in [t_fit, t_tr): never used for fitting. For each lead, s_a = the grid value with the
    lowest fair CRPS over the chaotic channels, widening the members around their own mean.
    """
    model.eval()
    A, c = model.A, model.chaos.bool()
    model.spread.fill_(1.0)
    torch.manual_seed(0)
    ens = model(z_all[None, :t_tr], members=members)[0]               # [t_tr,M,A,K]
    L = torch.arange(t_fit - 1, t_tr - A, device=z_all.device)
    e = ens[L][..., c]                                                 # [N,M,A,Kc]
    idx = L[:, None] + torch.arange(1, A + 1, device=z_all.device)[None, :]
    y = z_all[idx][..., c]                                             # [N,A,Kc]
    mu = e.mean(1, keepdim=True)
    best = np.ones(A)
    for a in range(A):
        scores = [float(fair_crps((mu + float(g) * (e - mu))[:, :, a].permute(0, 2, 1), y[:, a]).mean()) for g in grid]
        best[a] = grid[int(np.argmin(scores))]
    return best


@torch.no_grad()
def calibrate_score(model: HistoryForecaster, z_all: torch.Tensor, t_fit: int, t_tr: int, members: int = 32,
                    clip=(0.1, 50.0)):
    """Spread from a predefined CERTAINTY SCORE (SOP 04 "multi-scenario for the whole family").

    score[a, k] = RMSE of the scenario mean at lead a, channel k, on launches whose targets lie in the held-out
    [t_fit, t_tr) -- how far off the prediction is expected to be. Each channel's scenarios are then widened (or
    narrowed) per lead so that their spread equals that score: certain channel/lead -> small fan, uncertain -> wide.
    Returns (s [A,K] factors, score [A,K]).
    """
    model.eval()
    A = model.A
    model.spread.fill_(1.0)
    torch.manual_seed(0)
    ens = model(z_all[None, :t_tr], members=members)[0]               # [t_tr,M,A,K]
    L = torch.arange(t_fit - 1, t_tr - A, device=z_all.device)
    e = ens[L]                                                         # [N,M,A,K]
    idx = L[:, None] + torch.arange(1, A + 1, device=z_all.device)[None, :]
    y = z_all[idx]                                                     # [N,A,K]
    score = ((e.mean(1) - y) ** 2).mean(0).sqrt()                      # [A,K] expected error of the best estimate
    raw = ((members + 1) / members * e.var(1).mean(0)).sqrt()          # [A,K] raw scenario spread
    s = (score / raw.clamp_min(1e-6)).clamp(*clip)
    return s.cpu().numpy(), score.cpu().numpy()


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
    mu, sd_std = S[:t_fit].mean(0), S[:t_fit].std(0) + 1e-8
    # family labels decided on the FITTING range only (chaos mask for the ensemble head, stationary rescaling below)
    fam = [label_family(series_stats((S[:t_fit, i] - mu[i]) / sd_std[i]), amp_ratio=amp_ratio(S[:t_fit, i]))
           for i in range(K)]
    stationary = np.array([f == "stationary" for f in fam])
    sd = sd_std.copy()
    if str(hf.get("stationary_scale", "std")) == "level":
        # a stationary history is "level + leak": scale it by its signal size, not by the tiny std of its leak,
        # so the leak stays a ~4 % wobble around 0 (= the level) instead of becoming a unit-variance target
        sd[stationary] = np.sqrt(mu[stationary] ** 2 + sd_std[stationary] ** 2)
        print(f"[hf] stationary_scale=level: channels {np.flatnonzero(stationary).tolist()} scaled by signal size "
              f"(sd/size = {np.round(sd_std[stationary] / sd[stationary], 3).tolist()})")
    z_all = torch.tensor((S - mu) / sd, dtype=torch.float32, device=device)
    z_fit = z_all[:t_fit]                                              # the only data that trains
    print(f"[hf] encoder={run} S={S.shape} fit=[0,{t_fit}) train=[0,{t_tr}) validation=[{t_tr},{T_all})")

    ens_cfg = hf.get("ensemble", None)
    use_ens = bool(ens_cfg is not None and ens_cfg.enable)
    chaos = np.array([f == "chaotic" for f in fam], dtype=np.float32)
    members = int(ens_cfg.members) if use_ens else 0
    lam = float(ens_cfg.lambda_crps) if use_ens else 0.0
    chaos_label = chaos.copy()                                         # the family labels themselves, for the record
    if use_ens and str(ens_cfg.get("families", "chaotic")) == "all":
        # multi-scenario for the WHOLE family: every channel gets scenarios (and CRPS); the certainty score decides
        # afterwards how wide each one is. `chaos` is the scenario MASK the model uses -- here all ones.
        chaos = np.ones(K, dtype=np.float32)
    np.savez("norm.npz", mu=mu, sd=sd, sd_std=sd_std, stationary=stationary, t_fit=t_fit, t_tr=t_tr, chaos=chaos,
             chaos_label=chaos_label)
    if use_ens:
        print(f"[hf] ensemble ON: scenarios on {int(chaos.sum())} channels {np.flatnonzero(chaos).tolist()} "
              f"(chaotic-labelled: {np.flatnonzero(chaos_label).tolist()}); M={members} members, "
              f"noise_dim={int(ens_cfg.noise_dim)}")
    model = HistoryForecaster(K, leads=int(hf.leads), d=int(hf.d), layers=int(hf.layers),
                              heads=int(hf.heads), dropout=float(hf.dropout),
                              chaos_mask=chaos if use_ens else None,
                              noise_dim=int(ens_cfg.noise_dim) if use_ens else 16,
                              mode=str(ens_cfg.get("mode", "multi_scenario")) if use_ens else "multi_scenario").to(device)
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
            loss = loss + forecast_loss(model, z_fit[None, s0:], s0, t_fit, int(hf.min_sight), members, lam,
                                        float(hf.get("input_noise", 0.0)))
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

    if use_ens and bool(ens_cfg.get("calibrate", False)):
        assert t_fit < t_tr, "spread calibration needs a held-out slice: set hf.t_fit < hf.t_tr"
        if str(ens_cfg.get("spread_from", "crps")) == "score":
            # predefined certainty score per channel x lead -> each channel's own fan width (SOP 04)
            s_ak, score = calibrate_score(model, z_all, t_fit, t_tr)
            model.spread.copy_(torch.as_tensor(s_ak, dtype=torch.float32))
            np.save("spread.npy", s_ak)
            np.save("score.npy", score)
            print(f"[hf] certainty score on [{t_fit},{t_tr}) (RMSE per channel x lead), mean over channels: "
                  f"h1={score[0].mean():.3f} h8={score[7].mean():.3f} h16={score[15].mean():.3f} "
                  f"h64={score[-1].mean():.3f} | spread factor median h1={np.median(s_ak[0]):.2f} "
                  f"h64={np.median(s_ak[-1]):.2f}")
        else:
            s_a = calibrate_spread(model, z_all, t_fit, t_tr)
            model.spread.copy_(torch.as_tensor(s_a[:, None], dtype=torch.float32).expand(-1, K))
            np.save("spread.npy", s_a)
            print(f"[hf] spread calibrated on [{t_fit},{t_tr}): s(h1)={s_a[0]:.2f} s(h8)={s_a[7]:.2f} "
                  f"s(h16)={s_a[15]:.2f} s(h64)={s_a[-1]:.2f}")
    torch.save(dict(state=model.state_dict(), cfg=OmegaConf.to_container(hf, resolve=True), K=K,
                    encoder_run=run), "model.pt")
    json.dump(dict(final_loss=float(loss.detach()), t_fit=t_fit, t_tr=t_tr), open("metrics.json", "w"), indent=2)
    writer.close()
    print(f"[hf] done. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
