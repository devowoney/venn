"""Streaming FIELD forecast with its spread: hippocampus (chaotic latent ensemble) -> frontal cortex -> field (SOP 05).

The full emulator chain, scored on the VALIDATION set. At every launch t the hippocampus forecasts the latent state
from the history up to t (one causal pass, as in eval_history_forecaster). Its ensemble head spreads ONLY the chaotic
channels (stationary/cyclic channels are identical in every member). The frozen frontal decoder maps EVERY member to a
field, so the field forecast is an ensemble x_hat[M] whose spread is the chaotic uncertainty, placed where the chaotic
patterns are. (A deterministic hippocampus = 1 member, no spread.)

Per lead a, on launches whose targets all lie in [t_tr, T):
  RMSE        : HEADLINE (user: "use RMSE to compare the performance by lead time"). Field RMSE of the member MEAN
                vs the observed field, in field units (each variable unit-std), pooled over cells, against
                decoded truth (decoder on the TRUE future latent = decoding ceiling), persistence x(t), climatology.
                Floor = obs_noise (0.05): the observed field's own unpredictable noise
  skill       : balanced R^2 / noise ceiling (D-032) of the same forecasts (secondary, kept in the json)
  uncertainty : field spread (std across members) / RMSE of the member mean, both vs the NOISE-FREE state (same
                generator seed, obs_noise = 0) -- the members carry the state's uncertainty, not the iid noise;
                corr(spread map, error map) over cells -- is the spread where the error is?

Run:  python -m src.probes.eval_frontal_decoder --dec .tmps/runs_dec/dec10k_seed* \
          --hf <hippocampus runs, paired with the decoder through their shared encoder run> --out .tmps/eval_dec/<name>
"""
from __future__ import annotations

import argparse
import json
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
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402

from src.data.synthetic import GenConfig, generate_field   # noqa: E402
from src.models.frontal_decoder import FrontalDecoder, lag_window   # noqa: E402
from src.probes.eval_history_forecaster import load_run as load_hf   # noqa: E402
from src.train.fit_frontal_decoder import load_eye          # noqa: E402

METHODS = ("decoded_truth", "forecast", "persistence", "climatology")
COL = dict(decoded_truth="#8c8b86", forecast="#2a78d6", persistence="#1baf7a", climatology="#eda100")
LABEL = dict(decoded_truth="decoded TRUE latent (decoding ceiling)", forecast="hippocampus -> frontal decoder (member mean)",
             persistence="persistence x(t)", climatology="climatology")
SEED_COL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#eb6834", "#a8401b"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
LEADS_SHOW = (8, 64)


def load_decoder(run: str, device: str) -> FrontalDecoder:
    ck = torch.load(os.path.join(run, "decoder.pt"), map_location=device, weights_only=False)
    dec = FrontalDecoder(ck["K"], ck["lags"], ck["field_shape"], ck["ridge"]).to(device)
    dec.load_state_dict(ck["state"])
    dec.encoder_run, dec.t_tr = ck["encoder_run"], ck["t_tr"]
    return dec.eval()


@torch.no_grad()
def latent_members(model, nm, S: torch.Tensor, members: int = 32) -> torch.Tensor:
    """S_hat [T-1, M, A, K] in PHYSICAL latent units: the M members forecast at t from S_0..S_t.

    Ensemble model: M members, spread on the chaotic channels only. Deterministic model: M = 1.
    """
    mu, sd = (torch.tensor(nm[k], dtype=torch.float32, device=S.device) for k in ("mu", "sd"))
    z = ((S.float() - mu) / sd)[None, :-1]
    if model.ensemble:
        torch.manual_seed(0)                                   # same member draws as eval_history_forecaster
        zhat = model(z, members=members)[0]
    else:
        zhat = model(z)[0][:, None]
    return zhat * sd + mu


def forecast_window(S: torch.Tensor, Shat: torch.Tensor, L: torch.Tensor, a: int, lags) -> torch.Tensor:
    """Decoder window [n, M, F] at the target time t+a for launches L, one per member. Lag steps after t take that
    member's forecast, earlier ones the observed history (the emulator never sees an observation past the launch)."""
    n, M = len(L), Shat.shape[1]
    parts = []
    for g in lags:
        off = a - int(g)                                       # position of S(t+a-g) relative to the launch t
        parts.append(Shat[L, :, off - 1] if off >= 1 else S[L + off][:, None].expand(n, M, -1))
    return torch.cat(parts, dim=-1)


def r2_pooled(y: torch.Tensor, yh: torch.Tensor) -> float:
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def noise_free_state(encoder_run: str, Y: torch.Tensor) -> torch.Tensor:
    """The eye's field WITHOUT observation noise [T,N]: same generator config + seed, obs_noise = 0. The noise is drawn
    last in the generator, so every mode is identical; checked against the noisy field."""
    ecfg = OmegaConf.load(os.path.join(encoder_run, ".hydra/config.yaml"))
    gc = OmegaConf.to_container(ecfg.data, resolve=True)
    f, _ = generate_field(GenConfig(**{**gc, "obs_noise": 0.0}), seed=ecfg.seed)
    F = torch.tensor(f.reshape(len(f), -1), dtype=torch.float64, device=Y.device)
    assert float((Y - F).std()) < 1.2 * float(gc["obs_noise"]), "noise-free regeneration does not match the field"
    return F


@torch.no_grad()
def eval_pair(dec_run: str, hf_run: str, device: str, members: int = 32) -> tuple[dict, dict]:
    dec = load_decoder(dec_run, device)
    model, nm, _, t_tr = load_hf(hf_run, device)
    S, Y, shape, obs_noise, _ = load_eye(dec.encoder_run, device)
    assert t_tr == dec.t_tr, (t_tr, dec.t_tr)
    T, A, N = len(S), model.A, Y.shape[1]
    Sf = S.float()
    Shat = latent_members(model, nm, Sf, members)              # [T-1,M,A,K]
    M = Shat.shape[1]
    Yb = (Y.float() - dec.mu_y) / dec.sd_y                     # observed field, balanced [T,N]
    Fb = (noise_free_state(dec.encoder_run, Y).float() - dec.mu_y) / dec.sd_y   # noise-free state, balanced
    ceil = float((1 - obs_noise ** 2 / Y[:t_tr].var(0)).mean())
    var_ = torch.arange(N, device=device) // (N // shape[0])   # variable index of each flattened cell (SSH 0, SST 1)
    ceil_v = [float((1 - obs_noise ** 2 / Y[:t_tr][:, var_ == v].var(0)).mean()) for v in range(shape[0])]
    win_true = lag_window(Sf, dec.lags)                        # decoder windows of the TRUE latent series

    L = torch.arange(t_tr - 1, T - A, device=device)           # launches with every target in validation
    out = {m: np.zeros(A) for m in METHODS}
    rmse = {m: np.zeros(A) for m in METHODS}
    rmse_var = {m: np.zeros((A, shape[0])) for m in METHODS}
    per_var = {m: np.zeros((A, shape[0])) for m in ("forecast", "decoded_truth")}
    unc = dict(spread=np.zeros(A), rmse=np.zeros(A), ratio=np.zeros(A), map_corr=np.zeros(A))
    t0 = int(L[len(L) // 3])                                   # one launch for the snapshot figure
    j0 = len(L) // 3
    snap = dict(t0=t0, leads=list(LEADS_SHOW), truth=[], m1=[], m2=[], mean=[], spread=[])
    for a in range(1, A + 1):
        y = Yb[L + a]                                          # [n,N] observed truth (balanced)
        fe = dec(forecast_window(Sf, Shat, L, a, dec.lags), balanced=True).reshape(len(L), M, N)  # field members
        mean = fe.mean(1)
        preds = dict(decoded_truth=dec(win_true[L + a], balanced=True).reshape(len(L), N), forecast=mean,
                     persistence=Yb[L], climatology=torch.zeros_like(y))
        for m, yh in preds.items():
            out[m][a - 1] = r2_pooled(y, yh) / ceil
            e2 = (dec.sd_y * (y - yh)) ** 2                    # squared error back in field units
            rmse[m][a - 1] = float(e2.mean().sqrt())
            for v in range(shape[0]):
                rmse_var[m][a - 1, v] = float(e2[:, var_ == v].mean().sqrt())
        for m in per_var:
            for v in range(shape[0]):
                c = var_ == v
                per_var[m][a - 1, v] = r2_pooled(y[:, c], preds[m][:, c]) / ceil_v[v]
        if M > 1:
            # finite-M corrected member variance; error of the member mean against the noise-free state
            var = fe.var(1) * (M + 1) / M
            err2 = (mean - Fb[L + a]) ** 2
            unc["spread"][a - 1] = float(var.mean().sqrt())
            unc["rmse"][a - 1] = float(err2.mean().sqrt())
            unc["ratio"][a - 1] = unc["spread"][a - 1] / unc["rmse"][a - 1]
            sp_map, er_map = var.mean(0).sqrt(), err2.mean(0).sqrt()
            unc["map_corr"][a - 1] = float(torch.corrcoef(torch.stack([sp_map, er_map]))[0, 1])
        if a in LEADS_SHOW:                                    # physical units for the picture
            phys = lambda b: (dec.mu_y + dec.sd_y * b).reshape(shape).cpu().numpy()   # noqa: E731
            snap["truth"].append(phys(Fb[t0 + a]))
            snap["m1"].append(phys(fe[j0, 0]))
            snap["m2"].append(phys(fe[j0, min(1, M - 1)]))
            snap["mean"].append(phys(mean[j0]))
            snap["spread"].append((dec.sd_y * fe[j0].std(0)).reshape(shape).cpu().numpy() if M > 1
                                  else np.zeros(shape))
        del fe
    chaos = nm["chaos"].astype(bool) if "chaos" in nm.files else np.zeros(Shat.shape[-1], bool)
    nc = float(Shat[..., ~torch.as_tensor(chaos, device=device)].std(1).max()) if M > 1 else 0.0
    return dict(dec_run=os.path.relpath(dec_run), hf_run=hf_run, members=M, chaotic_channels=np.flatnonzero(chaos).tolist(),
                max_nonchaotic_latent_spread=nc, ceiling=ceil, n_launch=len(L),
                rmse={m: v.tolist() for m, v in rmse.items()}, rmse_var={m: v.tolist() for m, v in rmse_var.items()},
                obs_noise=obs_noise, r2c={m: v.tolist() for m, v in out.items()}, r2c_var={m: v.tolist() for m, v in per_var.items()},
                uncertainty={k: v.tolist() for k, v in unc.items()}), snap


def _style(x, ylabel: str | None = None) -> None:
    x.set_xlabel("lead a (steps)")
    x.set_xscale("log", base=2)
    if ylabel:
        x.set_ylabel(ylabel)
    x.grid(color="#e6e5df", lw=0.6)
    x.set_axisbelow(True)
    for s in ("top", "right"):
        x.spines[s].set_visible(False)


def plot_curves(res: list[dict], path: str, title: str) -> None:
    A = len(res[0]["rmse"]["forecast"])
    leads = np.arange(1, A + 1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for m in METHODS:
        v = np.array([r["rmse"][m] for r in res])
        ax[0, 0].plot(leads, v.mean(0), color=COL[m], lw=2, label=LABEL[m])
        ax[0, 0].fill_between(leads, v.min(0), v.max(0), color=COL[m], alpha=0.15, lw=0)
    ax[0, 0].axhline(res[0]["obs_noise"], color="#8c8b86", lw=1, ls=":", label="obs-noise floor")
    ax[0, 0].set_ylim(0, None)
    ax[0, 0].set_title("field RMSE of the member mean vs observed field (mean, band = seed range)", fontsize=10)
    ax[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax[0, 0], "RMSE (field units, unit-std variables)")
    for v, (name, ls) in enumerate((("SSH", "-"), ("SST", "--"))):
        f = np.array([r["rmse_var"]["forecast"] for r in res])[:, :, v]
        ax[0, 1].plot(leads, f.mean(0), color=COL["forecast"], ls=ls, lw=2, label=f"{name} forecast")
        d = np.array([r["rmse_var"]["decoded_truth"] for r in res])[:, :, v]
        ax[0, 1].plot(leads, d.mean(0), color=COL["decoded_truth"], ls=ls, lw=1.5, label=f"{name} decoding ceiling")
    ax[0, 1].set_ylim(0, None)
    ax[0, 1].set_title("per variable (forecast vs decoding ceiling)", fontsize=10)
    ax[0, 1].legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax[0, 1], "RMSE (field units)")
    for i, r in enumerate(res):
        lab = os.path.basename(r["dec_run"])
        ax[1, 0].plot(leads, r["uncertainty"]["ratio"], color=SEED_COL[i % 5], lw=1.8, label=lab)
        ax[1, 1].plot(leads, r["uncertainty"]["map_corr"], color=SEED_COL[i % 5], lw=1.8, label=lab)
    ax[1, 0].axhline(1, color="#8c8b86", lw=1, ls=":")
    ax[1, 0].set_title("field spread / RMSE of member mean (vs noise-free state); 1 = calibrated", fontsize=10)
    ax[1, 1].set_title("where: corr(spread map, error map) over cells", fontsize=10)
    ax[1, 1].set_ylim(0, 1)
    for x in ax[1]:
        x.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax[1, 0], "spread / error")
    _style(ax[1, 1], "correlation")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_snapshot(snap: dict, path: str, title: str) -> None:
    cols = [("truth", "noise-free state"), ("m1", "member 1"), ("m2", "member 2"), ("mean", "member mean"),
            ("spread", "member spread (std)")]
    nl = len(snap["leads"])
    fig, ax = plt.subplots(2 * nl, len(cols), figsize=(3.1 * len(cols), 2.9 * 2 * nl))
    for v, var in enumerate(("SSH", "SST")):
        for r, a in enumerate(snap["leads"]):
            row = v * nl + r
            lim = float(np.abs(snap["truth"][r][v]).max())
            for j, (k, name) in enumerate(cols):
                x = ax[row, j]
                if k == "spread":
                    im = x.imshow(snap[k][r][v], cmap=SEQ, vmin=0, origin="lower")
                    fig.colorbar(im, ax=x, fraction=0.046)
                else:
                    x.imshow(snap[k][r][v], cmap=DIV, vmin=-lim, vmax=lim, origin="lower")
                x.set_xticks([]); x.set_yticks([])
                x.set_title(f"{var} lead {a}: {name}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dec", nargs="+", required=True, help="frontal decoder run dirs (decoder.pt)")
    p.add_argument("--hf", nargs="+", required=True, help="hippocampus run dirs; paired by shared encoder run")
    p.add_argument("--members", type=int, default=32, help="ensemble members drawn from the hippocampus")
    p.add_argument("--out", default=".tmps/eval_dec")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hf_by_enc = {torch.load(os.path.join(h, "model.pt"), map_location="cpu", weights_only=False)["encoder_run"]: h
                 for h in args.hf}
    res = []
    for d in sorted(args.dec):
        enc = torch.load(os.path.join(d, "decoder.pt"), map_location="cpu", weights_only=False)["encoder_run"]
        if enc not in hf_by_enc:
            print(f"[skip] {d}: no hippocampus run on encoder {enc}")
            continue
        r, snap = eval_pair(d, hf_by_enc[enc], args.device, args.members)
        res.append(r)
        u = r["uncertainty"]
        rm = " ".join(f"{m[:5]} " + "/".join(f"{r['rmse'][m][a - 1]:.3f}" for a in (1, 8, 16, 64)) for m in METHODS)
        print(f"{os.path.basename(d)} <- {os.path.basename(r['hf_run'])} (M={r['members']}, chaotic "
              f"{r['chaotic_channels']}, non-chaotic latent spread {r['max_nonchaotic_latent_spread']:.0e}): "
              f"RMSE h1/h8/h16/h64 {rm} | spread/error h1 {u['ratio'][0]:.2f} "
              f"h8 {u['ratio'][7]:.2f} h16 {u['ratio'][15]:.2f} h64 {u['ratio'][-1]:.2f} | map corr h8 "
              f"{u['map_corr'][7]:.2f} h64 {u['map_corr'][-1]:.2f}", flush=True)
        plot_snapshot(snap, os.path.join(args.out, f"snap_{os.path.basename(d)}.png"),
                      f"{os.path.basename(d)}: field ensemble from launch t={snap['t0']} (validation, physical units)")
    json.dump(res, open(os.path.join(args.out, "eval_dec.json"), "w"), indent=1)
    plot_curves(res, os.path.join(args.out, "field_forecast.png"),
                f"Field forecast = chaotic latent ensemble (hippocampus) -> frontal cortex, {len(res)} seeds, validation")
    print(f"[eval_dec] wrote {args.out}")


if __name__ == "__main__":
    main()
