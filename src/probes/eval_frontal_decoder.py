"""Streaming FIELD forecast: hippocampus (latent forecast) -> frontal cortex (decoder) -> physical field (SOP 05).

The full emulator chain, scored on the VALIDATION set: at every launch t the hippocampus forecasts the latent state
S_hat(t+a) from the history up to t (one causal pass, as in eval_history_forecaster), and the frozen frontal decoder
maps it to the field x_hat(t+a). Only launches whose targets all lie in [t_tr, T) are scored.

Score per lead a: balanced R^2 of the field (every cell standardized with training stats, D-032) divided by the
noise ceiling C, i.e. the fraction of the recoverable field state forecast. Compared against
  decoded truth  : the decoder applied to the TRUE future latent state -> decoding ceiling (perfect hippocampus)
  persistence    : the last observed field x(t) held constant
  climatology    : the training-mean field (R^2 ~ 0 by construction)
  pseudo-inverse : D-012's forecaster x(t) + M+(S_hat(t+a) - M x(t)), which only moves the field inside the mask
                   subspace and freezes the rest at x(t) -- the design the frontal decoder replaces
Decoder error and forecast error separate as: decoded truth = what is lost in decoding alone; decoded truth minus
forecast = what is lost in the hippocampus.

Run:  python -m src.probes.eval_frontal_decoder --dec .tmps/runs_dec/dec10k_seed* \
          --hf <hf runs, paired with the decoder through their shared encoder run> --out .tmps/eval_dec/final10k
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.colors import LinearSegmentedColormap      # noqa: E402

from src.models.frontal_decoder import FrontalDecoder, lag_window   # noqa: E402
from src.probes.eval_history_forecaster import load_run as load_hf   # noqa: E402
from src.train.fit_frontal_decoder import load_eye          # noqa: E402

METHODS = ("decoded_truth", "forecast", "pseudo_inverse", "persistence", "climatology")
COL = dict(decoded_truth="#8c8b86", forecast="#2a78d6", pseudo_inverse="#eb6834", persistence="#1baf7a",
           climatology="#eda100")
LABEL = dict(decoded_truth="decoded TRUE latent (decoding ceiling)", forecast="hippocampus -> frontal decoder",
             pseudo_inverse="hippocampus -> pseudo-inverse (D-012)", persistence="persistence x(t)",
             climatology="climatology")
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#eb6834", "#a8401b"])
LEADS_SHOW = (1, 8, 16, 64)


def load_decoder(run: str, device: str) -> FrontalDecoder:
    ck = torch.load(os.path.join(run, "decoder.pt"), map_location=device, weights_only=False)
    dec = FrontalDecoder(ck["K"], ck["lags"], ck["field_shape"], ck["ridge"]).to(device)
    dec.load_state_dict(ck["state"])
    dec.encoder_run, dec.t_tr = ck["encoder_run"], ck["t_tr"]
    return dec.eval()


@torch.no_grad()
def latent_forecast(model, nm, S: torch.Tensor, members: int = 32) -> torch.Tensor:
    """S_hat [T-1, A, K] in PHYSICAL latent units: the forecast made at t from S_0..S_t (ensemble model: the mean)."""
    mu, sd = (torch.tensor(nm[k], dtype=torch.float32, device=S.device) for k in ("mu", "sd"))
    z = ((S.float() - mu) / sd)[None, :-1]
    if model.ensemble:
        torch.manual_seed(0)                                   # same member draws as eval_history_forecaster
        zhat = model(z, members=members)[0].mean(1)            # ensemble mean = the best estimate
    else:
        zhat = model(z)[0]
    return zhat * sd + mu


def forecast_window(S: torch.Tensor, Shat: torch.Tensor, L: torch.Tensor, a: int, lags) -> torch.Tensor:
    """Decoder window at the target time t+a for launches L: lag steps after t take the FORECAST, earlier ones the
    observed history (the emulator never sees an observation past the launch)."""
    parts = []
    for g in lags:
        off = a - int(g)                                       # position of S(t+a-g) relative to the launch t
        parts.append(Shat[L, off - 1] if off >= 1 else S[L + off])
    return torch.cat(parts, dim=1)


def r2_pooled(y: torch.Tensor, yh: torch.Tensor) -> float:
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


@torch.no_grad()
def eval_pair(dec_run: str, hf_run: str, device: str) -> dict:
    dec = load_decoder(dec_run, device)
    model, nm, _, t_tr = load_hf(hf_run, device)
    S, Y, shape, obs_noise, _ = load_eye(dec.encoder_run, device)
    assert t_tr == dec.t_tr, (t_tr, dec.t_tr)
    T, A, N = len(S), model.A, Y.shape[1]
    Sf = S.float()
    Shat = latent_forecast(model, nm, Sf)                      # [T-1,A,K]
    Yb = ((Y.float() - dec.mu_y) / dec.sd_y)                   # balanced field [T,N]
    ceil = float((1 - obs_noise ** 2 / Y[:t_tr].var(0)).mean())
    var_ = torch.arange(N, device=device) // (N // shape[0])   # variable index of each flattened cell (SSH 0, SST 1)

    # D-012 pseudo-inverse: masks M [K,N] (s = M.x exactly, verified); M+ = minimal-norm inverse
    masks = np.load(os.path.join(dec.encoder_run, "artifacts.npz"))["masks"]
    M = torch.tensor(masks.reshape(len(masks), -1), dtype=torch.float64, device=device)
    Mp = torch.linalg.pinv(M).float()                          # [N,K]
    win_true = lag_window(Sf, dec.lags)                        # decoder windows of the TRUE latent series

    L = torch.arange(t_tr - 1, T - A, device=device)           # launches with every target in validation
    out = {m: np.zeros(A) for m in METHODS}
    per_var = {m: np.zeros((A, shape[0])) for m in ("forecast", "decoded_truth")}
    for a in range(1, A + 1):
        y = Yb[L + a]                                          # [n,N] truth (balanced)
        preds = dict(
            decoded_truth=dec(win_true[L + a], balanced=True).reshape(len(L), N),
            forecast=dec(forecast_window(Sf, Shat, L, a, dec.lags), balanced=True).reshape(len(L), N),
            pseudo_inverse=(Y.float()[L] + (Shat[L, a - 1] - Sf[L]) @ Mp.T - dec.mu_y) / dec.sd_y,
            persistence=Yb[L],
            climatology=torch.zeros_like(y))
        for m, yh in preds.items():
            out[m][a - 1] = r2_pooled(y, yh) / ceil
        for m in per_var:
            for v in range(shape[0]):
                c = var_ == v
                per_var[m][a - 1, v] = r2_pooled(y[:, c], preds[m][:, c]) / float(
                    (1 - obs_noise ** 2 / Y[:t_tr][:, c].var(0)).mean())

    # one launch, physical units, for the snapshot figure
    t0 = int(L[len(L) // 3])
    snap = dict(t0=t0, truth=[], forecast=[], pinv=[])
    for a in LEADS_SHOW:
        snap["truth"].append(Y[t0 + a].reshape(shape).cpu().numpy())
        w = forecast_window(Sf, Shat, torch.tensor([t0], device=device), a, dec.lags)
        snap["forecast"].append(dec(w)[0].cpu().numpy())
        snap["pinv"].append((Y.float()[t0] + (Shat[t0, a - 1] - Sf[t0]) @ Mp.T).reshape(shape).cpu().numpy())
    return dict(dec_run=os.path.relpath(dec_run), hf_run=hf_run, ensemble=bool(model.ensemble), ceiling=ceil,
                n_launch=len(L), r2c={m: v.tolist() for m, v in out.items()},
                r2c_var={m: v.tolist() for m, v in per_var.items()}), snap


def plot_curves(res: list[dict], path: str, title: str) -> None:
    A = len(res[0]["r2c"]["forecast"])
    leads = np.arange(1, A + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for m in METHODS:
        v = np.array([r["r2c"][m] for r in res])
        ax[0].plot(leads, v.mean(0), color=COL[m], lw=2, label=LABEL[m])
        ax[0].fill_between(leads, v.min(0), v.max(0), color=COL[m], alpha=0.15, lw=0)
    ax[0].set_ylim(-0.6, 1.05)
    ax[0].set_ylabel("field R² / noise ceiling (validation)")
    ax[0].set_title("forecast field skill vs lead (mean, band = seed range)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    for v, (name, ls) in enumerate((("SSH", "-"), ("SST", "--"))):
        f = np.array([r["r2c_var"]["forecast"] for r in res])[:, :, v]
        ax[1].plot(leads, f.mean(0), color=COL["forecast"], ls=ls, lw=2, label=f"{name} forecast")
        d = np.array([r["r2c_var"]["decoded_truth"] for r in res])[:, :, v]
        ax[1].plot(leads, d.mean(0), color=COL["decoded_truth"], ls=ls, lw=1.5, label=f"{name} decoding ceiling")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("per variable", fontsize=10)
    ax[1].legend(fontsize=8, frameon=False, loc="lower left")
    for x in ax:
        x.set_xlabel("lead a (steps)")
        x.set_xscale("log", base=2)
        x.grid(color="#e6e5df", lw=0.6)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_snapshot(snap: dict, path: str, title: str) -> None:
    rows = [("truth", "truth x(t+a)"), ("forecast", "frontal decoder"), ("pinv", "pseudo-inverse (D-012)")]
    fig, ax = plt.subplots(2 * len(rows), len(LEADS_SHOW), figsize=(3 * len(LEADS_SHOW), 2.6 * 2 * len(rows)))
    for v, var in enumerate(("SSH", "SST")):
        lim = float(np.abs(np.array(snap["truth"])[:, v]).max())
        for r, (key, name) in enumerate(rows):
            for j, a in enumerate(LEADS_SHOW):
                x = ax[v * len(rows) + r, j]
                im = x.imshow(snap[key][j][v], cmap=DIV, vmin=-lim, vmax=lim, origin="lower")
                x.set_xticks([]); x.set_yticks([])
                if r == 0:
                    x.set_title(f"lead {a}", fontsize=10)
                if j == 0:
                    x.set_ylabel(f"{var}\n{name}", fontsize=9)
        fig.colorbar(im, ax=ax[v * len(rows):(v + 1) * len(rows), :].ravel().tolist(), fraction=0.02, shrink=0.6)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dec", nargs="+", required=True, help="frontal decoder run dirs (decoder.pt)")
    p.add_argument("--hf", nargs="+", required=True, help="hippocampus run dirs; paired by shared encoder run")
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
        r, snap = eval_pair(d, hf_by_enc[enc], args.device)
        res.append(r)
        f = np.array(r["r2c"]["forecast"])
        print(f"{os.path.basename(d)} <- {os.path.basename(r['hf_run'])}: C={r['ceiling']:.3f} | R2/C forecast "
              f"h1 {f[0]:.3f} h8 {f[7]:.3f} h16 {f[15]:.3f} h64 {f[-1]:.3f} | pinv h1 {r['r2c']['pseudo_inverse'][0]:.3f}"
              f" h8 {r['r2c']['pseudo_inverse'][7]:.3f} | pers h8 {r['r2c']['persistence'][7]:.3f}", flush=True)
        plot_snapshot(snap, os.path.join(args.out, f"snap_{os.path.basename(d)}.png"),
                      f"{os.path.basename(d)}: field forecast from launch t={snap['t0']} (validation)")
    json.dump(res, open(os.path.join(args.out, "eval_dec.json"), "w"), indent=1)
    plot_curves(res, os.path.join(args.out, "field_forecast.png"),
                f"Field forecast = hippocampus -> frontal cortex, {len(res)} seeds, validation launches")
    print(f"[eval_dec] wrote {args.out}")


if __name__ == "__main__":
    main()
