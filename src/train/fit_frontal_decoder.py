"""Fit the frontal decoder (module 3, "frontal cortex", SOP 05) on the TRAINING set only, then freeze it.

Input : a frozen eye-lobe run (artifacts.npz S[T,K] + .hydra/config.yaml to regenerate its field); only [0, t_tr)
        reaches the fit. Output: decoder.pt + metrics.json (reconstruction R^2 / noise ceiling on training and
        validation, from the TRUE latent state) in the Hydra run dir .tmps/runs_dec/<stamp>/.

Run:  python -m src.train.fit_frontal_decoder dec.encoder_run=.tmps/runs/hf10k_enc_seed0
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

from src.data.synthetic import GenConfig, generate_field  # noqa: E402
from src.models.frontal_decoder import FrontalDecoder, lag_window   # noqa: E402


def load_eye(run: str, device: str):
    """Frozen eye-lobe run -> (latent S [T,K], flattened field Y [T,N], field shape, obs_noise, encoder t_train).

    The field is regenerated from the run's own data config + seed (the generator is deterministic), so it is the
    exact field the eye was trained on and read.
    """
    art = np.load(os.path.join(run, "artifacts.npz"), allow_pickle=True)
    ecfg = OmegaConf.load(os.path.join(run, ".hydra/config.yaml"))
    field, _ = generate_field(GenConfig(**OmegaConf.to_container(ecfg.data, resolve=True)), seed=ecfg.seed)
    S = torch.tensor(art["S"], dtype=torch.float64, device=device)
    Y = torch.tensor(field.reshape(len(field), -1), dtype=torch.float64, device=device)
    return S, Y, field.shape[1:], float(ecfg.data.obs_noise), int(ecfg.train.get("t_train") or len(field))


def recon_r2(dec: FrontalDecoder, S: torch.Tensor, Y: torch.Tensor, sl: slice) -> float:
    """Balanced R^2 (every cell standardized with training stats) of the decoded TRUE latent state over slice sl."""
    with torch.no_grad():
        yh = dec(lag_window(S.float(), dec.lags), balanced=True).reshape(len(S), -1).double()[sl]
    y = ((Y - dec.mu_y.double()) / dec.sd_y.double())[sl]
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


@hydra.main(version_base=None, config_path="../../config", config_name="frontal_decoder")
def main(cfg: DictConfig) -> None:
    dc = cfg.dec
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    run = dc.encoder_run if os.path.isabs(dc.encoder_run) else os.path.join(hydra.utils.get_original_cwd(),
                                                                          dc.encoder_run)
    S, Y, shape, obs_noise, t_enc = load_eye(run, device)
    t_tr = int(dc.t_tr)
    # the three parts share ONE training set: a decoder fitted past the eye's training cut would see validation
    assert t_tr == t_enc, f"dec.t_tr={t_tr} != encoder train.t_train={t_enc}"
    print(f"[dec] encoder={run} S={tuple(S.shape)} field={tuple(shape)} train=[0,{t_tr}) val=[{t_tr},{len(S)})")

    dec = FrontalDecoder(S.shape[1], dc.lags, shape, ridge=float(dc.ridge)).to(device).fit(S, Y, t_tr)
    # noise ceiling (D-032): the generator's iid observation noise caps any reconstruction at C, so R^2/C = the
    # fraction of the recoverable state rebuilt
    ceiling = float((1 - obs_noise ** 2 / Y[:t_tr].var(0)).mean())
    r2_tr, r2_va = recon_r2(dec, S, Y, slice(0, t_tr)), recon_r2(dec, S, Y, slice(t_tr, len(S)))
    print(f"[dec] lags={list(dec.lags)} ceiling C={ceiling:.3f} | R2/C train {r2_tr / ceiling:.3f}  "
          f"val {r2_va / ceiling:.3f}")
    torch.save(dict(state=dec.state_dict(), K=dec.K, lags=list(dec.lags), field_shape=list(shape),
                    ridge=dec.ridge, encoder_run=run, t_tr=t_tr), "decoder.pt")
    json.dump(dict(ceiling=ceiling, r2_train=r2_tr, r2_val=r2_va, r2c_train=r2_tr / ceiling,
                   r2c_val=r2_va / ceiling, lags=list(dec.lags), t_tr=t_tr), open("metrics.json", "w"), indent=2)
    print(f"[dec] done. run dir: {os.getcwd()}")


if __name__ == "__main__":
    main()
