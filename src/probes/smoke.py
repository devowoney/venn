"""Phase L.2 handshake smoke test.

Generate one synthetic clip, assert the schema, run one forward + one backward pass of the
PROVISIONAL K=16 selection encoder with the slowness loss, and log tensor stats. No training.

Run:  conda run -n oceanai python -m src.probes.smoke
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# make `src` importable when run as a script or module from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.synthetic import GenConfig, generate_field   # noqa: E402
from src.models.encoder import SelectionEncoder             # noqa: E402

K = 16
SEED = 0
TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".tmps")


def main() -> int:
    torch.manual_seed(SEED)
    ok = True

    # --- (1) generate + schema assertions ---------------------------------------------------
    cfg = GenConfig()
    field_np, truth = generate_field(cfg, seed=SEED)
    assert field_np.shape == (2000, 2, 64, 64), field_np.shape
    assert field_np.dtype == np.float32, field_np.dtype
    assert np.isfinite(field_np).all()
    fams = [m["family"] for m in truth["modes"]]
    print(f"[1] field {field_np.shape} {field_np.dtype} finite=OK | "
          f"latent modes={truth['K_modes']} families={fams}")
    for v, name in enumerate(("SSH", "SST")):
        fv = field_np[:, v]
        print(f"    var{v} ({name}): mean={fv.mean():+.3f} std={fv.std():.3f} "
              f"min={fv.min():+.3f} max={fv.max():+.3f}")

    # --- (2) encoder forward ----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    field = torch.from_numpy(field_np).to(device)            # [T,V,H,W]
    enc = SelectionEncoder(K=K, V=2, H=64, W=64).to(device)
    S = enc(field)                                           # [T,K]
    assert S.shape == (2000, K), S.shape
    assert torch.isfinite(S).all()
    print(f"[2] encoder K={K} on {device}: S {tuple(S.shape)} "
          f"mean={S.mean().item():+.3f} std={S.std().item():.3f}")

    # --- (3) slowness loss + one backward ---------------------------------------------------
    loss = ((S[1:] - S[:-1]) ** 2).sum(dim=1).mean()         # L = mean_t sum_i (s[t+1]-s[t])^2
    loss.backward()
    g = enc.logits.grad
    gnorm = g.norm().item()
    grad_ok = torch.isfinite(g).all().item() and gnorm > 0
    print(f"[3] slowness loss={loss.item():.4f} | mask-grad norm={gnorm:.3e} "
          f"finite&nonzero={grad_ok}")

    # --- persist scalars for inspection (ephemeral) -----------------------------------------
    os.makedirs(TMP, exist_ok=True)
    np.savez(os.path.join(TMP, "smoke_S.npz"),
             S=S.detach().cpu().numpy(), families=np.array(fams))
    print(f"    saved S -> {os.path.join(TMP, 'smoke_S.npz')}")

    ok = grad_ok and bool(np.isfinite(field_np).all())
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
