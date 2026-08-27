"""Selection-mask encoder (v0, D-015 as revised 2026-08-26).

Soft-binary masks `sigmoid(logit/temp)` over the full (V,H,W) volume. `temp` is annealed toward
hard-binary by the training loop. Straight-through binary and the per-layer domain are deferred
upgrade paths.

Kernel semantics (D-013): each mask is a spatial sensor footprint at some scale/resolution;
masks may overlap and need NOT be orthogonal. The COLLECTION of channels is the encoding.

Init matters (D-016): a near-flat init (all masks ~0.5 everywhere) makes every channel read the
same spatial mean -> the channel correlation matrix is rank-1 and the whitening is degenerate from
step 0. `init="multiscale"` instead gives each channel a smooth random footprint at its own length
scale (ladder from a few cells to basin-scale), which is both the D-004 "random masks" start and
the D-013 multi-scale-sensor prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of [N,1,H,W] with reflect padding. sigma in cells."""
    if sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1))


def multiscale_logits(K: int, V: int, H: int, W: int, init_std: float,
                      sigma_min: float, sigma_max: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Random smooth mask logits [K,V,H,W]; channel k gets its own length scale sigma_k.

    Scales run geometrically from `sigma_min` (small energetic patch) to `sigma_max` (basin-scale
    footprint), so the K channels start out looking at genuinely different regions AND scales.
    Each channel is standardized to `init_std`, so `sigmoid(logit/temp0)` spans a real 0..1 range
    instead of hugging 0.5.
    """
    noise = torch.randn(K * V, 1, H, W, generator=generator)
    sig = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), K)
    out = torch.empty_like(noise)
    for k in range(K):
        sl = slice(k * V, (k + 1) * V)
        b = _blur2d(noise[sl], float(sig[k]))
        out[sl] = (b - b.mean()) / (b.std() + 1e-8)
    return (init_std * out).view(K, V, H, W)


class SelectionEncoder(nn.Module):
    """field[T,V,H,W] -> S[T,K] scalar channels via K soft-binary selection masks."""

    def __init__(self, K: int, V: int, H: int, W: int, temp: float = 1.0, norm: str = "none",
                 init: str = "multiscale", init_std: float = 1.5,
                 sigma_min: float = 1.0, sigma_max: float = 16.0, signed: bool = False,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.temp = temp
        self.signed = signed
        # normalization of s_i: "none" (whitening owns the scale -- default, D-015 rev),
        # "sqrt" (divide by sqrt soft-count), "softcount" (divide by soft-count -- scale-invariant,
        # makes unit-variance whitening UNREACHABLE; kept only for ablation).
        self.norm = norm
        # mask logits over the full (V,H,W) volume, one set per channel -> [K,V,H,W]
        if init == "multiscale":
            w = multiscale_logits(K, V, H, W, init_std, sigma_min, sigma_max, generator)
        elif init == "random":                                # white-noise logits (no structure)
            w = init_std * torch.randn(K, V, H, W, generator=generator)
        elif init == "flat":                                  # legacy near-0.5 init (degenerate)
            w = 0.01 * torch.randn(K, V, H, W, generator=generator)
        else:
            raise ValueError(f"unknown init: {init}")
        self.logits = nn.Parameter(w)

    def masks(self) -> torch.Tensor:
        """Selection masks [K,V,H,W]: soft-binary in [0,1], or signed in [-1,1] if `signed`.

        `signed=True` is a DIAGNOSTIC ABLATION only (D-005 rejects free continuous kernels): it
        tests whether the residual channel correlation is imposed by non-negativity, since a
        non-negative sensor can never cancel a shared positive large-scale mode.
        """
        if self.signed:
            return torch.tanh(self.logits / self.temp)
        return torch.sigmoid(self.logits / self.temp)

    def soft_count(self) -> torch.Tensor:
        """Effective number of selected cells per channel [K]."""
        return self.masks().abs().sum(dim=(1, 2, 3))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """field [T,V,H,W] -> S [T,K]. s_i = <mask_i, field> (optionally count-normalized)."""
        m = self.masks()                                     # [K,V,H,W]
        num = torch.einsum("tvhw,kvhw->tk", field, m)        # [T,K] masked sum
        if self.norm == "softcount":
            return num / m.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        if self.norm == "sqrt":
            return num / m.sum(dim=(1, 2, 3)).clamp_min(1e-6).sqrt()
        return num                                           # "none": whitening sets the scale
