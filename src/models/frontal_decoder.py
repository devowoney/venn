"""Frontal decoder — module 3, the emulator's "frontal cortex" (D-033, SOP 05).

The eye-lobe (frozen encoder) turns each field into a latent state S_t [K]; the hippocampus (history forecaster)
evolves that latent history forward, S_hat_{t+a}. The frontal cortex maps a latent state back to the physical
field, so a latent forecast becomes a field forecast:

    x_hat(tau) = mu + sd * ( b + W . window(tau) ),     window(tau) = [S(tau - L) for L in lags]   (standardized)

Why this form:
- LINEAR, closed-form ridge: the testbed field is a linear superposition of patterns and every channel is a linear
  read of it, so the best map from the latent state to the field is linear; a closed form has no training noise and
  the fitted decoder is frozen like the other two parts.
- fitted in "balanced" units (every cell standardized with training statistics), the same units as the
  reconstruction score (D-032), so quiet cells and loud cells are fitted alike; the output is un-scaled to the field.
- `lags` = the latent HISTORY the decoder reads. lags = (0,) reads the current state only. Longer lags fill the
  information hole when the eyes see fewer signals than the field holds (probe 2026-10-01: M=40/K=16 R^2/C
  0.76 -> 0.93); when the state is fully observed (eyes ~ modes) they add nothing, hence the default (0,).
- replaces D-012's pseudo-inverse x(t) + M+(s_hat - M x(t)), which can only move the field inside the K-dim mask
  subspace and freezes everything else at x(t); the decoder instead rebuilds every cell from what the latent
  state implies about it (learned cross-cell structure).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def lag_window(S: torch.Tensor, lags) -> torch.Tensor:
    """Causal latent window of an OBSERVED series S [T,K] -> [T, len(lags)*K]; row tau = [S(tau-L) for L in lags].

    Before the record starts, the first state is repeated (affects only the first max(lags) rows).
    """
    T = S.shape[0]
    t = torch.arange(T, device=S.device)
    return torch.cat([S[torch.clamp(t - int(L), 0, T - 1)] for L in lags], dim=1)


class FrontalDecoder(nn.Module):
    """Latent window [..., len(lags)*K] -> field [..., V, H, W] (physical units). Frozen once fitted."""

    def __init__(self, K: int, lags, field_shape, ridge: float = 1e-3):
        super().__init__()
        self.K, self.lags, self.field_shape, self.ridge = K, tuple(int(L) for L in lags), tuple(field_shape), ridge
        F, N = len(self.lags) * K, int(torch.tensor(field_shape).prod())
        # every statistic below comes from the training slice only (set by `fit`)
        self.register_buffer("mu_s", torch.zeros(F))       # latent-window standardization
        self.register_buffer("sd_s", torch.ones(F))
        self.register_buffer("mu_y", torch.zeros(N))       # per-cell field standardization ("balanced")
        self.register_buffer("sd_y", torch.ones(N))
        self.register_buffer("W", torch.zeros(F, N))       # ridge weights, balanced units
        self.register_buffer("b", torch.zeros(N))

    @torch.no_grad()
    def fit(self, S: torch.Tensor, Y: torch.Tensor, t_fit: int) -> "FrontalDecoder":
        """Closed-form ridge on [0, t_fit): S [T,K] observed latent series, Y [T,N] flattened field (same times)."""
        X = lag_window(S.double(), self.lags)[:t_fit]
        Yf = Y.double()[:t_fit]
        self.mu_s, self.sd_s = X.mean(0).float(), (X.std(0) + 1e-12).float()
        self.mu_y, self.sd_y = Yf.mean(0).float(), (Yf.std(0) + 1e-12).float()
        Xs = (X - self.mu_s.double()) / self.sd_s.double()
        Yb = (Yf - self.mu_y.double()) / self.sd_y.double()
        # centred features -> intercept = mean of Yb = 0; ridge relative to the sample count (scale-free)
        G = Xs.T @ Xs + self.ridge * t_fit * torch.eye(Xs.shape[1], dtype=Xs.dtype, device=Xs.device)
        self.W = torch.linalg.solve(G, Xs.T @ Yb).float()
        self.b = Yb.mean(0).float()
        return self

    def forward(self, window: torch.Tensor, balanced: bool = False) -> torch.Tensor:
        """window [..., F] (physical latent units) -> field [..., V,H,W]; balanced=True returns standardized cells."""
        yb = ((window - self.mu_s) / self.sd_s) @ self.W + self.b
        if balanced:
            return yb.reshape(*window.shape[:-1], *self.field_shape)
        return (self.mu_y + self.sd_y * yb).reshape(*window.shape[:-1], *self.field_shape)
