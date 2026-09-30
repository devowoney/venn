"""History forecaster — module 2, the emulator's "prefrontal cortex" (SOP 04).

The eye-lobe (frozen encoder) turns each field snapshot into a latent state psi_t [K]. Everything observed so
far, psi_0 ... psi_t, is the HISTORY of the system. This model holds no input and no state of its own beyond
that history: it reads the history and says what the latent state will be at its end (t+1), and further on
(t+a) as a readout of how far the history carries.

    zhat_{t+a} = z_t + head_a(h_t),        h_t = causal-attention(z_0 ... z_t)

Why each piece:
- one token per time step with ALL K channels: the phase of a mode is spread over several channels (each mask
  sees it differently), so a single channel cannot tell a rising from a falling phase — the joint state can.
- the token also carries the increment z_t - z_{t-1}, taken from the history itself: the eye-lobe is purely
  spatial, so one snapshot holds no rate of change.
- causal attention over the whole history = full sight; rotary positions make attention depend on the LAG
  between two states, not on absolute time, so the history can keep growing past the training length.
- direct heads, residual form, zero-initialized: the untrained model IS persistence, and forecasts are never
  fed back (no error feedback), so "how far can it predict" is measured without compounding its own errors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotary(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotate feature pairs of x [B,h,T,dh] by angle pos*freq, so that q.k depends only on the lag."""
    dh = x.shape[-1]
    # slowest pair turns once per ~8000 steps (beyond the 4000-step record): long lags stay distinguishable
    freq = torch.exp(-math.log(8000.0) * torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh)
    ang = pos[:, None].to(x.dtype) * freq[None, :]                     # [T,dh/2]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class CausalBlock(nn.Module):
    """Pre-norm block: causal self-attention over the history, then a per-step MLP."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.drop = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rotary(q, pos), rotary(k, pos)
        # is_causal: the step at t attends to steps <= t only -- the forecast made at t uses the history up to t
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + F.dropout(self.proj(a.transpose(1, 2).reshape(B, T, d)), self.drop, self.training)
        return x + F.dropout(self.mlp(self.n2(x)), self.drop, self.training)


class HistoryForecaster(nn.Module):
    """History z [B,T,K] -> forecasts zhat [B,T,A,K]; zhat[:, t, a-1] is the forecast of z_{t+a} made at t.

    Causal by construction, so ONE pass over a record gives, at every t, the forecast a streaming emulator
    would make with the history available at t (initialized on the training series, grown by appending).
    """

    def __init__(self, K: int, leads: int = 64, d: int = 64, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2, chaos_mask=None, noise_dim: int = 16):
        super().__init__()
        self.K, self.A = K, leads
        self.inp = nn.Linear(2 * K, d)                                 # token = [state, increment]
        self.blocks = nn.ModuleList(CausalBlock(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, leads * K)                            # one direct head per lead
        nn.init.zeros_(self.head.weight)                               # start exactly at persistence
        nn.init.zeros_(self.head.bias)
        # --- ensemble head (SOP 04 "chaotic channels carry the uncertainty"): only built when a chaos mask is
        # given. It turns (history summary h_t, noise eps) into a member-specific departure from the base
        # forecast, and the mask zeroes it on every non-chaotic channel, which therefore stay deterministic.
        self.ensemble = chaos_mask is not None
        if self.ensemble:
            self.noise_dim = noise_dim
            self.register_buffer("chaos", torch.as_tensor(chaos_mask, dtype=torch.float32))   # [K] 1 = chaotic
            self.gen = nn.Sequential(nn.Linear(d + noise_dim, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d),
                                     nn.GELU(), nn.Linear(2 * d, leads * K))
            # per-lead spread inflation, fitted AFTER training on held-out data (SOP 04 "spread calibration");
            # 1 = raw ensemble. Trained on one trajectory, the raw ensemble is overconfident on unseen days.
            self.register_buffer("spread", torch.ones(leads))

    def forward(self, z: torch.Tensor, members: int = 0) -> torch.Tensor:
        """members = 0: deterministic forecast [B,T,A,K]. members = M > 0 (ensemble model only): M possible
        futures [B,T,M,A,K]; non-chaotic channels are identical in every member (= the base forecast)."""
        B, T, K = z.shape
        dz = torch.cat([torch.zeros_like(z[:, :1]), z[:, 1:] - z[:, :-1]], dim=1)   # rate of change
        x = self.inp(torch.cat([z, dz], dim=-1))
        pos = torch.arange(T, device=z.device)
        for blk in self.blocks:
            x = blk(x, pos)
        h = self.norm(x)                                               # [B,T,d] summary of the history at t
        base = z[:, :, None, :] + self.head(h).view(B, T, self.A, K)   # residual: persistence = zero output
        if members == 0:
            return base
        assert self.ensemble, "members > 0 needs a model built with chaos_mask"
        # one noise draw per (launch, member), shared by all leads -> each member is a coherent possible future
        eps = torch.randn(B, T, members, self.noise_dim, device=z.device)
        hm = h[:, :, None, :].expand(B, T, members, h.shape[-1])
        dev = self.gen(torch.cat([hm, eps], dim=-1)).view(B, T, members, self.A, K)
        # calibration: widen the members around their own mean (the best estimate itself is not moved)
        dm = dev.mean(2, keepdim=True)
        dev = dm + self.spread[:, None] * (dev - dm)
        return base[:, :, None] + dev * self.chaos                     # chaos mask: only chaotic channels spread
