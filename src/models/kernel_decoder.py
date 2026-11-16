"""Kernel-aware Gaussian decoder (D-056): the raw field from the latent state, READING the observer's own kernel.

The observer is a fixed linear kernel, S = K x (frozen masks K [k, V*H*W], no normalization; checked: |Kx - S| / |S|
~ 2e-6). The decoder therefore knows exactly which part of the field the latent fixes, and only has to supply the rest:

    mu(z)   = K+ S  +  P_perp f_theta(z, K)       K+ = K^T (K K^T)^-1, P_perp = I - K+ K   (rank-k correction)
    p(x|z)  = N(x; mu(z), diag sigma^2(z))         sigma from a second head (heteroscedastic, per cell)

- K mu(z) = S exactly, for every z: the decoded field always reproduces the latent it came from; the network only
  shapes the null space (the ~99.5 % of the field no eye resolves: fronts, eddy shapes, small scales).
- Inputs make the kernel visible: each latent value painted on its own footprint, z_i m_i (k*V channels), the
  pseudo-inverse field K+ S and the adjoint field K^T z (where the latent says what), the summed footprint (coverage).
- Trained by the Gaussian likelihood on TRUE pairs (z(t), x(t)): p(x | z) is the decoder of the KL split
  KL(p(x) || p(x_hat)) = E[-log p_zhat(Kx)] + E[-log p(x | Kx)] + const (design record §17), independent of the
  forecast; at forecast time z ~ p(zhat) is pushed through it, x = mu(z) + sigma(z) eps.
- z is in CANONICAL units (live channels standardized: z = (S - mu_S) / sd_S); the kernel acts on S.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flow_decoder import PConv, ResBlock


class KernelDecoder(nn.Module):
    """mu(z), log sigma^2(z) on fields [B, V, H, W] from canonical latents z [B, k], given the kernel masks [k, V, H, W]."""

    def __init__(self, masks: torch.Tensor, mu_s: torch.Tensor, sd_s: torch.Tensor, ch=(64, 128, 192), emb: int = 256,
                 logvar_range=(-12.0, 4.0)):
        super().__init__()
        k, V, H, W = masks.shape
        self.k, self.V, self.H, self.W = k, V, H, W
        self.lv_lo, self.lv_hi = logvar_range
        Kf = masks.reshape(k, -1).double()
        self.register_buffer("K", Kf.float())                                   # [k, N]
        self.register_buffer("Kpinv", (Kf.T @ torch.linalg.inv(Kf @ Kf.T)).float())   # [N, k]
        peak = masks.abs().flatten(1).amax(1).clamp_min(1e-8)
        self.register_buffer("m_unit", (masks / peak[:, None, None, None]).float())    # footprints scaled to peak 1
        self.register_buffer("cover", self.m_unit.abs().sum(0, keepdim=True) / k)    # [1, V, H, W]
        self.register_buffer("mu_s", mu_s.float())
        self.register_buffer("sd_s", sd_s.float())
        self.emb = emb
        self.z_mlp = nn.Sequential(nn.Linear(k, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = PConv(k * V + 3 * V, ch[0])
        self.down = nn.ModuleList()
        c = ch[0]
        for co in ch:
            self.down.append(ResBlock(c, co, emb))
            c = co
        self.mid = nn.ModuleList([ResBlock(c, c, emb), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for co in reversed(ch):
            self.up.append(ResBlock(c + co, co, emb))
            c = co
        self.out_norm = nn.GroupNorm(8, c)
        self.out = PConv(c, 2 * V)                                             # [f (null-space shaping), log sigma^2]
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def seen(self, z: torch.Tensor) -> torch.Tensor:
        """K+ S: the minimum-norm field that reproduces the latent exactly [B, V, H, W]."""
        S = z * self.sd_s + self.mu_s
        return (S @ self.Kpinv.T).view(-1, self.V, self.H, self.W)

    def project_null(self, f: torch.Tensor) -> torch.Tensor:
        """P_perp f = f - K+ K f: removes from f anything the kernel would see."""
        ff = f.flatten(1)
        return (ff - (ff @ self.K.T) @ self.Kpinv.T).view_as(f)

    def forward(self, z: torch.Tensor):
        B = z.shape[0]
        seen = self.seen(z)
        paint = (z[:, :, None, None, None] * self.m_unit[None]).reshape(B, self.k * self.V, self.H, self.W)
        adj = torch.einsum("bk,kvhw->bvhw", z, self.m_unit) / self.k
        h = self.inp(torch.cat([paint, seen, adj, self.cover.expand(B, -1, -1, -1)], 1))
        e = self.z_mlp(z)
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, e)
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)
        for blk in self.mid:
            h = blk(h, e)
        for i, blk in enumerate(self.up):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(torch.cat([h, skips.pop()], 1), e)
        f, lv = self.out(F.silu(self.out_norm(h))).chunk(2, dim=1)
        mean = seen + self.project_null(f)
        logvar = lv.clamp(self.lv_lo, self.lv_hi)
        return mean, logvar

    def nll(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian negative log-likelihood per cell (constant 0.5 log 2pi dropped)."""
        mean, logvar = self(z)
        return 0.5 * ((x - mean) ** 2 * torch.exp(-logvar) + logvar).mean()

    @torch.no_grad()
    def sample(self, z: torch.Tensor, eps: torch.Tensor | None = None) -> torch.Tensor:
        mean, logvar = self(z)
        eps = torch.randn_like(mean) if eps is None else eps
        return mean + torch.exp(0.5 * logvar) * eps
