"""Flow-matching decoder — module 3 as a NON-LINEAR, probabilistic map from a latent forecast to the raw field (D-048,
SOP 05 "Flow-matching decoder").

The history forecaster gives the next latent state with a Gaussian uncertainty, z(t+1) ~ N(zhat, Sigma). This decoder
turns that whole latent distribution into a distribution of raw fields: each member draws its own latent state z~ and
its own noise, and the flow carries that noise to a field that fits z~.

Generative model = conditional FLOW MATCHING. "Time" here is FLOW time s in [0, 1]; it has nothing to do with the
physical time step t -> t+1. Two sources (`source`):

  noise  (D-049, default)  x0 = sigma0 * eps           pure Gaussian noise; the velocity reads x_s only, conditioned
                                                       on (s, z~): the field is decoded from the latent state ALONE
  launch (D-048)           x0 = x(t) + sigma0 * eps    the launch field, noised; x(t) is also a clean input channel

    x1   = x(t+1)                                      target: the observed raw field that follows
    x_s  = (1 - s) x0 + s x1                           straight path
    loss = | v(x_s, s | z~ [, x(t)]) - (x1 - x0) |^2   the network learns the velocity of that path
    member: integrate dx/ds = v from s = 0 to 1

Why each piece:
- source "noise" is the pure test of a NON-LINEAR decoder: nothing but z~ tells the flow which field to make, so any
  gain over the linear decoder D(z~) is decoding, not information smuggled in through x(t) (the open caveat of F-46).
- source "launch" keeps the D-048 design for comparison: most of x(t+1) is already in x(t), so that flow only moves what
  changes.
- z~ enters every block twice: by FiLM (one scale and shift per feature, the same at every cell) and, when zmap > 0, as
  a SPATIAL map -- a learned linear z~ -> [zmap, 16, 16] grid, resized to each resolution and added to the features.
  The spatial map is what tells a pure-noise flow WHERE things are: FiLM + circular convolutions are exactly
  x-translation equivariant, so without it the members put the eddies at random x-positions (F-47, D-050).
- convolutions are circular in x (the testbed is periodic in x) and replicate-padded in y.
- the output layer starts at zero: the untrained flow moves nothing (each member = its source sample).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of the flow time s [B] -> [B, dim] (frequencies 1 .. 1000, like diffusion timesteps)."""
    half = dim // 2
    freq = torch.exp(-math.log(1000.0) * torch.arange(half, device=s.device) / half)
    ang = 1000.0 * s[:, None] * freq[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PConv(nn.Module):
    """3x3 convolution, periodic in x (last axis), edge-replicated in y."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return self.conv(x)


class ResBlock(nn.Module):
    """Two convolutions with a FiLM of the conditioning (flow time + latent state) in between, plus the latent state's
    spatial map (if any) added at the same point."""

    def __init__(self, cin: int, cout: int, emb: int, zmap: int = 0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), PConv(cin, cout)
        self.film = nn.Linear(emb, 2 * cout)
        self.zproj = nn.Conv2d(zmap, cout, 1) if zmap > 0 else None
        self.n2, self.c2 = nn.GroupNorm(8, cout), PConv(cout, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, e: torch.Tensor, zm: torch.Tensor | None = None) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(e)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + g) + b
        if self.zproj is not None:
            h = h + self.zproj(F.interpolate(zm, size=h.shape[-2:], mode="bilinear", align_corners=False))
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class FlowDecoder(nn.Module):
    """Velocity v(x_s, s | x(t), z~) on fields [B, V, H, W] (field units), conditioned on a latent state z~ [B, K]."""

    def __init__(self, K: int, V: int = 2, ch=(64, 128, 192), emb: int = 256, sigma0: float = 1.0,
                 source: str = "noise", zmap: int = 0, zgrid: int = 16):
        super().__init__()
        assert source in ("noise", "launch"), source
        self.K, self.V, self.sigma0, self.source = K, V, float(sigma0), source
        self.zmap, self.zgrid = int(zmap), int(zgrid)
        # the latent state's spatial map: each latent channel learns WHERE in the field it acts (like a column of the
        # linear decoder, but onto a coarse feature grid rather than onto the field itself)
        self.z_grid = nn.Linear(K, self.zmap * self.zgrid ** 2) if self.zmap > 0 else None
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.z_mlp = nn.Sequential(nn.Linear(K, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.emb = emb
        self.inp = PConv((2 if source == "launch" else 1) * V, ch[0])  # [x_s, x(t)] or x_s alone
        self.down = nn.ModuleList()
        c = ch[0]
        for i, co in enumerate(ch):
            self.down.append(ResBlock(c, co, emb, self.zmap))
            c = co
        self.mid = nn.ModuleList([ResBlock(c, c, emb, self.zmap), ResBlock(c, c, emb, self.zmap)])
        self.up = nn.ModuleList()
        for co in reversed(ch):
            self.up.append(ResBlock(c + co, co, emb, self.zmap))                  # skip from the same resolution
            c = co
        self.out_norm = nn.GroupNorm(8, c)
        self.out = PConv(c, V)
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def velocity(self, x: torch.Tensor, s: torch.Tensor, x_launch: torch.Tensor | None, z: torch.Tensor) -> torch.Tensor:
        e = self.t_mlp(time_embedding(s, self.emb)) + self.z_mlp(z)
        zm = (self.z_grid(z).view(-1, self.zmap, self.zgrid, self.zgrid) if self.z_grid is not None else None)
        h = self.inp(torch.cat([x, x_launch], dim=1) if self.source == "launch" else x)
        skips = []
        for i, blk in enumerate(self.down):
            h = blk(h, e, zm)
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)                                 # 64 -> 32 -> 16
        for blk in self.mid:
            h = blk(h, e, zm)
        for i, blk in enumerate(self.up):
            if i > 0:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = blk(torch.cat([h, skips.pop()], dim=1), e, zm)
        return self.out(F.silu(self.out_norm(h)))

    def x0(self, x_launch: torch.Tensor | None, eps: torch.Tensor) -> torch.Tensor:
        """Source sample of the flow: pure noise, or the noised launch field."""
        return self.sigma0 * eps if self.source == "noise" else x_launch + self.sigma0 * eps

    def loss(self, x_launch: torch.Tensor | None, x_next: torch.Tensor, z: torch.Tensor,
             eps: torch.Tensor | None = None, s: torch.Tensor | None = None) -> torch.Tensor:
        """Flow-matching loss of a batch. Fixed eps / s give a reproducible score (model selection on held-out data)."""
        eps = torch.randn_like(x_next) if eps is None else eps
        s = torch.rand(x_next.shape[0], device=x_next.device) if s is None else s
        x0 = self.x0(x_launch, eps)
        xs = (1 - s)[:, None, None, None] * x0 + s[:, None, None, None] * x_next
        return ((self.velocity(xs, s, x_launch, z) - (x_next - x0)) ** 2).mean()

    @torch.no_grad()
    def sample(self, x_launch: torch.Tensor | None, z: torch.Tensor, steps: int = 16,
               eps: torch.Tensor | None = None, shape: tuple | None = None) -> torch.Tensor:
        """One member per row: latent draws z~ [B, K] (+ launch fields [B, V, H, W] for source "launch") -> fields
        [B, V, H, W] (Heun). With source "noise" the launch field is ignored; `shape` = (V, H, W) when it is None."""
        if eps is None:
            eps = torch.randn(z.shape[0], *(x_launch.shape[1:] if x_launch is not None else shape), device=z.device)
        x = self.x0(x_launch, eps)
        B, dt = x.shape[0], 1.0 / steps
        for i in range(steps):
            s0 = torch.full((B,), i * dt, device=x.device)
            v0 = self.velocity(x, s0, x_launch, z)
            v1 = self.velocity(x + dt * v0, s0 + dt, x_launch, z)
            x = x + 0.5 * dt * (v0 + v1)                               # Heun: 2nd order, as in the per-eye flow
        return x
