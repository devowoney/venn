"""Per-eye flow forecaster — a spread from ONE eye's own 1-D history (D-044, SOP 04 "Per-eye flow forecaster").

A separate forecaster beside the transformer emulator. It reads the last L values of a single eye and draws possible
futures of that eye: one sample = the whole next `leads` values, so every member is a coherent scenario and nothing is
rolled out in physical time. One network serves every eye; it is never told which eye or which family it reads, so a
narrow or a wide fan can only come from the history itself.

Generative model = conditional FLOW MATCHING. "Time" here is FLOW time τ ∈ [0,1], which carries Gaussian noise
(τ = 0) to a future (τ = 1); it has nothing to do with the physical time of the history.

    y      = z[t+1 .. t+A] − z[t]                         target: increments from the launch value (A = leads)
    y_τ    = (1 − τ) ε + τ y,   ε ~ N(0, I_A)            straight path from noise to the future
    loss   = ‖ v(y_τ, τ, h) − (y − ε) ‖²                 the network learns the velocity of that path
    sample : integrate dy/dτ = v from τ = 0 (noise) to 1  -> one future
    density: log p(y | h) = log N(ε(y)) − ∫ div v dτ      (the same ODE run backwards from y; change of variables)

Why each piece:
- increments from z_t as the target: the launch value is known exactly, so lead 1 starts where the eye is; the
  history tokens still carry absolute values (z) so the network can see how far the eye sits from its usual level
  (mean reversion of a bounded signal is part of what limits the spread).
- the history is cut into patches of 8 steps (32 tokens) and encoded ONCE per launch; the velocity of every member and
  every flow step re-reads it by cross-attention. Members are cheap: only the future decoder runs per step.
- the future is 16 tokens of 4 leads each: attention between future tokens is what makes the 64 leads one JOINT
  sample (a member that is high at lead 10 stays high at lead 11), instead of 64 independent marginals.
- τ enters every decoder layer: the same network must act as a coarse denoiser near τ = 0 and a fine one near τ = 1.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def tau_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features of the flow time τ [B] -> [B, dim] (frequencies 1 .. 1000, like diffusion timesteps)."""
    half = dim // 2
    freq = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
    ang = 1000.0 * tau[:, None] * freq[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class EyeFlow(nn.Module):
    """Velocity field v(y_τ, τ, history) of one eye's future, shared by all eyes.

    history : [B, L] values of ONE eye in its own z units, the last one = the launch value z_t
    y       : [B, A] increments of the future from z_t (A = leads)
    """

    def __init__(self, hist_len: int = 256, leads: int = 64, hist_patch: int = 8, lead_patch: int = 4,
                 d: int = 64, enc_layers: int = 3, dec_layers: int = 3, heads: int = 4, dropout: float = 0.1,
                 target_scale: str = "none"):
        super().__init__()
        assert hist_len % hist_patch == 0 and leads % lead_patch == 0
        assert target_scale in ("none", "lead", "history_lead"), target_scale
        self.L, self.A, self.hp, self.lp, self.d = hist_len, leads, hist_patch, lead_patch, d
        # TARGET SCALE (D-044 step 2): the flow starts from N(0, I); where the future is 20-50x smaller than that
        # (lead 1, cyclic eyes) it must shrink the noise that much and stopped short -> fans too wide (F-39).
        # The flow therefore runs on y / σ(h). g_a [A] is measured on the training windows (set_lead_scale).
        self.target_scale = target_scale
        self.register_buffer("lead_scale", torch.ones(leads))
        n_hist, n_fut = hist_len // hist_patch, leads // lead_patch
        # history token = one patch of 8 steps, each step read twice: absolute value z and the offset z − z_t
        self.hist_in = nn.Linear(2 * hist_patch, d)
        self.hist_pos = nn.Parameter(0.02 * torch.randn(n_hist, d))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True), enc_layers,
            enable_nested_tensor=False)
        # future token = the noisy increments y_τ of 4 consecutive leads
        self.fut_in = nn.Linear(lead_patch, d)
        self.fut_pos = nn.Parameter(0.02 * torch.randn(n_fut, d))
        self.tau_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.tau_to_layer = nn.ModuleList([nn.Linear(d, d) for _ in range(dec_layers)])
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
            for _ in range(dec_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, lead_patch)

    @staticmethod
    def history_step(history: torch.Tensor) -> torch.Tensor:
        """s_h [B, 1]: RMS one-step change of the window itself -- how much this eye has been moving lately.
        Read from the history only, so scaling by it tells the network nothing the history does not already hold.
        Floor 1e-4 (z units) for a window that does not move at all."""
        return torch.diff(history, dim=-1).pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)

    def scale(self, history: torch.Tensor) -> torch.Tensor:
        """σ(h) [B, A], the per-lead size the flow's target is divided by (ones for `none`)."""
        g = self.lead_scale[None, :].expand(history.shape[0], -1)
        return g * self.history_step(history) if self.target_scale == "history_lead" else g

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """[B, L] history -> [B, L/8, d] memory, computed once per launch and reused by every member and flow step."""
        B = history.shape[0]
        rel = history - history[:, -1:]                                # offset from the launch value
        x = torch.stack([history, rel], dim=-1).reshape(B, -1, 2 * self.hp)
        return self.encoder(self.hist_in(x) + self.hist_pos)

    def velocity(self, y: torch.Tensor, tau: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """dy/dτ at the point y [B, A], flow time τ [B], given the encoded history memory [B, n, d] -> [B, A]."""
        B = y.shape[0]
        x = self.fut_in(y.reshape(B, -1, self.lp)) + self.fut_pos
        temb = self.tau_mlp(tau_embedding(tau, self.d))                 # [B, d]
        for proj, layer in zip(self.tau_to_layer, self.decoder):
            # no causal mask: the future tokens see each other both ways -> the leads are drawn jointly
            x = layer(x + proj(temb)[:, None, :], memory)
        return self.out(self.out_norm(x)).reshape(B, self.A)

    def loss(self, history: torch.Tensor, y: torch.Tensor, eps: torch.Tensor | None = None,
             tau: torch.Tensor | None = None) -> torch.Tensor:
        """Flow-matching loss of a batch of (history, true increments). Fixed eps / tau give a reproducible score
        (used for model selection on the held-out slice); during training both are drawn fresh."""
        y = y / self.scale(history)                                     # the flow lives in scaled units
        eps = torch.randn_like(y) if eps is None else eps
        tau = torch.rand(y.shape[0], device=y.device) if tau is None else tau
        y_tau = (1 - tau[:, None]) * eps + tau[:, None] * y
        return ((self.velocity(y_tau, tau, self.encode(history)) - (y - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, history: torch.Tensor, members: int, steps: int = 16) -> torch.Tensor:
        """[B, L] histories -> [B, M, A] increments: M futures per launch, by Heun integration noise -> data."""
        B = history.shape[0]
        mem = self.encode(history).repeat_interleave(members, dim=0)   # history read once, shared by the members
        y = torch.randn(B * members, self.A, device=history.device)
        dt = 1.0 / steps
        for i in range(steps):
            t0 = torch.full((B * members,), i * dt, device=y.device)
            v0 = self.velocity(y, t0, mem)
            y1 = y + dt * v0
            v1 = self.velocity(y1, t0 + dt, mem)
            y = y + 0.5 * dt * (v0 + v1)                                # Heun: 2nd order, cheap and stable here
        return y.reshape(B, members, self.A) * self.scale(history)[:, None, :]   # back to z units

    def log_prob(self, history: torch.Tensor, y: torch.Tensor, steps: int = 16, probes: int = 4) -> torch.Tensor:
        """log p(y | history) in nats [B], for given futures y [B, A] (e.g. the truth).

        Change of variables along the flow: run the ODE BACKWARDS from y (τ = 1) to its noise ε (τ = 0) and
        accumulate the divergence of v on the way:  log p_1(y) = log N(ε; 0, I) − ∫_0^1 div v dτ.
        div v = trace of the A x A Jacobian, estimated by Hutchinson (E[uᵀ J u], u Rademacher) with `probes` fixed
        probes per launch -- unbiased, and cheap next to A exact Jacobian columns. Euler-Heun backwards, `steps` steps.
        """
        with torch.no_grad():
            mem = self.encode(history)
            sig = self.scale(history)
        # the flow's density is over y / σ; in z units the volume change adds − Σ_a log σ_a
        log_det = sig.log().sum(-1)
        y = y / sig
        B, A = y.shape
        g = torch.Generator(device=y.device).manual_seed(0)
        u = (torch.randint(0, 2, (probes, B, A), generator=g, device=y.device) * 2 - 1).float()

        def v_and_div(yy: torch.Tensor, tt: torch.Tensor):
            yy = yy.detach().requires_grad_(True)
            with torch.enable_grad():
                v = self.velocity(yy, tt, mem)
                div = 0.0
                for p in range(probes):
                    (jv,) = torch.autograd.grad(v, yy, u[p], retain_graph=p < probes - 1)
                    div = div + (jv * u[p]).sum(-1)
            return v.detach(), div.detach() / probes

        dt = 1.0 / steps
        x, int_div = y.clone(), torch.zeros(B, device=y.device)
        for i in range(steps):                                          # τ: 1 -> 0
            t1 = torch.full((B,), 1.0 - i * dt, device=y.device)
            v1, d1 = v_and_div(x, t1)
            x0 = x - dt * v1
            v0, d0 = v_and_div(x0, t1 - dt)
            x = x - 0.5 * dt * (v1 + v0)
            int_div = int_div + 0.5 * dt * (d1 + d0)
        log_p0 = -0.5 * (x ** 2).sum(-1) - 0.5 * A * math.log(2 * math.pi)
        return log_p0 - int_div - log_det
