"""Spread from Hessian (B-norm) singular vectors of the latent emulator — D-043, SOP 04, memory/hsv_spread_design.md.

The history forecaster was trained to say every lead at once from the history. A MEMBER of the spread ensemble is
instead a ROLLOUT: start from a (perturbed) last state z_t, take the forecaster's lead-1 answer, append it to the
history as if it had been observed, and ask again — 64 times. Why a rollout: the direct long-lead heads were trained
toward the conditional mean, which forgets the start for chaos, so a start error would SHRINK there; composing the
accurate lead-1 step lets a start error grow the way the dynamics make it grow.

Cost: re-reading an ~8000-step history at every rolled step would be far too slow. The causal forecaster never lets
the past depend on the future, so one shared pass over the record gives every layer's keys/values for all real states;
a rolled step is then ONE new token attending to the real past (positions < t) plus the launch token and the steps
rolled so far. Exactness: feeding back the TRUE states (teacher forcing) reproduces the full causal pass.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

from src.models.history_forecaster import HistoryForecaster, rotary


def _rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary position for single tokens at their own positions: x [N,h,dh], pos [N] (same rule as `rotary`)."""
    dh = x.shape[-1]
    freq = torch.exp(-math.log(8000.0) * torch.arange(0, dh, 2, device=x.device, dtype=x.dtype) / dh)
    ang = pos[:, None, None].to(x.dtype) * freq                         # [N,1,dh/2]
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class CachedRollout:
    """Rollouts of a frozen HistoryForecaster from many launches at once, sharing one pass over the real record.

    z [T,K] = the whole real latent record (model units). Launch at time t: the history is z_0..z_{t-1} (real) and the
    start state x_0 (z_t, possibly perturbed); step j feeds the lead-1 forecast back as the state at t+j.
    """

    @torch.no_grad()
    def __init__(self, model: HistoryForecaster, z: torch.Tensor):
        self.m = model.eval()
        # frozen dynamics: no backward graph, so forward-mode JVPs through 64 steps stay small in memory
        self.m.requires_grad_(False)
        self.z = z
        T, K = z.shape
        self.K, self.A = K, model.A
        dz = torch.cat([torch.zeros_like(z[:1]), z[1:] - z[:-1]], 0)
        x = model.inp(torch.cat([z, dz], -1))[None]                     # [1,T,d]
        pos = torch.arange(T, device=z.device)
        self.kc, self.vc = [], []                                       # per layer: [h,T,dh] (keys already rotated)
        for blk in model.blocks:
            B_, T_, d = x.shape
            q, k, v = blk.qkv(blk.n1(x)).view(B_, T_, 3, blk.h, d // blk.h).permute(2, 0, 3, 1, 4)
            self.kc.append(rotary(k, pos)[0])
            self.vc.append(v[0])
            x = blk(x, pos)                                             # the real record's next-layer input

    def _step(self, z_new: torch.Tensor, z_prev: torch.Tensor, pos: torch.Tensor, t0: torch.Tensor, new_kv: list):
        """One new token per row: state z_new [N,K] at absolute position pos [N] of a launch at t0 [N].

        Attends to the real past (cache positions < t0) and to this row's own new tokens (new_kv, appended here).
        Returns the forecaster's head output [N,A,K] read at this token.
        """
        m = self.m
        x = m.inp(torch.cat([z_new, z_new - z_prev], -1))              # [N,d]
        N, d = x.shape
        T = self.z.shape[0]
        past = torch.arange(T, device=x.device)[None, :] < t0[:, None]  # [N,T] which real states this launch has seen
        for li, blk in enumerate(m.blocks):
            h = blk.h
            q, k, v = blk.qkv(blk.n1(x)).view(N, 3, h, d // h).unbind(1)   # each [N,h,dh]
            q, k = _rotate(q, pos), _rotate(k, pos)
            new_kv[li][0].append(k)
            new_kv[li][1].append(v)
            kn, vn = torch.stack(new_kv[li][0], 2), torch.stack(new_kv[li][1], 2)   # [N,h,j+1,dh]
            scale = 1.0 / math.sqrt(d // h)
            s_past = torch.einsum("nhd,htd->nht", q, self.kc[li]) * scale
            s_past = s_past.masked_fill(~past[:, None, :], float("-inf"))
            s_new = torch.einsum("nhd,nhjd->nhj", q, kn) * scale
            p = torch.softmax(torch.cat([s_past, s_new], -1), -1)
            a = torch.einsum("nht,htd->nhd", p[..., :T], self.vc[li]) + torch.einsum("nhj,nhjd->nhd", p[..., T:], vn)
            x = x + blk.proj(a.reshape(N, d))
            x = x + blk.mlp(blk.n2(x))
        return m.head(m.norm(x)).view(N, self.A, self.K)

    def rollout(self, t0: torch.Tensor, x0: torch.Tensor, steps: int, teacher: torch.Tensor | None = None,
                chunk: int = 1):
        """From launches t0 [N] and start states x0 [N,K] -> (trajectory [N,steps,K] for leads 1..steps,
        direct forecast [N,A,K] read at the start token).

        chunk = how many direct leads are taken before the forecaster is asked again: 1 = a pure rollout (the lead-1
        answer is fed back every step); c > 1 = CHUNKED rollout: leads 1..c of the current answer are appended to the
        history as if observed, and the next answer is read at the last of them (fewer feedbacks, less compounding).
        teacher [N,steps,K] (chunk 1 only): feed these states back instead of the forecasts (exactness check).
        """
        assert teacher is None or chunk == 1
        new_kv = [([], []) for _ in self.m.blocks]
        z_prev = self.z[(t0 - 1).clamp_min(0)]
        out = self._step(x0, z_prev, t0, t0, new_kv)
        direct = x0[:, None] + out                                      # residual heads: persistence + increment
        traj, cur = [], x0
        while len(traj) < steps:
            c = min(chunk, steps - len(traj))
            block = cur[:, None] + out[:, :c]                           # the next c states, from the current answer
            if teacher is not None:
                block = teacher[:, len(traj):len(traj) + 1]
                traj.append(cur + out[:, 0])
            else:
                traj.extend(block.unbind(1))
            if len(traj) == steps:
                break
            prev = cur
            for i in range(c):                                          # append them; only the last answer is used
                out = self._step(block[:, i], prev, t0 + len(traj) - c + i + 1, t0, new_kv)
                prev = block[:, i]
            cur = block[:, -1]
        return torch.stack(traj, 1), direct


# ---------------------------------------------------------------------------------------------------------------------
# B: the prior start-error covariance. Defined on the RAW field (a rule, never an N x N matrix) and used through the
# linear observer, B_z = E B_x E^T. Only what the eyes see of a raw error can change the forecast.
# ---------------------------------------------------------------------------------------------------------------------

def field_change_stats(field_fit: np.ndarray, max_lag: int = 12):
    """Smooth part of the field's one-step change (design Q2-amend, Hollingsworth–Lönnberg).

    The change x_{t+1} − x_t mixes the ocean's real movement (smooth in space) with the generator's cell-by-cell noise
    (no spatial reach). Their covariance-by-distance tells them apart: the noise exists only at distance 0, the movement
    fades over ~ell cells. Fit a·exp(−r²/2ℓ²) on distances r >= 1 (shared ℓ, own a per variable); the gap at r = 0 is
    the noise ("nugget"), removed from each cell's variance.
    field_fit [T,V,H,W] (training slice) -> sigma [V,H,W] (smooth-part std per cell), ell (cells), nugget [V], a [V].
    """
    d = np.diff(field_fit.astype(np.float32), axis=0)
    d -= d.mean(0)
    var_cell = (d ** 2).mean(0)                                         # [V,H,W]
    V = d.shape[1]
    lags = np.arange(1, max_lag + 1)
    c = np.zeros((V, len(lags)))
    for v in range(V):
        dv = d[:, v]
        c0 = float((dv ** 2).mean())
        for i, r in enumerate(lags):
            cx = float((dv * np.roll(dv, r, axis=2)).mean())            # along x (periodic)
            cy = float((dv[:, r:] * dv[:, :-r]).mean())                 # along y (no wrap)
            c[v, i] = 0.5 * (cx + cy) / c0
    # log c = log a_v − r² / (2ℓ²): shared slope, own intercept, on lags where the correlation is still clear
    rows, rhs = [], []
    for v in range(V):
        for i, r in enumerate(lags):
            if c[v, i] > 0.05:
                row = np.zeros(V + 1)
                row[v], row[V] = 1.0, -float(r) ** 2
                rows.append(row), rhs.append(np.log(c[v, i]))
    sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    a = np.exp(sol[:V])
    ell = float(np.sqrt(1.0 / (2.0 * sol[V])))
    c0 = var_cell.reshape(V, -1).mean(1)
    nugget = c0 * (1.0 - a)                                             # white part of the change, per variable
    sigma = np.sqrt(np.clip(var_cell - nugget[:, None, None], 0.0, None))
    return sigma, ell, nugget, a


def apply_bx(m: torch.Tensor, sigma: torch.Tensor, ell: float) -> torch.Tensor:
    """B_x m = Σ C Σ m for raw-field maps m [n,V,H,W]: weight by the cell's error size, spread over ~ell cells
    (Gaussian correlation, peak 1; periodic in x, open in y; no SSH–SST cross term), weight again."""
    n, V, H, W = m.shape
    R = int(np.ceil(3 * ell))
    r = torch.arange(-R, R + 1, device=m.device, dtype=m.dtype)
    g = torch.exp(-r ** 2 / (2 * ell ** 2))
    kern = (g[:, None] * g[None, :])[None, None]                        # [1,1,2R+1,2R+1]
    x = (sigma[None] * m).reshape(n * V, 1, H, W)
    x = F.pad(x, (R, R, 0, 0), mode="circular")
    x = F.pad(x, (0, 0, R, R))
    return sigma[None] * F.conv2d(x, kern).reshape(n, V, H, W)


def latent_b(masks: np.ndarray, sd: np.ndarray, sigma: np.ndarray, ell: float, device: str = "cuda") -> torch.Tensor:
    """B_z = E_z B_x E_zᵀ [K,K]: how wrong the eyes' readings would be (and together), in model units.
    E_z rows = mask / sd (the forecaster reads z = (S − mu) / sd with S = <mask, field>)."""
    E = torch.tensor(masks / sd[:, None, None, None], dtype=torch.float32, device=device)
    BE = apply_bx(E, torch.tensor(sigma, dtype=torch.float32, device=device), ell)
    Bz = torch.einsum("kvhw,jvhw->kj", E, BE)
    return 0.5 * (Bz + Bz.T)


def sqrt_psd(B: torch.Tensor, rel_floor: float = 1e-6) -> torch.Tensor:
    """Symmetric square root of a covariance (tiny / negative eigenvalues floored)."""
    lam, U = torch.linalg.eigh(B)
    lam = lam.clamp_min(rel_floor * lam.max())
    return (U * lam.sqrt()) @ U.T


# ---------------------------------------------------------------------------------------------------------------------
# Singular vectors and members
# ---------------------------------------------------------------------------------------------------------------------

def tangent_linear(cr: CachedRollout, t0: torch.Tensor, lead: int, chunk: int) -> torch.Tensor:
    """L [N,K,K]: d(member state at `lead`) / d(start state z_t), exact, by K forward-mode JVPs (one per start channel)."""
    x0 = cr.z[t0]
    N, K = x0.shape
    cols = []
    for k in range(K):
        e = torch.zeros_like(x0)
        e[:, k] = 1.0
        _, col = jvp(lambda x: cr.rollout(t0, x, lead, chunk=chunk)[0][:, -1], (x0,), (e,))
        cols.append(col)
    return torch.stack(cols, -1)


def singular_vectors(L: torch.Tensor, Bh: torch.Tensor, n: int):
    """B-norm singular vectors: likely start errors (B_z) that grow most by the optimization lead (latent norm).

    SVD of L B^{1/2} = U Σ Vᵀ; start patterns v_i = B^{1/2} V_i (v_iᵀ B⁻¹ v_i = 1: each is a '1-sigma' plausible start
    error), growth σ_i, evolved patterns U_i. Returns v [N,K,n], sigma [N,n], U [N,K,n].
    """
    U, S, Vh = torch.linalg.svd(L @ Bh)
    v = Bh @ Vh[:, :n].transpose(1, 2)
    return v, S[:, :n], U[:, :, :n]


def hsv_members(cr: CachedRollout, t0: torch.Tensor, v: torch.Tensor, alpha: float, xi: torch.Tensor,
                steps: int, chunk: int) -> torch.Tensor:
    """Members [N,M,steps,K]: rollouts from z_t ± α Σ_i ξ_{m,i} v_i. xi [M/2, n] (shared by all launches: the same
    random mixing everywhere, so α can be compared on common random numbers). ± pairs keep the members centred."""
    d = alpha * torch.einsum("nki,mi->nmk", v, xi)                      # [N,M/2,K]
    d = torch.cat([d, -d], 1)
    N, M, K = d.shape
    x0 = (cr.z[t0][:, None] + d).reshape(N * M, K)
    traj, _ = cr.rollout(t0.repeat_interleave(M), x0, steps, chunk=chunk)
    return traj.view(N, M, steps, K)


def measured_eye_scale(Bz: torch.Tensor, z_fit: np.ndarray, masks: np.ndarray, sd: np.ndarray,
                       nugget: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Rescale B_z so each eye's start-error SIZE is what that eye really shows (design B-fix (a), user 2026-11-10).

    The distance rule says neighbouring cells err alike, so it cannot know that layer 2 shaped some masks (the
    stationary children) so that the movement inside them cancels: it gives them a large start error while their real
    smooth change is ~0 (all they show from step to step is noise). Measured size per eye = variance of its own
    1-step change minus the noise seen through its mask; the rule keeps the correlations between eyes.
    Returns (D B_z D, measured smooth std per eye).
    """
    dz = np.diff(z_fit, axis=0)
    E = masks.astype(np.float64) / sd[:, None, None, None]
    noise = np.einsum("kvhw,kvhw,v->k", E, E, nugget)                   # white noise variance seen by each eye
    meas = np.clip(dz.var(0) - noise, 0.0, None)
    rule = Bz.diagonal().double().cpu().numpy()
    D = torch.tensor(np.sqrt(meas / np.maximum(rule, 1e-12)), dtype=Bz.dtype, device=Bz.device)
    return D[:, None] * Bz * D[None, :], np.sqrt(meas)
