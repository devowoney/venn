"""Synthetic SSH/SST testbed generator (Milestone 0).

Implements SOP ./architecture/01_synthetic_generator.md (D-009 as reframed by D-013).

Produces a spatially heterogeneous, multi-scale field so that *which region/scale a mask sees*
determines the temporal character it reads. Latent modes are returned as a HIDDEN answer key for
post-hoc analysis only -- they are never fed to the model.

Output: field float32 [T=2000, V=2, H=64, W=64] (var0=SSH-like, var1=SST-like), per-var
standardized to ~0 mean / unit variance over (t, x, y); plus a `truth` dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import List, Tuple

import numpy as np


@dataclass
class GenConfig:
    """All knobs for the generator. Everything is reproducible from (seed, GenConfig)."""
    T: int = 2000                 # time steps
    H: int = 64                   # latitude cells
    W: int = 64                   # longitude cells
    n_stationary: int = 1         # LARGE scale (equatorial band), slow OU
    n_cyclic: int = 3             # MEDIUM scale blobs, sinusoids
    n_chaotic: int = 6            # SMALL scale energetic patches, Lorenz
    cyclic_periods: Tuple[float, ...] = (60.0, 140.0, 300.0)
    ou_tau: float = 200.0         # OU decorrelation time (large -> slow drift)
    lorenz_dt: float = 0.01       # Lorenz integration step
    lorenz_subsample: int = 5     # keep every Nth integrated step
    sst_lag: int = 3              # SST lags SSH on shared (stationary+cyclic) modes
    sst_shared_w: float = 0.8     # SST weight on shared modes
    sst_chaotic_w: float = 0.5    # SST (private-ish) weight on small chaotic modes
    sst_blur_sigma: float = 2.0   # spatial smoothing of shared patterns for SST
    obs_noise: float = 0.05       # additive obs noise, in units of (standardized) field std


# ----------------------------------------------------------------------------- spatial patterns
def _grid(H: int, W: int):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # [H,W] each
    return yy.astype(np.float64), xx.astype(np.float64)


def _gauss_bump(H: int, W: int, cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
    """Anisotropic Gaussian bump, normalized to max=1. Shape [H,W]."""
    yy, xx = _grid(H, W)
    phi = np.exp(-(((yy - cy) ** 2) / (2 * sy ** 2) + ((xx - cx) ** 2) / (2 * sx ** 2)))
    return phi / phi.max()


def _equatorial_band(H: int, W: int, cy: float, sy: float) -> np.ndarray:
    """Broad horizontal stripe (large-scale, long-range in x). Shape [H,W], max=1."""
    yy, _ = _grid(H, W)
    phi = np.exp(-((yy - cy) ** 2) / (2 * sy ** 2))
    return phi / phi.max()


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via 1D convolution along each axis (no scipy dependency)."""
    if sigma <= 0:
        return img
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, img)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)
    return out


# ---------------------------------------------------------------------------- temporal generators
def _standardize(a: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance along time (guards zero variance)."""
    return (a - a.mean()) / (a.std() + 1e-8)


def _ou_series(rng: np.random.Generator, T: int, tau: float) -> np.ndarray:
    """Ornstein-Uhlenbeck (mean-reverting, slow). Returns standardized [T]."""
    a = np.zeros(T)
    decay = 1.0 / tau
    for t in range(1, T):
        a[t] = a[t - 1] * (1 - decay) + np.sqrt(2 * decay) * rng.standard_normal()
    return _standardize(a)


def _sinusoid_series(rng: np.random.Generator, T: int, period: float) -> np.ndarray:
    """Cyclic sinusoid + small noise. Returns standardized [T]."""
    t = np.arange(T)
    a = np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)) + 0.05 * rng.standard_normal(T)
    return _standardize(a)


def _lorenz_series(rng: np.random.Generator, T: int, dt: float, subsample: int) -> np.ndarray:
    """One Lorenz system integrated with RK4 -> 3 chaotic coords. Returns standardized [3, T]."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def deriv(state):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    n_steps = T * subsample
    s = np.array([0.0, 1.0, 1.05]) + 0.01 * rng.standard_normal(3)  # near-attractor init + jitter
    traj = np.empty((n_steps, 3))
    for i in range(n_steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = s
    coords = traj[::subsample][:T].T  # [3, T]
    return np.stack([_standardize(c) for c in coords])  # [3, T]


# ---------------------------------------------------------------------------------- assembly
def generate_field(cfg: GenConfig = GenConfig(), seed: int = 0):
    """Generate the synthetic field + hidden truth.

    Returns
    -------
    field : np.ndarray float32 [T, 2, H, W]   -- var0 SSH-like, var1 SST-like
    truth : dict                              -- hidden answer key (modes, config, seed)
    """
    rng = np.random.default_rng(seed)
    T, H, W = cfg.T, cfg.H, cfg.W

    modes: List[dict] = []  # each: {scale, family, phi[H,W], amp[T]}

    # --- stationary (LARGE): equatorial band, slow OU ---------------------------------------
    for _ in range(cfg.n_stationary):
        phi = _equatorial_band(H, W, cy=H / 2 + rng.uniform(-4, 4), sy=rng.uniform(9, 13))
        amp = _ou_series(rng, T, cfg.ou_tau)
        modes.append(dict(scale="large", family="stationary", phi=phi, amp=amp))

    # --- cyclic (MEDIUM): regional blobs, sinusoids ------------------------------------------
    for i in range(cfg.n_cyclic):
        cy, cx = rng.uniform(12, H - 12), rng.uniform(12, W - 12)
        phi = _gauss_bump(H, W, cy, cx, sy=rng.uniform(8, 12), sx=rng.uniform(8, 12))
        period = cfg.cyclic_periods[i % len(cfg.cyclic_periods)]
        amp = _sinusoid_series(rng, T, period)
        modes.append(dict(scale="medium", family="cyclic", phi=phi, amp=amp))

    # --- chaotic (SMALL): tight energetic patches, Lorenz coords ------------------------------
    n_lorenz = int(np.ceil(cfg.n_chaotic / 3))
    chaotic_amps = np.concatenate(
        [_lorenz_series(rng, T, cfg.lorenz_dt, cfg.lorenz_subsample) for _ in range(n_lorenz)]
    )  # [3*n_lorenz, T]
    for i in range(cfg.n_chaotic):
        cy, cx = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        phi = _gauss_bump(H, W, cy, cx, sy=rng.uniform(2, 4), sx=rng.uniform(2, 4))
        modes.append(dict(scale="small", family="chaotic", phi=phi, amp=chaotic_amps[i]))

    K = len(modes)

    # --- per-var weights & lags (SST shares large+medium lagged; small mostly private) --------
    A = np.zeros((2, T, K))       # amplitudes per var, lagged  [V,T,K]
    Phi = np.zeros((2, K, H, W))  # spatial patterns per var    [V,K,H,W]
    for k, m in enumerate(modes):
        shared = m["family"] in ("stationary", "cyclic")
        # var0 = SSH: full weight, no lag, sharp pattern
        A[0, :, k] = m["amp"]
        Phi[0, k] = m["phi"]
        # var1 = SST: lagged + smoothed on shared modes, private-weight on chaotic
        if shared:
            w, lag = cfg.sst_shared_w, cfg.sst_lag
        else:
            w, lag = cfg.sst_chaotic_w, 0
        idx = np.clip(np.arange(T) - lag, 0, T - 1)  # hold initial value for t < lag
        A[1, :, k] = w * m["amp"][idx]
        Phi[1, k] = _gaussian_blur(m["phi"], cfg.sst_blur_sigma) if shared else m["phi"]

    # --- superpose, add obs noise, standardize per var ---------------------------------------
    field = np.empty((T, 2, H, W), dtype=np.float64)
    for v in range(2):
        fv = np.einsum("tk,khw->thw", A[v], Phi[v])          # [T,H,W]
        fv = (fv - fv.mean()) / (fv.std() + 1e-8)            # standardize var
        fv = fv + cfg.obs_noise * rng.standard_normal(fv.shape)
        field[:, v] = fv

    field = field.astype(np.float32)
    assert field.shape == (T, 2, H, W), field.shape
    assert np.isfinite(field).all(), "non-finite values in field"

    truth = dict(
        modes=[dict(scale=m["scale"], family=m["family"],
                    phi=m["phi"].astype(np.float32), amp=m["amp"].astype(np.float32))
               for m in modes],
        K_modes=K, seed=seed, config=cfg.__dict__.copy(),
    )
    return field, truth


if __name__ == "__main__":
    f, t = generate_field()
    print("field", f.shape, f.dtype, "| modes:", t["K_modes"],
          "| families:", [m["family"] for m in t["modes"]])
