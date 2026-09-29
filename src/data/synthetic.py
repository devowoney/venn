"""Synthetic SSH/SST testbed generator (Milestone 0).

Implements SOP ./memory/sop/01_synthetic_generator.md (D-009 as reframed by D-013).

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
    # --- rev2 (D-021). Every default below REPRODUCES THE OLD FIELD; the new values live in
    # config/config.yaml, so runs recorded before 2026-08-27 regenerate exactly as they were.
    signed_patterns: bool = False   # dipole + low-wavenumber wave phi_k, and a zero-mean band,
                                    # so cell-pair covariance can be NEGATIVE (fixes F-4)
    periodic_x: bool = False        # wrap the x axis (longitude) when building patterns
    place_full_domain: bool = False # draw mode centres over the whole grid, not inside a margin
    wave_kmax: int = 3              # largest wavenumber for the wave patterns
    wave_env_sigma: float = 0.0     # >0: window each wave into a regional packet of this width.
                                    # 0 = domain-filling wave, which swamps the field's variance
    sst_tau: float = 0.0            # >0: SST is an AR1 low-pass response to the SSH forcing with
                                    # this time constant, replacing the fixed `sst_lag` shift
    n_private_ssh: int = 0          # modes visible ONLY in SSH (taken from the cyclic+chaotic set)
    n_private_sst: int = 0          # modes visible ONLY in SST
    # --- rev3 (D-025, user 2026-09-01): STATIONARY MEANS CONSTANT ------------------------------
    # The old "stationary" mode was an OU drift with tau=200 -- and `_ou_series` ends in
    # `_standardize`, so it carried UNIT VARIANCE, exactly as much temporal energy as the sinusoids
    # and the Lorenz modes. It was a slow wanderer, never a stationary signal. The user's
    # definition, from the original design discussion: stationary = a CONSTANT signal, flat in time.
    # With this on, mode 0 is a static spatial pattern with amp[t] = stationary_amp and ZERO
    # temporal variance; it lives entirely in the field's TIME MEAN. A zero-mean `phi` plus the
    # field's global (scalar) mean removal means the pattern survives standardization intact.
    # Default False = the old OU behaviour, so pre-2026-09-01 runs still regenerate exactly.
    stationary_constant: bool = False
    stationary_amp: float = 1.0     # the constant level, in units of the other modes' unit std


# ----------------------------------------------------------------------------- spatial patterns
def _grid(H: int, W: int):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")  # [H,W] each
    return yy.astype(np.float64), xx.astype(np.float64)


def _gauss_bump(H: int, W: int, cy: float, cx: float, sy: float, sx: float,
                periodic_x: bool = False) -> np.ndarray:
    """Anisotropic Gaussian bump, normalized to max=1. Shape [H,W].

    With `periodic_x` the x distance wraps around the domain (longitude), so a bump centred near
    the edge continues on the other side instead of leaving a dead border (D-021).
    """
    yy, xx = _grid(H, W)
    dx = xx - cx
    if periodic_x:
        dx = (dx + W / 2.0) % W - W / 2.0
    phi = np.exp(-((yy - cy) ** 2 / (2 * sy ** 2) + dx ** 2 / (2 * sx ** 2)))
    return phi / phi.max()


def _equatorial_band(H: int, W: int, cy: float, sy: float,
                     zero_mean: bool = False) -> np.ndarray:
    """Broad horizontal stripe (large-scale, long-range in x). Shape [H,W], max|phi|=1.

    `zero_mean` removes the spatial mean, giving a positive core with a negative surround — a
    mass-conserving basin mode, and the large-scale carrier of SIGN structure (D-021).
    """
    yy, _ = _grid(H, W)
    phi = np.exp(-((yy - cy) ** 2) / (2 * sy ** 2))
    if zero_mean:
        phi = phi - phi.mean()
    return phi / np.abs(phi).max()


def _dipole(H: int, W: int, cy: float, cx: float, sy: float, sx: float, sep: float,
            angle: float, periodic_x: bool = False) -> np.ndarray:
    """Two opposite-signed lobes separated by `sep` along `angle`. Shape [H,W], max|phi|=1.

    The ocean analogue is a teleconnection dipole / eddy pair: the two lobes are ANTI-correlated,
    which is what lets a non-negative selection mask decorrelate two channels (F-4 / D-021).
    """
    dy, dx = sep * np.sin(angle) / 2.0, sep * np.cos(angle) / 2.0
    phi = (_gauss_bump(H, W, cy + dy, cx + dx, sy, sx, periodic_x)
           - _gauss_bump(H, W, cy - dy, cx - dx, sy, sx, periodic_x))
    return phi / np.abs(phi).max()


def _wave(H: int, W: int, kx: int, ky: int, phase: float, env_sigma: float = 0.0,
          cy: float = 0.0, cx: float = 0.0, periodic_x: bool = False) -> np.ndarray:
    """Low-wavenumber wave, optionally windowed into a regional PACKET. [H,W], max|phi|=1.

    Sign-varying, which is the point (D-021). `env_sigma > 0` multiplies it by a broad Gaussian
    envelope centred at (cy,cx), making it a Rossby-wave-packet-like regional structure instead of
    a domain-filling plaid. Without the envelope the wave carries |phi|=1 over all 4096 cells and
    completely dominates the field's variance -- visible immediately in field_snapshots.png.
    """
    yy, xx = _grid(H, W)
    dx = xx - cx
    if periodic_x:
        dx = (dx + W / 2.0) % W - W / 2.0
    phi = np.cos(2 * np.pi * (kx * dx / W + ky * (yy - cy) / H) + phase)
    if env_sigma > 0:
        phi = phi * _gauss_bump(H, W, cy, cx, env_sigma, env_sigma, periodic_x)
    return phi / np.abs(phi).max()


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


def _ar1_response(a: np.ndarray, tau: float) -> np.ndarray:
    """First-order low-pass response to forcing `a`, time constant `tau`. Standardized [T].

    `y[t] = rho*y[t-1] + (1-rho)*a[t]`, `rho = exp(-1/tau)`. This is the physical SST story (SST
    integrates surface flux), and unlike a fixed index shift it damps amplitude AND shifts phase,
    which is what actually decorrelates SST from SSH (D-021).
    """
    rho = float(np.exp(-1.0 / max(tau, 1e-6)))
    y = np.zeros_like(a)
    y[0] = a[0]
    for t in range(1, len(a)):
        y[t] = rho * y[t - 1] + (1.0 - rho) * a[t]
    return _standardize(y)


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

    px = cfg.periodic_x
    # placement margins: rev2 draws centres over the FULL domain so no border ring is left dead
    my, mx = (0.0, 0.0) if cfg.place_full_domain else (12.0, 12.0)
    my_s, mx_s = (0.0, 0.0) if cfg.place_full_domain else (6.0, 6.0)

    # --- stationary (LARGE): equatorial band ---------------------------------------------------
    # rev3: a CONSTANT amplitude (the user's definition of stationary). The old OU drift is kept
    # behind the flag and is now understood as a `cyclic`-class slow wanderer, not a stationary mode.
    for _ in range(cfg.n_stationary):
        phi = _equatorial_band(H, W, cy=H / 2 + rng.uniform(-4, 4), sy=rng.uniform(9, 13),
                               zero_mean=cfg.signed_patterns)
        if cfg.stationary_constant:
            amp = np.full(T, float(cfg.stationary_amp))     # flat: var = 0, NOT standardized
        else:
            amp = _ou_series(rng, T, cfg.ou_tau)            # legacy drift (unit variance)
        modes.append(dict(scale="large", family="stationary", phi=phi, amp=amp))

    # --- cyclic (MEDIUM): sign-varying dipoles / basin waves (rev2), or blobs (legacy) --------
    for i in range(cfg.n_cyclic):
        cy, cx = rng.uniform(my, H - my), rng.uniform(mx, W - mx)
        sy, sx = rng.uniform(8, 12), rng.uniform(8, 12)
        if not cfg.signed_patterns:
            phi = _gauss_bump(H, W, cy, cx, sy, sx, px)
        elif i % 2 == 0:                                   # dipole: two anti-correlated lobes
            phi = _dipole(H, W, cy, cx, sy * 0.7, sx * 0.7,
                          sep=rng.uniform(18, 28), angle=rng.uniform(0, np.pi), periodic_x=px)
        else:                                              # regional wave packet
            phi = _wave(H, W, kx=int(rng.integers(1, cfg.wave_kmax + 1)),
                        ky=int(rng.integers(0, cfg.wave_kmax + 1)),
                        phase=rng.uniform(0, 2 * np.pi), env_sigma=cfg.wave_env_sigma,
                        cy=cy, cx=cx, periodic_x=px)
        period = cfg.cyclic_periods[i % len(cfg.cyclic_periods)]
        amp = _sinusoid_series(rng, T, period)
        modes.append(dict(scale="medium", family="cyclic", phi=phi, amp=amp))

    # --- chaotic (SMALL): tight energetic patches, Lorenz coords ------------------------------
    # Kept as MONOPOLES on purpose: a small eddy is physically a monopole; the sign structure
    # belongs to the large/medium scales.
    n_lorenz = int(np.ceil(cfg.n_chaotic / 3))
    chaotic_amps = np.concatenate(
        [_lorenz_series(rng, T, cfg.lorenz_dt, cfg.lorenz_subsample) for _ in range(n_lorenz)]
    )  # [3*n_lorenz, T]
    for i in range(cfg.n_chaotic):
        cy, cx = rng.uniform(my_s, H - my_s), rng.uniform(mx_s, W - mx_s)
        phi = _gauss_bump(H, W, cy, cx, sy=rng.uniform(2, 4), sx=rng.uniform(2, 4), periodic_x=px)
        modes.append(dict(scale="small", family="chaotic", phi=phi, amp=chaotic_amps[i]))

    K = len(modes)

    # --- private modes (rev2): a few modes appear in ONE variable only, so SSH and SST no longer
    # see the same 10 signals. Drawn from the cyclic+chaotic set, never the single large mode.
    only = ["both"] * K
    if cfg.n_private_ssh or cfg.n_private_sst:
        pool = rng.permutation(np.arange(cfg.n_stationary, K))
        n0, n1 = int(cfg.n_private_ssh), int(cfg.n_private_sst)
        for j in pool[:n0]:
            only[j] = "ssh"
        for j in pool[n0:n0 + n1]:
            only[j] = "sst"

    # --- per-var weights & response (SST = AR1 low-pass of the SSH forcing in rev2) -----------
    A = np.zeros((2, T, K))       # amplitudes per var  [V,T,K]
    Phi = np.zeros((2, K, H, W))  # spatial patterns per var    [V,K,H,W]
    for k, m in enumerate(modes):
        shared = m["family"] in ("stationary", "cyclic")
        # var0 = SSH: full weight, no lag, sharp pattern
        A[0, :, k] = m["amp"] if only[k] != "sst" else 0.0
        Phi[0, k] = m["phi"]
        # var1 = SST: damped + phase-shifted response on shared modes
        w = cfg.sst_shared_w if shared else cfg.sst_chaotic_w
        if np.std(m["amp"]) < 1e-12:
            # A CONSTANT mode has no anomaly to lag or damp: the AR1 steady state IS that constant,
            # so SST sees the same static offset. Going through `_ar1_response` would be wrong twice
            # over -- it would add a startup transient, and its closing `_standardize` divides a
            # zero-variance series by ~0, wiping the mode out of SST entirely (rev3, D-025).
            a1 = m["amp"]
        elif cfg.sst_tau > 0:                              # rev2: first-order response
            a1 = _ar1_response(m["amp"], cfg.sst_tau)
        else:                                              # legacy: fixed index shift
            lag = cfg.sst_lag if shared else 0
            a1 = m["amp"][np.clip(np.arange(T) - lag, 0, T - 1)]
        A[1, :, k] = (w * a1) if only[k] != "ssh" else 0.0
        Phi[1, k] = _gaussian_blur(m["phi"], cfg.sst_blur_sigma) if shared else m["phi"]
        m["only"] = only[k]                                # record in the hidden answer key

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
        modes=[dict(scale=m["scale"], family=m["family"], only=m.get("only", "both"),
                    phi=m["phi"].astype(np.float32), amp=m["amp"].astype(np.float32))
               for m in modes],
        K_modes=K, seed=seed, config=cfg.__dict__.copy(),
    )
    return field, truth


if __name__ == "__main__":
    f, t = generate_field()
    print("field", f.shape, f.dtype, "| modes:", t["K_modes"],
          "| families:", [m["family"] for m in t["modes"]])
