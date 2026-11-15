"""Family-shaped latent forecast p(z(t+a)) -- module 2's forecast read through the observer's dynamic families (D-051,
design record `memory/conditional_forecast_design.md` §15).

The plain prior is one Gaussian N(zhat, Sigma) for every channel. Here each live channel gets the error SHAPE of its
family (the observer's own vote, `fam` in module 2's norm.npz), around the same forecast zhat:

  stationary, noise   Gaussian            zhat + sigma * eta
  chaotic             uniform window      zhat + sqrt(3) sigma * (2 Phi(eta) - 1)    (same variance as the forecast error;
                                          user ruling: a window at short leads, widening toward the range at long ones)
  cyclic              ramp + tail         along the direction of travel s, u = s (z - zhat):
                                              p(u) ∝ Phi(u / sb) * exp(-max(0, u - r)^2 / (2 st^2))
                                          the cycle is more likely to keep going than to turn back (CDF ramp) up to the
                                          REACH r, then falls off like a tail. r = min(distance to the crest / trough,
                                          kappa * the cycle's own move over the lead); near the crest r -> 0.

The cyclic launch state comes from a sliding fit c + a cos(w tau) + b sin(w tau) over the channel's last period
(w from its FFT period on the training set), ending at zhat: tau = 0 is the target time, so the fitted slope there is
w b (s = sign b), the amplitude is sqrt(a^2 + b^2) and the crest / trough is c + s A. Only the past and zhat are read.

The channels are joined by a GAUSSIAN COPULA: correlated normal scores eta ~ N(0, R) are pushed through each channel's
own quantile function, so every channel keeps its family shape and the channels keep their joint behaviour. R is the
correlation of the normal scores of the out-of-fold departures under the fitted marginals.

Everything lives in canonical z units (live channels standardized on [0, t_tr)). Per launch, every channel's quantile
function is stored as a table of offsets from zhat on a fixed grid of normal scores, so drawing a correlated sample is
one interpolation (cheap enough for every training step of the decoder).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri

ETA = np.linspace(-4.0, 4.0, 81)                       # normal-score grid of the quantile tables
SQ3 = np.sqrt(3.0)


def fft_period(x: np.ndarray, pmin: int = 8) -> int:
    """Dominant period (steps) of a training series: the strongest Fourier line, mean removed, periods >= pmin."""
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x))
    ok = (f > 0) & (f <= 1.0 / pmin)
    return int(round(1.0 / f[ok][np.argmax(P[ok])]))


def cycle_state(z: np.ndarray, zhat: np.ndarray, P: int, lead: int) -> dict:
    """Sliding cos/sin fit of ONE channel for every launch t (rows of zhat = forecast of z(t+lead) made at t).

    Window: one full period of the channel's own past, z(t-P+1 .. t), and the forecast zhat(t) at tau = 0 (the target
    time t+lead); a full period of past at every lead, so leads longer than the period still have a window.
    Returns per launch: s (direction of travel), dist (how far zhat is from the crest / trough it is heading to, >= 0),
    move (the fitted cycle's own change over the lead), A (amplitude) and theta (phase at the target time). Launches without a full window are NaN."""
    T, w = len(z), 2 * np.pi / P
    # window times relative to the target: the past z(t-P+1 .. t) sit at -lead-P+1 .. -lead, the forecast at 0 (for
    # lead > 1 the steps between t and the target are not known, so the window has a gap there)
    tau = np.concatenate([np.arange(-lead - P + 1, -lead + 1), [0]])
    X = np.stack([np.ones(len(tau)), np.cos(w * tau), np.sin(w * tau)], 1)
    G = np.linalg.inv(X.T @ X)                                     # same window shape for every launch
    past = np.lib.stride_tricks.sliding_window_view(z, P)          # row j = z(j .. j+P-1)
    out = {k: np.full(T, np.nan) for k in ("s", "dist", "move", "A", "theta")}
    t = np.arange(P - 1, T)                                        # launches with a full window
    Y = np.concatenate([past[t - (P - 1)], zhat[t, None]], 1)      # [n, P + 1]
    c, a, b = (Y @ X @ G).T
    A = np.hypot(a, b)
    s = np.where(b >= 0, 1.0, -1.0)
    out["s"][t] = s
    out["A"][t] = A
    out["theta"][t] = np.arctan2(-b, a)                            # phase at the target: f(0) = c + A cos(theta)
    out["dist"][t] = np.maximum(A - s * (zhat[t] - c), 0.0)
    out["move"][t] = np.abs(a * (1 - np.cos(w * lead)) + b * np.sin(w * lead))
    return out


def ramp_logpdf(u: np.ndarray, r: np.ndarray, sb: float, st: float) -> np.ndarray:
    """log p(u) of the ramp + tail density, normalized per launch (r varies per launch).

    Normalizer = int_{-inf}^{r} Phi(u/sb) du  (closed form: sb [rho Phi(rho) + phi(rho)], rho = r / sb)
               + int_0^inf Phi((r+v)/sb) exp(-v^2 / 2st^2) dv   (trapezoid on a fixed grid of v)."""
    rho = r / sb
    I1 = sb * (rho * ndtr(rho) + np.exp(-0.5 * rho ** 2) / np.sqrt(2 * np.pi))
    v = np.linspace(0, 8 * st, 161)
    I2 = np.trapezoid(ndtr((r[:, None] + v[None]) / sb) * np.exp(-0.5 * (v / st) ** 2)[None], v, axis=1)
    logp = np.log(np.maximum(ndtr(u / sb), 1e-300)) - 0.5 * (np.maximum(u - r, 0.0) / st) ** 2
    return logp - np.log(I1 + I2)


def ramp_quantiles(r: np.ndarray, sb: float, st: float, levels: np.ndarray, n: int = 600) -> np.ndarray:
    """Quantiles of the ramp + tail density at probability `levels` [L], for every launch's reach r [N] -> [N, L]."""
    lo, hi = -7 * sb, r + 7 * st
    g = lo + (hi - lo)[:, None] * np.linspace(0, 1, n)[None]       # per-launch grid [N, n]
    p = ndtr(g / sb) * np.exp(-0.5 * (np.maximum(g - r[:, None], 0.0) / st) ** 2)
    cdf = np.concatenate([np.zeros((len(r), 1)), np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(g, axis=1), 1)], 1)
    cdf /= cdf[:, -1:]
    return np.stack([np.interp(levels, cdf[i], g[i]) for i in range(len(r))])


class FamilyPrior:
    """p(z(t+a) | zhat(t)) with family-shaped marginals and a Gaussian copula. Build with `fit`, draw with `sample`."""

    def fit(self, z: np.ndarray, zhat: np.ndarray, Sigma: np.ndarray, fam, lead: int, fit_t: np.ndarray,
            shrink: float = 0.1) -> "FamilyPrior":
        """z, zhat [T, K] canonical units (zhat out-of-fold on the training launches); fit_t = training launches whose
        departures z(t+a) - zhat(t) fit the cyclic shapes and the copula."""
        T, K = z.shape
        self.fam, self.lead, self.K = np.asarray(fam), int(lead), K
        self.sigma = np.sqrt(np.diag(Sigma))
        self.cyc = np.flatnonzero(self.fam == "cyclic")
        self.period = {}
        self.par = {}                                              # cyclic channel -> (sb, st, kappa)
        self.state = {}
        t_fit_max = int(fit_t.max()) + 1
        for i in self.cyc:
            P = fft_period(z[: t_fit_max + lead, i])
            st_ = cycle_state(z[:, i], zhat[:, i], P, self.lead)
            self.period[i], self.state[i] = P, st_
        # cyclic launches need a full window: the usable launches start after the longest period
        self.t0 = max([self.period[i] for i in self.cyc], default=0)
        ft = fit_t[fit_t >= self.t0]
        for i in self.cyc:
            st_ = self.state[i]
            u = st_["s"][ft] * (z[ft + lead, i] - zhat[ft, i])
            dist, move = st_["dist"][ft], st_["move"][ft]
            sig = self.sigma[i]

            def nll(q):
                sb, st, kap = np.exp(q)
                return -ramp_logpdf(u, np.minimum(dist, kap * move), sb, st).mean()

            best = None
            for k0 in (0.1, 1.0, 5.0):                             # a few starts: the reach scale is the shaky one
                res = minimize(nll, np.log([sig, sig, k0]), method="Nelder-Mead",
                               options=dict(xatol=1e-4, fatol=1e-6, maxiter=2000))
                best = res if best is None or res.fun < best.fun else best
            self.par[i] = tuple(np.exp(best.x))
        # copula: normal scores of the fit departures under the fitted marginals
        eta = ndtri(np.clip(self.pit(z[ft + lead], zhat[ft], ft), 1e-4, 1 - 1e-4))
        C = np.corrcoef(eta.T)
        self.R = (1 - shrink) * C + shrink * np.eye(K)
        self.Lr = np.linalg.cholesky(self.R)
        return self

    def reach(self, i: int, t: np.ndarray) -> np.ndarray:
        st_ = self.state[i]
        return np.minimum(st_["dist"][t], self.par[i][2] * st_["move"][t])

    def pit(self, ztrue: np.ndarray, zhat: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Probability integral transform of the truth under each channel's marginal: [n, K] in [0, 1]."""
        d = (ztrue - zhat) / self.sigma
        out = ndtr(d)                                              # Gaussian (stationary, noise)
        ch = self.fam == "chaotic"
        out[:, ch] = np.clip((d[:, ch] / SQ3 + 1) / 2, 0, 1)
        lv = np.linspace(0, 1, 401)
        for i in self.cyc:
            s = self.state[i]["s"][t]
            q = ramp_quantiles(self.reach(i, t), *self.par[i][:2], lv)          # [n, 401] in u
            u = s * (ztrue[:, i] - zhat[:, i])
            pu = np.array([np.interp(u[j], q[j], lv) for j in range(len(t))])
            out[:, i] = np.where(s > 0, pu, 1 - pu)                # PIT in z: flip when travelling down
        return out

    def tables(self, t: np.ndarray) -> np.ndarray:
        """Quantile tables [n, K, len(ETA)]: offset of z from zhat at each normal score, for launches t."""
        n = len(t)
        tab = np.empty((n, self.K, len(ETA)), dtype=np.float32)
        tab[:] = (self.sigma[:, None] * ETA[None])[None]
        ch = self.fam == "chaotic"
        tab[:, ch] = (SQ3 * self.sigma[ch, None] * (2 * ndtr(ETA) - 1)[None])[None]
        for i in self.cyc:
            s = self.state[i]["s"][t]
            q = ramp_quantiles(self.reach(i, t), *self.par[i][:2], ndtr(ETA))   # [n, L] in u, increasing
            # z - zhat = s u; travelling down reverses the order, so the table stays increasing in eta
            tab[:, i] = np.where(s[:, None] > 0, q, -q[:, ::-1])
        return tab


def sample_from_tables(zhat: torch.Tensor, tab: torch.Tensor, Lr: torch.Tensor, M: int,
                       eta: torch.Tensor | None = None) -> torch.Tensor:
    """Correlated draws: zhat [B, K], tables [B, K, L] (on the ETA grid), copula Cholesky Lr [K, K] -> [B, M, K].
    eta (optional) = independent standard normals [B, M, K], to reuse fixed noise."""
    B, K, L = tab.shape
    eta = torch.randn(B, M, K, device=tab.device) if eta is None else eta
    e = (eta @ Lr.T).clamp(float(ETA[0]), float(ETA[-1]))          # correlated normal scores
    x = (e - float(ETA[0])) / float(ETA[1] - ETA[0])               # fractional grid index
    j = x.floor().long().clamp(0, L - 2)
    f = x - j
    tb = tab[:, None].expand(B, M, K, L)
    lo = tb.gather(-1, j[..., None])[..., 0]
    hi = tb.gather(-1, (j + 1)[..., None])[..., 0]
    return zhat[:, None] + lo + f * (hi - lo)
