"""A small network that gives the DISTRIBUTION of the next latent state z(t+1), in the shape of each channel's dynamic
family (D-052, design record `memory/conditional_forecast_design.md` §16).

F-49 fitted the family shapes with FIXED parameters per channel; here the parameters are predicted per launch from what
is known at launch, so the distribution can widen, shift or skew with the state of the channel. Shapes (user, §15), on
the scaled error e = (z(t+1) - zhat(t)) / sigma_i (sigma_i = out-of-fold RMS of channel i):

  stationary, noise   Gaussian            centre m, width w
  cyclic              ramp + tail         along the direction of travel s, u = s (e - m):
                                              p(u) ∝ Phi(u / sb) * exp(-max(0, u - r)^2 / (2 st^2))
  chaotic             uniform window      centre m, half-width h (edges softened over 0.05 so it can be trained)

Inputs per channel (past + forecast only): zhat, z(t), the forecast step zhat - z(t), the std of the last 16 steps, a
channel embedding, and its family's own features --
  cyclic      phase (cos, sin) and amplitude of the sliding cos/sin fit, distance to the crest / A, move over the lead
  chaotic     the channel's last 8 values every 2 steps, relative to z(t) (the observer's chaos embedding, D-040)
  stationary  mean and std of the last 64 steps, relative to z(t)
One small MLP per family, shared by its channels. Trained by CRPS, computed from each density's CDF on a fixed grid of e
(a likelihood would be infinite for a truth outside a uniform window).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.family_prior import cycle_state, fft_period

GRID = torch.linspace(-8.0, 8.0, 641)                           # scaled-error grid for densities, CDFs and CRPS
SHAPES = dict(stationary="gauss", noise="gauss", cyclic="ramp", chaotic="uniform")
NPAR = dict(gauss=2, ramp=4, uniform=2)


def launch_features(z: np.ndarray, zhat: np.ndarray, sigma: np.ndarray, fam, lead: int, t_fit_max: int):
    """Per-channel input features for every launch t (row t = forecast of z(t+lead) made at t), scaled by sigma_i.

    Returns (common [T, K, 4], family features {family: [T, n_f, d_f]}, travel direction s [T, K] (cyclic; 1 elsewhere),
    first launch with every window full, periods {channel: P})."""
    T, K = z.shape
    zs, zh = z / sigma, zhat / sigma
    lag = lambda x, k: np.concatenate([np.full((k,) + x.shape[1:], np.nan), x[:-k]]) if k else x
    roll_std = lambda x, n: np.stack([lag(x, k) for k in range(n)]).std(0)
    common = np.stack([zh, zs, zh - zs, roll_std(zs, 16)], -1)
    fam = np.asarray(fam)
    feats, s, periods = {}, np.ones((T, K)), {}
    cyc = np.flatnonzero(fam == "cyclic")
    f = []
    for i in cyc:
        P = fft_period(z[:t_fit_max + lead, i])
        st = cycle_state(z[:, i], zhat[:, i], P, lead)
        periods[int(i)] = P
        s[:, i] = np.nan_to_num(st["s"], nan=1.0)
        f.append(np.stack([np.cos(st["theta"]), np.sin(st["theta"]), st["A"] / sigma[i],
                           st["dist"] / np.maximum(st["A"], 1e-8), st["move"] / sigma[i]], -1))
    feats["cyclic"] = np.stack(f, 1) if f else np.zeros((T, 0, 5))
    ch = np.flatnonzero(fam == "chaotic")
    feats["chaotic"] = np.stack([lag(zs[:, ch], 2 * k) - zs[:, ch] for k in range(1, 9)], -1)
    for name in ("stationary", "noise"):
        idx = np.flatnonzero(fam == name)
        L = np.stack([lag(zs[:, idx], k) for k in range(64)])
        feats[name] = np.stack([L.mean(0) - zs[:, idx], L.std(0)], -1)
    t0 = max([64] + [periods[i] for i in periods])
    return common, feats, s, t0, periods


def density(shape: str, par: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
    """Unnormalized density on GRID for parameters par [..., NPAR] -> [..., G] (raw network outputs, mapped here)."""
    g = GRID.to(par.device)
    m = par[..., :1]
    if shape == "gauss":
        w = F.softplus(par[..., 1:2]) + 0.05
        return torch.exp(-0.5 * ((g - m) / w) ** 2)
    if shape == "uniform":
        h = F.softplus(par[..., 1:2]) + 0.05
        return torch.sigmoid((h - (g - m).abs()) / 0.05)
    sb, st = (F.softplus(par[..., k:k + 1]) + 0.05 for k in (1, 2))
    r = F.softplus(par[..., 3:4])
    u = s[..., None] * (g - m)
    return 0.5 * torch.erfc(-u / (sb * 2 ** 0.5)) * torch.exp(-0.5 * (F.relu(u - r) / st) ** 2)


def cdf_on_grid(p: torch.Tensor) -> torch.Tensor:
    """Normalized CDF on GRID from an unnormalized density [..., G] (trapezoid)."""
    c = torch.cumsum(0.5 * (p[..., 1:] + p[..., :-1]), -1)
    c = torch.cat([torch.zeros_like(c[..., :1]), c], -1)
    return c / c[..., -1:].clamp_min(1e-12)


def crps_on_grid(cdf: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CRPS = int (F(x) - 1{x >= y})^2 dx, for CDFs [..., G] and truths y [...] in scaled-error units."""
    g = GRID.to(cdf.device)
    H = (g >= y[..., None]).float()
    return ((cdf - H) ** 2).sum(-1) * float(g[1] - g[0])


class FamilyHead(nn.Module):
    """One family's MLP: [common 4 + family features d + channel embedding 8] -> its shape's parameters."""

    def __init__(self, d: int, n_ch: int, npar: int, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(max(n_ch, 1), 8)
        self.net = nn.Sequential(nn.Linear(4 + d + 8, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, npar))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, common: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        B, n, _ = common.shape
        e = self.emb.weight[None].expand(B, n, 8)
        return self.net(torch.cat([common, feat, e], -1))


class FamilyNet(nn.Module):
    """Distribution of z(t+1) per channel, in its family's shape. `shapes` overrides the family -> shape map (e.g. all
    Gaussian, the control that keeps the per-launch adaptation but drops the family shape)."""

    def __init__(self, fam, feat_dims: dict, shapes: dict | None = None, hidden: int = 64):
        super().__init__()
        self.fam = np.asarray(fam)
        self.shapes = dict(SHAPES, **(shapes or {}))
        self.idx = {f: torch.as_tensor(np.flatnonzero(self.fam == f)) for f in SHAPES}
        self.heads = nn.ModuleDict({f: FamilyHead(feat_dims[f], len(self.idx[f]), NPAR[self.shapes[f]], hidden)
                                    for f in SHAPES if len(self.idx[f])})

    def cdf(self, common: torch.Tensor, feats: dict, s: torch.Tensor) -> torch.Tensor:
        """CDF of every channel's scaled error on GRID: [B, K, G]."""
        B, K, _ = common.shape
        out = torch.empty(B, K, len(GRID), device=common.device)
        for f, head in self.heads.items():
            i = self.idx[f].to(common.device)
            par = head(common[:, i], feats[f])
            out[:, i] = cdf_on_grid(density(self.shapes[f], par, s[:, i]))
        return out


@torch.no_grad()
def net_prior(net_dir: str, z: np.ndarray, zhat: np.ndarray, sigma: np.ndarray, fam, lead: int,
              device: str = "cpu") -> dict:
    """The all-Gaussian network (`gauss_net.pt` in net_dir) as a JOINT latent prior for the decoder (D-053).

    Per launch t and channel i: z(t+a) ~ zhat + sigma_i (m_ti + w_ti eta_i), with eta ~ N(0, R). m, w = the network's
    centre and width (scaled-error units); R = correlation of its standardized residuals (y - m) / w on the fit launches,
    shrunk 10 % toward I (the network gives marginals; R restores how the channels err together).
    Returns m, w [T, K] (NaN-free; launches before t0 have m = 0, w = 1), chol_R [K, K], t0 (first launch with features)."""
    ck = torch.load(f"{net_dir}/gauss_net.pt", map_location=device, weights_only=False)
    common, feats, s, t0, _ = launch_features(z, zhat, sigma, fam, lead, 7750)
    model = FamilyNet(fam, ck["feat_dims"], ck["shapes"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    T, K = z.shape
    m, w = np.zeros((T, K), np.float32), np.ones((T, K), np.float32)
    tt = np.arange(t0, T)
    for i0 in range(0, len(tt), 1000):
        t = tt[i0:i0 + 1000]
        c = torch.tensor(np.nan_to_num(common[t]), dtype=torch.float32, device=device)
        for f, head in model.heads.items():
            idx = model.idx[f].numpy()
            fe = torch.tensor(np.nan_to_num(feats[f][t]), dtype=torch.float32, device=device)
            par = head(c[:, idx], fe)
            m[t[:, None], idx[None]] = par[..., 0].cpu().numpy()
            w[t[:, None], idx[None]] = (F.softplus(par[..., 1]) + 0.05).cpu().numpy()
    fit = np.arange(t0, 7750 - lead)
    r = ((z[fit + lead] - zhat[fit]) / sigma - m[fit]) / w[fit]
    R = 0.9 * np.corrcoef(r.T) + 0.1 * np.eye(K)
    return dict(m=m, w=w, chol_R=np.linalg.cholesky(R), t0=t0)


# --- an EXAGGERATED family prior, for testing what the decoder makes of strongly shaped latent distributions (D-054) --
# user, after inspecting the estimated distributions: "stationary: Gaussian is good, but narrower; cyclic: more skewed and
# biased on the cyclic slope, and on the cyclic peak the skewed direction changes; chaotic: seems not bad; noise:
# literally uniform". Widths are in units of the network's own width w (scaled-error units), so the shapes ride on the
# network's per-launch centre m and width w.
# Chaotic (user, 2026-11-15: "go back to previous distribution (flat plateau shape)" ... "return to family shape
# distribution (represented in blue in latent.png)") = EXACTLY the family network's own density per launch (family.pt
# of the same lead): the plateau with its softened edges and tails, read from its CDF -- not a hard window.
# Noise (user, 2026-11-15: "for noise, let's put family shape as well, not uniform") = likewise the family network's own
# density per launch (a Gaussian for noise), replacing the uniform over the training range.
DRAMATIC = dict(stationary_width=0.5,          # Gaussian, width x 0.5
                cyclic_sb=0.3,                 # sharp rise behind the direction of travel
                cyclic_reach=1.5,              # plateau ahead, reach r = 1.5 w |sin theta| (0 at the crest / trough)
                cyclic_st=0.5)                 # tail past the reach


@torch.no_grad()
def family_quantiles(fn_dir: str, family: str, z: np.ndarray, zhat: np.ndarray, sigma: np.ndarray, fam, lead: int,
                     t: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles [n, n_ch, L] (scaled-error units) of the family network's own density for one family at launches t --
    the shape it was trained with (the blue curves of latent.png), inverted from its CDF on GRID. Not for cyclic (its
    density needs the travel direction s)."""
    ck = torch.load(f"{fn_dir}/family.pt", map_location="cpu", weights_only=False)
    common, feats, _, _, _ = launch_features(z, zhat, sigma, fam, lead, 7750)
    model = FamilyNet(fam, ck["feat_dims"], ck["shapes"])
    model.load_state_dict(ck["state"])
    model.eval()
    assert family != "cyclic", family
    idx = model.idx[family].numpy()
    par = model.heads[family](torch.tensor(np.nan_to_num(common[t][:, idx]), dtype=torch.float32),
                              torch.tensor(np.nan_to_num(feats[family][t]), dtype=torch.float32))
    cdf = cdf_on_grid(density(model.shapes[family], par)).numpy()                        # [n, n_ch, G]
    g = GRID.numpy()
    out = np.empty(cdf.shape[:2] + (len(levels),), np.float32)
    for a_ in range(cdf.shape[0]):
        for b_ in range(cdf.shape[1]):
            c = np.maximum.accumulate(cdf[a_, b_])
            keep = np.concatenate([[True], np.diff(c) > 1e-12])                           # strictly increasing for interp
            out[a_, b_] = np.interp(levels, c[keep], g[keep])
    return out


def dramatic_tables(npr: dict, z: np.ndarray, zhat: np.ndarray, sigma: np.ndarray, fam, lead: int, t: np.ndarray,
                    z_range: tuple, own: dict | None = None, eta: np.ndarray | None = None) -> np.ndarray:
    """Quantile tables [n, K, L] (offset of z from zhat on the normal-score grid ETA) of the exaggerated prior.

    npr = `net_prior` output (m, w per launch); z_range = (lo, hi) per channel on the training set (noise uniform).
      stationary  zhat + sigma (m + 0.5 w eta)
      cyclic      ramp + tail along s (sliding-fit direction at the target), centre m, scales x w, reach x |sin theta|:
                  on the slope most of the mass lies ahead of the forecast; past the crest s flips, so does the skew
      chaotic     the family network's own density (plateau with soft edges): zhat + sigma q(eta)
      noise       the family network's own density (Gaussian), zhat + sigma q(eta)
                  (own = {family: q [n, n_ch, L]} from `family_quantiles`; a family missing from it falls back to the
                  network Gaussian for chaotic and to the uniform over the training range [z_range] for noise)"""
    from scipy.special import ndtr
    from src.models.family_prior import ETA, ramp_quantiles
    eta = ETA if eta is None else eta
    fam = np.asarray(fam)
    m, w = npr["m"][t], npr["w"][t]
    tab = sigma[None, :, None] * (m[..., None] + w[..., None] * eta[None, None])          # network Gaussian
    st_ = fam == "stationary"
    tab[:, st_] = sigma[None, st_, None] * (m[:, st_, None] + DRAMATIC["stationary_width"] * w[:, st_, None] * eta)
    own = own or {}
    nz = fam == "noise"
    lo, hi = z_range
    tab[:, nz] = (lo[nz][None, :, None] + (hi - lo)[nz][None, :, None] * ndtr(eta)[None, None]) - zhat[t][:, nz, None]
    for f, q in own.items():
        k = fam == f
        tab[:, k] = sigma[None, k, None] * q
    for i in np.flatnonzero(fam == "cyclic"):
        P = fft_period(z[:7750 + lead, i])
        cs = cycle_state(z[:, i], zhat[:, i], P, lead)
        s = np.nan_to_num(cs["s"][t], nan=1.0)
        r = DRAMATIC["cyclic_reach"] * np.abs(np.sin(np.nan_to_num(cs["theta"][t])))
        q = ramp_quantiles(r, DRAMATIC["cyclic_sb"], DRAMATIC["cyclic_st"], ndtr(eta))     # [n, L], in units of w
        q = np.where(s[:, None] > 0, q, -q[:, ::-1])                                       # increasing in eta
        tab[:, i] = sigma[i] * (m[:, i, None] + w[:, i, None] * q)
    return tab.astype(np.float32)


class DramaticSampler:
    """Joint draws of the exaggerated prior for a decoder run trained with fd.prior=net (uses its prior.npz,
    netprior.npz and the observer families). Tables are built once for the requested launches."""

    def __init__(self, run: str, hf_dir: str, obs_seed: int, launches: np.ndarray, device: str,
                 fn_dir: str | None = None):
        pr = np.load(f"{run}/prior.npz", allow_pickle=True)
        npr = dict(np.load(f"{run}/netprior.npz"))
        self.fam = np.load(f"{hf_dir}/twin_seed{obs_seed}/norm.npz")["fam"]
        z, zhat, lead, t_tr = pr["z"], pr["zhat"], int(pr["lead"]), int(pr["t_tr"])
        z_range = (z[:t_tr].min(0), z[:t_tr].max(0))
        if fn_dir is None:                                         # a finished run: its own config names the network
            fn_dir = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)["cfg"]["prior_net"]
        from scipy.special import ndtr
        from src.models.family_prior import ETA
        own = {f: family_quantiles(fn_dir, f, z, zhat, npr["sigma"], self.fam, lead, launches, ndtr(ETA))
               for f in ("chaotic", "noise")}
        tab = dramatic_tables(npr, z, zhat, npr["sigma"], self.fam, lead, launches, z_range, own)
        self.row = {int(t): j for j, t in enumerate(launches)}
        self.tab = torch.tensor(tab, device=device)
        self.zhat = torch.tensor(zhat, dtype=torch.float32, device=device)
        self.Lr = torch.tensor(npr["chol_R"], dtype=torch.float32, device=device)

    def sample(self, t, M: int, eta: torch.Tensor | None = None) -> torch.Tensor:
        """Launches t (tensor or array, all among `launches`) -> [B, M, K] canonical z (eta: fixed normals [B, M, K])."""
        from src.models.family_prior import sample_from_tables
        tl = [int(x) for x in (t.tolist() if torch.is_tensor(t) else t)]
        j = torch.as_tensor([self.row[x] for x in tl], device=self.tab.device)
        return sample_from_tables(self.zhat[torch.as_tensor(tl, device=self.tab.device)], self.tab[j], self.Lr, M, eta)
