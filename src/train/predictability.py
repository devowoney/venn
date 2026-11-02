"""Predictability family measure (D-035, W2): one implementation for the loss AND the post-hoc labeller.

WHY this replaces the spectral shape test. `structure_share` cannot tell chaos from coloured noise and calls a
quasi-periodic signal chaotic (F-30: 36/36 wrong). Chaos is a property of the DYNAMICS, so each channel is scored by
how well its OWN past predicts its future -- an observer-side measuring stick, not module 2:

  p_long = 1 - e_long(64)   LINEAR ridge on a long delay window (span 120): predictable at the forecast horizon?
  g      = mean_{h<=22} (e_lin(h) - e_nl(h))   on a short window (span 14): does a NONLINEAR ridge beat the linear one?
           A linear score is fixed by the spectrum (Wiener-Kolmogorov); only this gain sees determinism.

e(h) = test MSE / var, fit on the first half of the series and scored on the second (in-sample, 512 random features
would "predict" white noise). Everything is closed-form ridge, so the scores are differentiable in the series and
batched over channels. Calibration of windows and cuts: F-30 and `src/probes/predictability.py`.
"""
from __future__ import annotations

import torch

PRED_FAMILIES = ("slow", "cyclic", "fast", "noise")   # q column -> role name (slow/fast = stationary/chaotic, as before)


def make_rff(m: int, n_feat: int, seed: int, device=None) -> dict:
    """Fixed random Fourier features for the nonlinear score: drawn once per run, so g is a deterministic function of
    the series (a fresh draw every step would add noise to the family vote)."""
    gen = torch.Generator().manual_seed(int(seed))
    W = torch.randn(m, n_feat, generator=gen, dtype=torch.float64)
    b = 2 * torch.pi * torch.rand(n_feat, generator=gen, dtype=torch.float64)
    return dict(W=W.to(device), b=b.to(device))


def _embed(x: torch.Tensor, m: int, s: int, t: torch.Tensor) -> torch.Tensor:
    """x [K,T], rows t [N] -> delay embedding [K,N,m] = x[t], x[t-s], ..., x[t-(m-1)s]."""
    return torch.stack([x[:, t - j * s] for j in range(m)], dim=2)


def _ridge_err(F: torch.Tensor, Y: torch.Tensor, fit: torch.Tensor, te: torch.Tensor, lam: float) -> torch.Tensor:
    """Batched closed-form ridge per channel. F [K,N,P], Y [K,N,H] -> normalized test error e [K,H]."""
    Ff, Yf = F[:, fit], Y[:, fit]
    G = Ff.transpose(1, 2) @ Ff
    G = G + lam * Ff.shape[1] * torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    beta = torch.linalg.solve(G, Ff.transpose(1, 2) @ Yf)
    Yt = Y[:, te]
    return ((F[:, te] @ beta - Yt) ** 2).mean(dim=1) / (Yt.var(dim=1) + 1e-12)


def pred_scores(S: torch.Tensor, pc, rff: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """p_long [K], g [K] for the channel series S [T,K]. `pc` = cfg.train.spectral.pred.

    Computed in float64: the 521x521 nonlinear normal equations are badly conditioned in float32.
    """
    m_s, s_s, m_l, s_l = int(pc.m_short), int(pc.s_short), int(pc.m_long), int(pc.s_long)
    h_max, h_g = int(pc.h_long), int(pc.h_gain)
    x = S.T.to(torch.float64)                                  # [K,T]
    T = x.shape[1]
    half = T // 2
    x = (x - x[:, :half].mean(1, keepdim=True)) / (x[:, :half].std(1, keepdim=True) + 1e-12)
    t = torch.arange((m_l - 1) * s_l, T - h_max, device=x.device)   # same rows for both windows, so scores compare
    fit, te = t < half - h_max, t >= half                      # halves separated by h_max: no leaked target

    # --- long window, linear: predictable at lead h_long? ------------------------------------------------------
    one = torch.ones(x.shape[0], len(t), 1, dtype=x.dtype, device=x.device)
    F_long = torch.cat([one, _embed(x, m_l, s_l, t)], dim=2)
    e_long = _ridge_err(F_long, x[:, t + h_max][:, :, None], fit, te, float(pc.lam_lin))[:, 0]

    # --- short window, linear vs nonlinear: deterministic beyond the spectrum? -------------------------------
    E = _embed(x, m_s, s_s, t)                                 # [K,N,m_s]
    Y = torch.stack([x[:, t + h] for h in range(1, h_g + 1)], dim=2)   # [K,N,h_g]
    F_lin = torch.cat([one, E], dim=2)
    with torch.no_grad():                                      # kernel width = median embedded distance (a scale,
        sub = E[:, fit][:, :: max(1, int(fit.sum()) // 400)]   # not something the eye should optimize)
        ell = torch.cdist(sub, sub).flatten(1).median(dim=1).values.clamp_min(1e-6)   # [K]
    Z = (E @ rff["W"]) / ell[:, None, None] + rff["b"]
    F_nl = torch.cat([F_lin, (2.0 / rff["W"].shape[1]) ** 0.5 * torch.cos(Z)], dim=2)
    e_lin = _ridge_err(F_lin, Y, fit, te, float(pc.lam_lin))
    e_nl = _ridge_err(F_nl, Y, fit, te, float(pc.lam_nl))
    g = (e_lin - e_nl).mean(dim=1)
    return (1.0 - e_long).to(S.dtype), g.to(S.dtype)


def pred_family_q(q_stat: torch.Tensor, p_long: torch.Tensor, g: torch.Tensor, pc, tau: float) -> torch.Tensor:
    """q [K,4] = (stat, cyc, chao, noise), nested in the labeller's ORDER: flat -> predictable -> deterministic -> rest.

    The flatness vote `q_stat` comes from the unchanged soft flatness measure. argmax q = `label_predictability`.
    """
    s_c = torch.sigmoid(float(pc.beta_p) * (p_long - float(pc.p_cut)) / tau)
    s_g = torch.sigmoid(float(pc.beta_g) * (g - float(pc.g_cut)) / tau)
    rest = 1 - q_stat
    return torch.stack([q_stat, rest * s_c, rest * (1 - s_c) * s_g, rest * (1 - s_c) * (1 - s_g)], dim=1)


def pred_family_terms(S_all: torch.Tensor, q_stat: torch.Tensor, pen_flat: torch.Tensor, sp, pop: torch.Tensor,
                      rff: dict, tau: float):
    """Same contract as `spectral.soft_family_terms`, in the predictability quantities (D-035).

      stationary : pen_flat (the unchanged flatness hinge, weighted as in the spectral measure)
      cyclic     : lambda_pcyc * relu(p_target - p_long)^2                    "be predictable at the horizon"
      chaotic    : lambda_g * relu(g_target - g)^2 + lambda_pcap * relu(p_long - p_cap)^2
                   "be deterministic beyond your spectrum, and not long-predictable"
      noise      : 0 -- what is left; decomposing it is the deeper eyes' job (W1, user 2026-10-30)
                   or, with `pred.noise_penalty: structure`, min(cyclic, chaotic) -- pushed to its nearest family
    """
    pc = sp.pred
    p_long, g = pred_scores(S_all, pc, rff)
    q = pred_family_q(q_stat, p_long, g, pc, tau)
    terms = dict(pcyc=float(pc.lambda_pcyc) * torch.relu(float(pc.p_target) - p_long) ** 2,
                 g=float(pc.lambda_g) * torch.relu(float(pc.g_target) - g) ** 2,
                 pcap=float(pc.lambda_pcap) * torch.relu(p_long - float(pc.p_cap)) ** 2)
    pen_cyc, pen_chao = terms["pcyc"], terms["g"] + terms["pcap"]
    if str(pc.get("noise_penalty", "none")) == "structure":
        # noise pays what its NEAREST real family would pay (user 2026-11-02, "push toward structure"): the channel
        # feels that family's hinge, and noise is never cheaper than it, so becoming noise is no longer rewarded.
        # Not min(pcyc, g) alone: that would still undercut the chaotic family (which also pays the p_long cap).
        pen_noise = torch.minimum(pen_cyc, pen_chao)
    else:
        pen_noise = torch.zeros_like(p_long)                   # D-035 default: noise is free (W1 decomposes it)
    pen = torch.stack([pen_flat, pen_cyc, pen_chao, pen_noise], dim=1)   # [K,4]
    l_family = (q * pen).sum(dim=1).mean()
    qbar = q.mean(dim=0).clamp_min(1e-8)
    l_pop = (pop * (pop.clamp_min(1e-8).log() - qbar.log())).sum()   # noise share 0 in pop: contributes nothing
    l_ent = -(q * q.clamp_min(1e-8).log()).sum(dim=1).mean()
    # `terms` stay LIVE (not detached): train.py gradient-matches each hinge against L_white at step 0
    return l_family, l_pop, l_ent, q, dict(p_long=p_long.detach(), g=g.detach(), pen=pen.detach(), terms=terms)
