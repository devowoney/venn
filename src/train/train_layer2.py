"""Train LAYER 2 ("sharpen sight", D-039 / SOP 02b) on top of a frozen, accepted layer-1 run.

Layer 1 (D-038) is a family-free sensor: it sees everything, but its eyes are mixtures (band + blob = static level +
one dynamic patch; cycle + chaos). Layer 2 splits every NON-cyclic eye into `layer2.n_child` children -- a soft
partition of the eye's cells -- and lets each child take its own family (stationary / cyclic / chaotic / noise) under
the tight D-036 criteria. Cyclic eyes are not split (user: "the cyclic signal should not be decomposed").

Loss (children only; the parents are frozen):
  L_family   the D-036 4-family terms on every child, weighted by its mass (an emptied child costs nothing)
  L_div      siblings take DIFFERENT families: sum_{i<j} w_i w_j <q_i, q_j> per parent
  L_copy     no near-copies among siblings: relu(|corr| - copy_thr)^2
Faithfulness needs no term: the children sum to their parent by construction.

Run (one per accepted layer-1 seed; data / seed overrides must match the parent run):
  python -m src.train.train_layer2 seed=0 data.T=10000 train.t_train=8000 \
      layer2.parent_run=$PWD/.tmps/runs/obs_t8000_l1_seed0 hydra.run.dir=$PWD/.tmps/runs/obs_t8000_l2_seed0
"""
from __future__ import annotations

import json
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from src.data.synthetic import GenConfig, generate_field, normalize_input
from src.models.child_encoder import ChildPartition
from src.probes.family import label_predictability
from src.train.predictability import PRED_FAMILIES, make_rff, pred_family_terms, pred_scores
from src.train.spectral import flat_vote
from src.train.train import _anneal


def sibling_terms(S: torch.Tensor, q: torch.Tensor, w: torch.Tensor, n_split: int, n_child: int, thr: float):
    """L_div and L_copy for children grouped by parent. S [T, n_split*n_child], q [.., 4], w [n_split, n_child]."""
    qg = q.view(n_split, n_child, -1)                                  # [P, C, 4]
    wq = w[..., None] * qg
    # sum_{i<j} w_i w_j <q_i, q_j> = ( |sum_i w_i q_i|^2 - sum_i w_i^2 |q_i|^2 ) / 2, per parent
    l_div = (0.5 * ((wq.sum(1) ** 2).sum(-1) - (wq ** 2).sum((1, 2)))).mean()
    Z = (S - S.mean(0)) / (S.std(0) + 1e-12)
    Zg = Z.view(Z.shape[0], n_split, n_child).permute(1, 0, 2)          # [P, T, C]
    C = Zg.transpose(1, 2) @ Zg / Z.shape[0]                            # [P, C, C] sibling correlation
    iu = torch.triu_indices(n_child, n_child, 1)
    pair_w = w[:, iu[0]] * w[:, iu[1]]
    l_copy = (pair_w * torch.relu(C[:, iu[0], iu[1]].abs() - thr) ** 2).sum() / pair_w.sum().clamp_min(1e-6)
    return l_div, l_copy, C.detach()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    L2, sp = cfg.layer2, cfg.train.spectral
    assert L2.parent_run, "layer2.parent_run is required (an accepted layer-1 run dir)"
    torch.manual_seed(cfg.seed)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"

    # --- the frozen parent run and ITS field (same data config and seed, else the masks see another field) --------
    par = np.load(os.path.join(L2.parent_run, "artifacts.npz"), allow_pickle=True)
    pcfg = OmegaConf.load(os.path.join(L2.parent_run, ".hydra", "config.yaml"))
    if OmegaConf.to_container(pcfg.data) != OmegaConf.to_container(cfg.data) or int(pcfg.seed) != int(cfg.seed):
        raise ValueError("data config / seed differ from the parent run: pass the same data.* and seed overrides")
    field_np, truth = generate_field(GenConfig(**OmegaConf.to_container(pcfg.data, resolve=True)), seed=int(pcfg.seed))
    t_fit = int(par["t_fit"])
    field_np = normalize_input(field_np, pcfg.train.get("input_norm", "none"), t_fit)
    field = torch.from_numpy(field_np).to(device)
    field_fit = field[:t_fit]
    parent_masks = torch.from_numpy(par["masks"].astype(np.float32)).to(device)
    K = parent_masks.shape[0]

    # --- parent families, fixed before training (layer 1 has no vote: the predictability labeller names them) ------
    S_par = par["S"].astype(np.float64)
    parent_label, _ = label_predictability(S_par[:t_fit], pc=sp.pred)
    split_idx = [k for k in range(K) if parent_label[k] != "cyclic"]
    print(f"[layer2] parent families: {parent_label} -> split {len(split_idx)} eyes, pass {K - len(split_idx)} cyclic")

    nC = int(L2.n_child)
    enc = ChildPartition(parent_masks, split_idx, n_child=nC, temp=float(L2.temp0), init_std=float(L2.init_std),
                         sigma=float(L2.init_sigma), generator=torch.Generator().manual_seed(int(cfg.seed))).to(device)
    nP = len(split_idx)
    opt = torch.optim.Adam(enc.parameters(), lr=float(L2.lr))
    rff = make_rff(int(sp.pred.m_short), int(sp.pred.n_rff), int(cfg.seed), device=device)
    pop = torch.zeros(4, device=device)                     # no population target (D-036); l_pop is then 0
    steps, snap_every, log_every = int(L2.max_steps), int(L2.snap_every), int(L2.log_every)
    writer = SummaryWriter(log_dir="tb")
    q_snaps, m_snaps, snap_steps = [], [], []
    tau_end = sp.soft.get("tau_end")

    for step in range(steps):
        enc.temp = _anneal(step, steps, float(L2.temp0), float(L2.temp1))
        tau = _anneal(step, steps, float(sp.soft.tau), float(tau_end)) if tau_end else float(sp.soft.tau)
        S = enc(field_fit, children_only=True)                           # [t_fit, nP*nC]
        w = (enc.mass() / float(L2.fade_frac)).clamp(0, 1).detach()      # [nP, nC] alive weight of each child
        q_stat, pen_flat = flat_vote(S, sp, tau)
        l_family, _, _, q, diag = pred_family_terms(S, q_stat, pen_flat, sp, pop, rff, tau, weights=w.flatten())
        l_div, l_copy, Csib = sibling_terms(S, q, w, nP, nC, float(L2.copy_thr))
        if step == 0:   # D-019: report each term's init gradient against the family term (the reference)
            g = {nm: torch.autograd.grad(t, enc.logits, retain_graph=True, allow_unused=True)[0]
                 for nm, t in (("family", l_family), ("div", l_div), ("copy", l_copy))}
            gn = {nm: float(v.norm()) if v is not None else 0.0 for nm, v in g.items()}
            print(f"[layer2] init gradients: |g_family|={gn['family']:.2e} |g_div|={gn['div']:.2e} "
                  f"|g_copy|={gn['copy']:.2e} (lambda_div {L2.lambda_div}, lambda_copy {L2.lambda_copy})")
        loss = l_family + float(L2.lambda_div) * l_div + float(L2.lambda_copy) * l_copy
        opt.zero_grad()
        loss.backward()
        opt.step()

        if snap_every and (step % snap_every == 0 or step == steps - 1):
            q_snaps.append(q.detach().cpu().numpy())
            m_snaps.append(enc.child_masks().detach().cpu().numpy().astype(np.float16))
            snap_steps.append(step)
        if step % log_every == 0 or step == steps - 1:
            alive = (w.flatten() >= 1.0).cpu().numpy()
            hard = q.detach().argmax(1).cpu().numpy()
            pops = [int(((hard == j) & alive).sum()) for j in range(4)]
            print(f"  step {step:5d} | L={loss.item():.4f} family={l_family.item():.4f} div={l_div.item():.4f} "
                  f"copy={l_copy.item():.4f} | live children {int(alive.sum())}/{nP * nC} stat/cyc/chao/noise "
                  f"{'/'.join(map(str, pops))} | max |sib corr| (live pairs) "
                  f"{(Csib.abs() * (w[:, :, None] * w[:, None, :] >= 1)).triu(1).max():.2f} "
                  f"temp={enc.temp:.2f} tau={tau:.2f}", flush=True)
            for nm, v in (("loss/family", l_family), ("loss/div", l_div), ("loss/copy", l_copy)):
                writer.add_scalar(nm, v.item(), step)
            writer.add_scalar("children/live", float(alive.sum()), step)

    # --- artifacts: the layer-1 keys, so capture / coverage / kernels read layer 2 the same way ---------------------
    with torch.no_grad():
        S_out = enc(field).cpu().numpy()                                 # [T, nP*nC + n_pass]
        masks = enc.masks().cpu().numpy()
        S_ch = enc(field_fit, children_only=True)
        w = (enc.mass() / float(L2.fade_frac)).clamp(0, 1)
        q_stat, pen_flat = flat_vote(S_ch, sp, tau)
        _, _, _, q_ch, fin = pred_family_terms(S_ch, q_stat, pen_flat, sp, pop, rff, tau, weights=w.flatten())
        S_pass = enc(field_fit)[:, nP * nC:]
        pl_pass, g_pass = pred_scores(S_pass, sp.pred, rff) if S_pass.shape[1] else (torch.zeros(0), torch.zeros(0))
    n_pass = len(enc.pass_idx)
    q_pass = np.zeros((n_pass, 4), dtype=np.float32)
    q_pass[:, 1] = 1.0                                                   # pass-through eyes were labelled cyclic
    q_all = np.concatenate([q_ch.cpu().numpy(), q_pass])
    roles = [PRED_FAMILIES[j] for j in q_ch.argmax(1).cpu().numpy()] + ["cyclic"] * n_pass
    faded = np.concatenate([(enc.mass() < float(L2.fade_frac)).flatten().cpu().numpy(), np.zeros(n_pass, bool)])
    parent_idx = np.array([split_idx[i // nC] for i in range(nP * nC)] + enc.pass_idx)
    kind = np.array(["child"] * (nP * nC) + ["pass"] * n_pass)
    np.savez("artifacts.npz", S=S_out, masks=masks, q=q_all, roles=np.array(roles), t_fit=np.int64(t_fit),
             p_long=np.concatenate([fin["p_long"].cpu().numpy(), pl_pass.cpu().numpy()]),
             g=np.concatenate([fin["g"].cpu().numpy(), g_pass.cpu().numpy()]),
             kind=kind, parent_idx=parent_idx, faded=faded, parent_label=np.array(parent_label),
             parent_S=par["S"], parent_masks=par["masks"], split_idx=np.array(split_idx),
             families=np.array([m["family"] for m in truth["modes"]]),
             truth_amp=np.stack([m["amp"] for m in truth["modes"]]),
             truth_phi=np.stack([m["phi"] for m in truth["modes"]]),
             truth_scale=np.array([m["scale"] for m in truth["modes"]]),
             q_snap=np.stack(q_snaps), masks_snap=np.stack(m_snaps), snap_steps=np.array(snap_steps))
    # check: the children of each split parent add up to the parent's signal (the partition loses nothing)
    rebuild = [np.corrcoef(S_out[:, i * nC:(i + 1) * nC].sum(1), par["S"][:, k])[0, 1] for i, k in enumerate(split_idx)]
    live = int((~faded[:nP * nC]).sum())
    summary = dict(n_split=nP, n_pass=n_pass, live_children=live,
                   child_roles=[r for r, f in zip(roles[:nP * nC], faded[:nP * nC]) if not f],
                   min_rebuild_corr=float(np.min(rebuild)) if rebuild else None)
    print(f"[layer2] done. {summary}")
    with open("metrics.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
