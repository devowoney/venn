"""Layer-2 children (D-039, "sharpen sight"): each non-cyclic layer-1 eye is split into n_child sub-eyes.

The children are a soft PARTITION of their parent's cells:
    child_c = parent_mask * softmax_c(logits_c / temp)          c = 1..n_child, per cell
so they always add up to the parent. WHY a partition: "the children rebuild their parent" then holds by
construction (nothing layer 1 saw is lost), and a child with nothing distinct to carry can give all its cells away
("decompose into 4 families IF POSSIBLE"). Still pure selection (D-005): no signed weights.
Cyclic parents are not split; they pass through as their own channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import multiscale_logits


class ChildPartition(nn.Module):
    """parent_masks [K,V,H,W] (frozen) -> outputs: children of the split parents, then the pass-through parents."""

    def __init__(self, parent_masks: torch.Tensor, split_idx: list, n_child: int = 4, temp: float = 1.0,
                 init_std: float = 1.5, sigma: float = 3.0, generator: torch.Generator | None = None):
        super().__init__()
        K, V, H, W = parent_masks.shape
        self.register_buffer("parent", parent_masks.clone())
        self.split_idx = list(split_idx)
        self.pass_idx = [k for k in range(K) if k not in set(self.split_idx)]
        self.n_child, self.temp = int(n_child), float(temp)
        n = len(self.split_idx) * self.n_child
        # independent smooth random logits per child, so siblings start as DIFFERENT random parts of the parent
        w = multiscale_logits(max(n, 1), V, H, W, init_std, sigma, sigma, generator)[:n]
        self.logits = nn.Parameter(w.view(len(self.split_idx), self.n_child, V, H, W))

    def shares(self) -> torch.Tensor:
        """[n_split, n_child, V, H, W]: which child each parent cell goes to (sums to 1 over children)."""
        return torch.softmax(self.logits / self.temp, dim=1)

    def child_masks(self) -> torch.Tensor:
        """[n_split*n_child, V, H, W], ordered parent by parent."""
        p = self.parent[self.split_idx]                         # [n_split, V, H, W]
        return (p[:, None] * self.shares()).flatten(0, 1)

    def masks(self) -> torch.Tensor:
        """All layer-2 output masks: children first, then the pass-through parents."""
        return torch.cat([self.child_masks(), self.parent[self.pass_idx]], dim=0)

    def mass(self) -> torch.Tensor:
        """[n_split, n_child]: each child's share of its parent's selected mass (0..1, sums to 1 per parent)."""
        return self.shares().mul(self.parent[self.split_idx][:, None]).sum(dim=(2, 3, 4)) / \
            self.parent[self.split_idx].sum(dim=(1, 2, 3))[:, None].clamp_min(1e-6)

    def forward(self, field: torch.Tensor, children_only: bool = False) -> torch.Tensor:
        """field [T,V,H,W] -> [T, n_out] selection sums (children first, then pass-through)."""
        m = self.child_masks() if children_only else self.masks()
        return torch.einsum("tvhw,kvhw->tk", field, m)
