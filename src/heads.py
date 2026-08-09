"""Interaction heads: how the protein vector ``p`` and DNA vector ``d`` meet.

The head must let ``p`` and ``d`` *interact* (not just sit side by side); a
model that emits a per-protein constant scores 0 under Pearson, so the head is
deliberately built to use the DNA. The default (``concat_product``) adds the
element-wise product ``p ⊙ d`` — the "do these match?" signal, in the spirit
of a factorization machine.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatHead(nn.Module):
    """``[p, d] -> MLP -> 1`` (simplest; good ablation baseline)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = _mlp(2 * dim, hidden_dim, dropout)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([p, d], dim=1)).squeeze(-1)


class ConcatProductHead(nn.Module):
    """``[p, d, p ⊙ d] -> MLP -> 1`` (default). The product forces interaction."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = _mlp(3 * dim, hidden_dim, dropout)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([p, d, p * d], dim=1)
        return self.mlp(combined).squeeze(-1)


class BilinearHead(nn.Module):
    """``pᵀ W d`` plus a small additive MLP on ``[p, d]`` for stability.

    Closest to Affinity Regression's protein × DNA interaction term.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)
        self.mlp = _mlp(2 * dim, hidden_dim, dropout)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (self.bilinear(p, d) + self.mlp(torch.cat([p, d], dim=1))).squeeze(-1)


class FiLMHead(nn.Module):
    """``p`` predicts a scale+shift that modulates ``d``, then ``-> 1``.

    FiLM lets the protein condition the DNA features multiplicatively.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.to_film = nn.Linear(dim, 2 * dim)   # produces (gamma, beta)
        self.mlp = _mlp(dim, hidden_dim, dropout)

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_film(p).chunk(2, dim=1)
        modulated = gamma * d + beta
        return self.mlp(modulated).squeeze(-1)


def _mlp(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Shared 2-layer MLP: ``in_dim -> hidden -> 1`` with ReLU + dropout."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


_HEADS = {
    "concat": ConcatHead,
    "concat_product": ConcatProductHead,
    "bilinear": BilinearHead,
    "film": FiLMHead,
}


def build_head(cfg, proj_dim: int) -> nn.Module:
    """Factory selecting the head class by ``head.type``. Output shape ``[N]``."""
    if cfg.type not in _HEADS:
        raise ValueError(f"Unknown head.type={cfg.type!r}; choices={list(_HEADS)}")
    return _HEADS[cfg.type](dim=proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
