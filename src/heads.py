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

    # Heads that consume the *pooled* DNA vector set this False; the
    # cross-attention head needs the per-position DNA map instead.
    needs_positions = False

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


class CrossAttentionHead(nn.Module):
    """Protein attends to the per-position DNA map, then predicts a score (B).

    The protein vector ``p`` is the query; the DNA position features are the
    keys/values, so ``p`` selectively "scans" the probe for the sites it cares
    about (the TransBind idea). The attended DNA summary is combined with ``p``
    via the same concat+product interaction before the final MLP.
    """

    needs_positions = True

    def __init__(self, dim: int, hidden_dim: int, dropout: float,
                 kv_dim: int, attn_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=attn_heads,
                                          dropout=dropout, batch_first=True)
        self.mlp = _mlp(3 * dim, hidden_dim, dropout)

    def forward(self, p: torch.Tensor, d_positions: torch.Tensor) -> torch.Tensor:
        # p: [N, dim];  d_positions: [N, L, kv_dim]
        q = self.q_proj(p).unsqueeze(1)                 # [N, 1, dim]
        k = self.k_proj(d_positions)                    # [N, L, dim]
        v = self.v_proj(d_positions)                    # [N, L, dim]
        attended, _ = self.attn(q, k, v)                # [N, 1, dim]
        d = attended.squeeze(1)                         # [N, dim]
        combined = torch.cat([p, d, p * d], dim=1)      # [N, 3*dim]
        return self.mlp(combined).squeeze(-1)


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


def build_head(cfg, proj_dim: int, kv_dim: int | None = None) -> nn.Module:
    """Factory selecting the head class by ``head.type``. Output shape ``[N]``.

    ``kv_dim`` is the DNA per-position feature dim, required only by the
    cross-attention head.
    """
    if cfg.type == "cross_attention":
        if kv_dim is None:
            raise ValueError("cross_attention head requires kv_dim (DNA position_dim).")
        return CrossAttentionHead(dim=proj_dim, hidden_dim=cfg.hidden_dim,
                                  dropout=cfg.dropout, kv_dim=kv_dim,
                                  attn_heads=cfg.attn_heads)
    if cfg.type not in _HEADS:
        raise ValueError(f"Unknown head.type={cfg.type!r}; choices={list(_HEADS) + ['cross_attention']}")
    return _HEADS[cfg.type](dim=proj_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
