"""DNA tower variants.

* ``DNAEncoder`` (default, "cnn") — a 1-D CNN motif scanner. A ``Conv1d`` over
  one-hot DNA is exactly a learnable Position Weight Matrix scanner; global
  max-pool asks "does this motif appear anywhere?" (DeepBind).
* ``DNATransformerEncoder`` ("cnn_transformer", idea F) — the CNN motif features
  are then passed through a small self-attention stack over the 36 positions,
  so the model can combine motifs that co-occur at a distance.
* ``MoEEncoder`` ("moe", idea E) — several CNN experts plus a gating network
  that weights them per input; experts can specialize to different motif
  families, which helps the out-of-distribution (unseen-protein) regime.

All encoders expose the SAME interface: ``forward(onehot) -> [N, proj_dim]`` and
``position_features(onehot) -> [N, L, feat_dim]`` (the per-position features used
by the cross-attention head, idea B). ``DNAEncoder``'s parameters are unchanged
so existing checkpoints keep loading.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DNAEncoder(nn.Module):
    """One-hot ``[N, 4, L]`` -> a ``proj_dim`` DNA vector (parallel multi-width CNN)."""

    def __init__(self,
                 conv_channels: int = 128,
                 kernel_sizes: List[int] | None = None,
                 proj_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        kernel_sizes = list(kernel_sizes or [15])
        self.kernel_sizes = kernel_sizes
        self._proj_dim = proj_dim
        self.conv_channels = conv_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=4, out_channels=conv_channels,
                      kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        concat_dim = conv_channels * len(kernel_sizes)
        self.project = nn.Sequential(
            nn.Linear(concat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def output_dim(self) -> int:
        return self._proj_dim

    @property
    def position_dim(self) -> int:
        """Feature dim of the per-position motif map (before pooling)."""
        return self.conv_channels * len(self.kernel_sizes)

    def _conv_map(self, onehot: torch.Tensor) -> torch.Tensor:
        """Per-position conv activations concatenated over kernels: [N, L, C_total]."""
        maps = [torch.relu(conv(onehot)) for conv in self.convs]  # each [N, C, L]
        # Kernels use padding=k//2; even kernels can add one extra position, so
        # crop all branches to the shortest length before concatenating.
        min_len = min(m.shape[-1] for m in maps)
        maps = [m[..., :min_len] for m in maps]
        concat = torch.cat(maps, dim=1)                           # [N, C_total, L]
        return concat.transpose(1, 2)                             # [N, L, C_total]

    def position_features(self, onehot: torch.Tensor) -> torch.Tensor:
        """[N, 4, L] -> [N, L, C_total] per-position features (for cross-attention)."""
        return self._conv_map(onehot)

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """[N, 4, L] -> [N, proj_dim]."""
        feats = []
        for conv in self.convs:
            activated = torch.relu(conv(onehot))       # [N, C, L]
            pooled = self.pool(activated).squeeze(-1)  # [N, C]
            feats.append(pooled)
        concat = torch.cat(feats, dim=1)               # [N, C * num_kernels]
        return self.project(concat)                    # [N, proj_dim]


class DNATransformerEncoder(nn.Module):
    """CNN motif features -> small Transformer over positions -> pooled vector (F).

    The CNN finds motifs; the self-attention lets positions exchange information
    so the model can reason about motif combinations/spacing, not just presence.
    """

    def __init__(self,
                 conv_channels: int = 128,
                 kernel_sizes: List[int] | None = None,
                 proj_dim: int = 128,
                 dropout: float = 0.2,
                 n_layers: int = 1,
                 n_heads: int = 4,
                 max_positions: int = 40):
        super().__init__()
        kernel_sizes = list(kernel_sizes or [15])
        self._proj_dim = proj_dim
        self.convs = nn.ModuleList([
            nn.Conv1d(4, conv_channels, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        concat_dim = conv_channels * len(kernel_sizes)
        self.to_model = nn.Linear(concat_dim, proj_dim)          # per-position -> d_model
        self.pos_emb = nn.Embedding(max_positions, proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=n_heads, dim_feedforward=proj_dim * 2,
            dropout=dropout, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    @property
    def output_dim(self) -> int:
        return self._proj_dim

    @property
    def position_dim(self) -> int:
        return self._proj_dim

    def _encode(self, onehot: torch.Tensor) -> torch.Tensor:
        """[N, 4, L] -> [N, L, proj_dim] contextualized per-position features."""
        maps = [torch.relu(conv(onehot)) for conv in self.convs]
        min_len = min(m.shape[-1] for m in maps)
        maps = [m[..., :min_len] for m in maps]
        x = torch.cat(maps, dim=1).transpose(1, 2)                # [N, L, C_total]
        x = self.to_model(x)                                      # [N, L, d]
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(positions)[None, :, :]
        return self.transformer(x)                                # [N, L, d]

    def position_features(self, onehot: torch.Tensor) -> torch.Tensor:
        return self._encode(onehot)

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        x = self._encode(onehot)                                  # [N, L, d]
        pooled = x.max(dim=1).values                              # best-over-positions
        return self.dropout(pooled)


class MoEEncoder(nn.Module):
    """Mixture of CNN experts with a per-input gating network (E).

    Each expert is a full CNN motif scanner; a gate (softmax over experts,
    conditioned on the probe) mixes their outputs, so experts can specialize.
    """

    def __init__(self,
                 conv_channels: int = 128,
                 kernel_sizes: List[int] | None = None,
                 proj_dim: int = 128,
                 dropout: float = 0.2,
                 n_experts: int = 4):
        super().__init__()
        self._proj_dim = proj_dim
        self.experts = nn.ModuleList([
            DNAEncoder(conv_channels, kernel_sizes, proj_dim, dropout) for _ in range(n_experts)
        ])
        # Gate: cheap global summary of the probe -> weights over experts.
        self.gate_conv = nn.Conv1d(4, conv_channels, kernel_size=9, padding=4)
        self.gate_fc = nn.Linear(conv_channels, n_experts)

    @property
    def output_dim(self) -> int:
        return self._proj_dim

    @property
    def position_dim(self) -> int:
        return self.experts[0].position_dim

    def _gate(self, onehot: torch.Tensor) -> torch.Tensor:
        g = torch.relu(self.gate_conv(onehot)).max(dim=-1).values  # [N, C]
        return torch.softmax(self.gate_fc(g), dim=-1)              # [N, n_experts]

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        weights = self._gate(onehot)                              # [N, E]
        outs = torch.stack([e(onehot) for e in self.experts], dim=1)  # [N, E, proj]
        return (weights.unsqueeze(-1) * outs).sum(dim=1)          # [N, proj]

    def position_features(self, onehot: torch.Tensor) -> torch.Tensor:
        weights = self._gate(onehot)                              # [N, E]
        pos = torch.stack([e.position_features(onehot) for e in self.experts], dim=1)  # [N,E,L,C]
        return (weights[:, :, None, None] * pos).sum(dim=1)       # [N, L, C]


def build_dna_encoder(cfg) -> nn.Module:
    """Factory dispatching on ``dna_encoder.type``."""
    kind = getattr(cfg, "type", "cnn")
    if kind == "cnn":
        return DNAEncoder(cfg.conv_channels, cfg.kernel_sizes, cfg.proj_dim, cfg.dropout)
    if kind == "cnn_transformer":
        return DNATransformerEncoder(
            cfg.conv_channels, cfg.kernel_sizes, cfg.proj_dim, cfg.dropout,
            n_layers=cfg.transformer_layers, n_heads=cfg.transformer_heads)
    if kind == "moe":
        return MoEEncoder(cfg.conv_channels, cfg.kernel_sizes, cfg.proj_dim,
                          cfg.dropout, n_experts=cfg.n_experts)
    raise ValueError(f"Unknown dna_encoder.type={kind!r}")
