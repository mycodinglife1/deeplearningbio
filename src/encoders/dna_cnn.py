"""DNA tower: a 1-D CNN motif scanner over one-hot probes.

A ``Conv1d`` over one-hot DNA is exactly a learnable Position Weight Matrix
scanner: each filter's weights are per-position, per-base scores, and the
convolution computes the PWM match at every offset. Global max-pool then asks
"does this motif appear anywhere, and how strongly?" — the DeepBind idea, and
the neural-network version of the PWM picture from the course slides.

Multiple kernel widths in parallel capture motifs of different lengths.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class DNAEncoder(nn.Module):
    """One-hot ``[N, 4, L]`` -> a ``proj_dim`` DNA vector.

    Parallel convolutions (one per kernel width) each followed by ReLU and
    global max-pool over the sequence; their outputs are concatenated and
    projected to the shared embedding dimension.
    """

    def __init__(self,
                 conv_channels: int = 128,
                 kernel_sizes: List[int] | None = None,
                 proj_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        kernel_sizes = list(kernel_sizes or [15])
        self.kernel_sizes = kernel_sizes
        self._proj_dim = proj_dim

        # One conv branch per kernel width. padding=k//2 keeps the length so
        # short motifs near the probe ends are still scannable.
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

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        """``onehot``: [N, 4, L] -> [N, proj_dim]."""
        feats = []
        for conv in self.convs:
            activated = torch.relu(conv(onehot))      # [N, C, L]
            pooled = self.pool(activated).squeeze(-1)  # [N, C]
            feats.append(pooled)
        concat = torch.cat(feats, dim=1)               # [N, C * num_kernels]
        return self.project(concat)                    # [N, proj_dim]


def build_dna_encoder(cfg) -> DNAEncoder:
    """Factory from the ``dna_encoder`` config section."""
    return DNAEncoder(
        conv_channels=cfg.conv_channels,
        kernel_sizes=cfg.kernel_sizes,
        proj_dim=cfg.proj_dim,
        dropout=cfg.dropout,
    )
