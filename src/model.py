"""``BindingModel`` — the orchestrator that composes the two towers and head.

It is deliberately thin: it routes the protein input through the protein
encoder, the DNA one-hot (and its reverse complement) through the *shared*
DNA encoder, then combines the two vectors with the interaction head. Swapping
any component is a config edit handled by :func:`build_model`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import Config
from .encoders.dna_cnn import build_dna_encoder
from .encoders.protein import build_protein_encoder
from .heads import build_head


def reverse_complement_onehot(onehot: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a one-hot batch ``[N, 4, L]`` (channels A,C,G,T).

    Flipping the channel axis maps A<->T and C<->G (the complement); flipping
    the length axis reverses the strand. Done together this is the RC.
    """
    return onehot.flip(dims=(1, 2))


class BindingModel(nn.Module):
    """Two-tower binding-intensity predictor.

    forward(protein_input, dna_onehot) -> [N] scores, where ``protein_input``
    is the cached ESM vectors ``[N, esm_dim]`` (default) or token ids for the
    learned encoder.
    """

    def __init__(self,
                 protein_encoder: nn.Module,
                 dna_encoder: nn.Module,
                 head: nn.Module,
                 use_reverse_complement: bool = True,
                 rc_combine: str = "mean"):
        super().__init__()
        if protein_encoder.output_dim != dna_encoder.output_dim:
            raise ValueError(
                f"Tower output dims must match: protein={protein_encoder.output_dim} "
                f"vs dna={dna_encoder.output_dim}"
            )
        self.protein_encoder = protein_encoder
        self.dna_encoder = dna_encoder
        self.head = head
        self.use_reverse_complement = use_reverse_complement
        self.rc_combine = rc_combine

    def encode_dna(self, dna_onehot: torch.Tensor) -> torch.Tensor:
        """Encode DNA, optionally averaging/maxing forward & reverse-complement.

        Both strands pass through the *same* weights (a protein may bind either
        strand, and the array reports only one).
        """
        d = self.dna_encoder(dna_onehot)
        if not self.use_reverse_complement:
            return d
        d_rc = self.dna_encoder(reverse_complement_onehot(dna_onehot))
        if self.rc_combine == "max":
            return torch.maximum(d, d_rc)
        return 0.5 * (d + d_rc)

    def forward(self, protein_input: torch.Tensor, dna_onehot: torch.Tensor) -> torch.Tensor:
        p = self.protein_encoder(protein_input)   # [N, proj_dim]
        d = self.encode_dna(dna_onehot)            # [N, proj_dim]
        return self.head(p, d)                     # [N]

    @torch.inference_mode()
    def predict_for_protein(self,
                            protein_vec: torch.Tensor,
                            dna_onehot_batch: torch.Tensor) -> torch.Tensor:
        """Score many probes for ONE protein. ``protein_vec``: [esm_dim].

        The protein vector is projected once and broadcast across probes, so
        the cost on the prediction path is dominated by the cheap DNA tower.
        """
        self.eval()
        p = self.protein_encoder(protein_vec.unsqueeze(0))     # [1, proj_dim]
        d = self.encode_dna(dna_onehot_batch)                  # [B, proj_dim]
        p = p.expand(d.shape[0], -1)                           # [B, proj_dim]
        return self.head(p, d)                                 # [B]


def build_model(cfg: Config, protein_input_dim: Optional[int] = None) -> BindingModel:
    """Assemble a ``BindingModel`` entirely from config.

    ``protein_input_dim`` is the cached ESM dimension (read from the embedding
    cache at train/predict time). For the ``learned`` encoder it is ignored.
    """
    protein_encoder = build_protein_encoder(cfg.protein_encoder, input_dim=protein_input_dim)
    dna_encoder = build_dna_encoder(cfg.dna_encoder)
    head = build_head(cfg.head, proj_dim=dna_encoder.output_dim)
    return BindingModel(
        protein_encoder=protein_encoder,
        dna_encoder=dna_encoder,
        head=head,
        use_reverse_complement=cfg.dna_encoder.use_reverse_complement,
        rc_combine=cfg.dna_encoder.rc_combine,
    )
