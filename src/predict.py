"""Fast prediction path. Crucially, this imports NO ``transformers`` and never
runs ESM-2 — it consumes cached protein vectors only (lazy ESM fallback exists
solely for a missing vector, which should never happen for the 64 test DBPs).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import (
    Config,
    DNAEncoderConfig,
    HeadConfig,
    ProteinEncoderConfig,
    TargetConfig,
)
from .data.dna import one_hot_batch
from .encoders.protein import load_embedding_cache, EmbeddingCache
from .model import BindingModel, build_model
from .utils import get_device


def _rebuild_config_for_arch(cfg: Config, ckpt_cfg: dict) -> Config:
    """Use the *checkpoint's* architecture but the live config's paths.

    This guarantees the model is rebuilt with the exact architecture it was
    trained with, even if ``config.yaml`` was edited afterwards.
    """
    arch = Config(
        seed=cfg.seed,
        paths=cfg.paths,
        protein_encoder=ProteinEncoderConfig(**ckpt_cfg["protein_encoder"]),
        dna_encoder=DNAEncoderConfig(**ckpt_cfg["dna_encoder"]),
        head=HeadConfig(**ckpt_cfg["head"]),
        target=TargetConfig(**ckpt_cfg["target"]),
        train=cfg.train,
        predict=cfg.predict,
    )
    return arch


def load_for_prediction(cfg: Config) -> Tuple[BindingModel, EmbeddingCache]:
    """Load the trained model and the embedding cache for fast scoring.

    Returns ``(model_in_eval_mode, embedding_cache)``. No ESM, no transformers.
    """
    device = get_device()
    ckpt = torch.load(cfg.paths.model_ckpt, map_location=device, weights_only=False)
    arch_cfg = _rebuild_config_for_arch(cfg, ckpt["config"])
    model = build_model(arch_cfg, protein_input_dim=ckpt.get("protein_input_dim"))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    cache = load_embedding_cache(cfg.paths.embeddings_cache)
    return model, cache


@torch.inference_mode()
def predict(model: BindingModel,
            protein_vec: torch.Tensor,
            dna_seqs: List[str],
            batch_size: int = 4096,
            device: Optional[torch.device] = None) -> List[float]:
    """Score every probe for one protein, returning floats in input order.

    The protein vector is projected once; probes are one-hot encoded and pushed
    through the DNA tower + head in batches (cheap, CPU-friendly).
    """
    device = device or get_device()
    model.eval()
    model.to(device)
    if not torch.is_tensor(protein_vec):
        protein_vec = torch.as_tensor(protein_vec, dtype=torch.float32)
    protein_vec = protein_vec.to(device).float()

    # Project the protein once and reuse across all probe batches.
    p = model.protein_encoder(protein_vec.unsqueeze(0))     # [1, proj_dim]

    onehot = one_hot_batch(dna_seqs)                         # [N, 4, L]
    scores: List[float] = []
    for start in range(0, len(dna_seqs), batch_size):
        batch = torch.from_numpy(onehot[start:start + batch_size]).to(device)
        d = model.encode_dna(batch)                          # [B, proj_dim]
        s = model.head(p.expand(d.shape[0], -1), d)          # [B]
        scores.extend(float(x) for x in s.cpu().numpy())
    return scores


def protein_vector_for_dbp(cache: EmbeddingCache, dbp_index: int) -> np.ndarray:
    """Return the cached ESM vector for a 0-based test DBP index."""
    if dbp_index < 0 or dbp_index >= cache.test_emb.shape[0]:
        raise IndexError(f"DBP index {dbp_index} out of range (have {cache.test_emb.shape[0]}).")
    return cache.test_emb[dbp_index]
