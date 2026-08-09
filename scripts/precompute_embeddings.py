"""Run ESM-2 once over all train + test proteins and cache the pooled vectors.

This is the only step that needs ``transformers`` / the ESM weights. The
resulting ``artifacts/protein_embeddings.npz`` is what training and prediction
consume, so ESM never runs on the (graded) prediction path.

Usage:
    python scripts/precompute_embeddings.py [--config config.yaml] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import io
from src.encoders.protein import (
    ESM_OUTPUT_DIMS,
    ESM2Encoder,
    save_embedding_cache,
)
from src.utils import get_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute ESM-2 protein embeddings.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="ESM forward batch size (lower if memory-constrained).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()

    train_proteins = io.read_proteins(cfg.paths.train_dbps)
    test_proteins = io.read_proteins(cfg.paths.test_dbps)
    print(f"Train proteins: {len(train_proteins)}  |  Test proteins: {len(test_proteins)}")

    esm_model = cfg.protein_encoder.esm_model
    input_dim = ESM_OUTPUT_DIMS.get(esm_model)
    encoder = ESM2Encoder(
        input_dim=input_dim if input_dim is not None else 480,
        proj_dim=cfg.protein_encoder.proj_dim,
        esm_model=esm_model,
        pooling=cfg.protein_encoder.pooling,
        max_len=cfg.protein_encoder.max_len,
    )

    print(f"Loading ESM model {esm_model!r} on {device} (downloads on first run)...")
    t0 = time.time()
    train_emb = encoder.precompute(train_proteins, batch_size=args.batch_size, device=device)
    test_emb = encoder.precompute(test_proteins, batch_size=args.batch_size, device=device)
    print(f"ESM embeddings computed in {time.time() - t0:.1f}s. "
          f"train_emb={train_emb.shape}, test_emb={test_emb.shape}")

    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    save_embedding_cache(cfg.paths.embeddings_cache, train_emb, test_emb,
                         train_proteins, test_proteins, esm_model)
    print(f"Saved embedding cache -> {cfg.paths.embeddings_cache}")


if __name__ == "__main__":
    main()
