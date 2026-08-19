"""Best-effort precompute of ESM-DBP embeddings (idea A).

ESM-DBP is ESM2-650M domain-adapted on DNA-binding proteins. Its weights ship as
a single 2.6 GB ``ESM-DBP.model`` on HuggingFace (zengwenwu/ESM-DBP), loadable
with Facebook's ``fair-esm`` library. This script downloads it, extracts
mean-pooled per-protein embeddings for the train + test proteins, and saves them
in our standard .npz cache format.

It is deliberately defensive: any failure (missing lib, download error, unknown
checkpoint format, OOM) exits non-zero so the orchestrator simply skips ESM-DBP.

    python scripts/precompute_esm_dbp.py --out artifacts/protein_embeddings_esmdbp.npz
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.io import read_proteins
from src.encoders.protein import save_embedding_cache

REPO = "zengwenwu/ESM-DBP"
MODEL_FILE = "ESM-DBP.model"
LAST_LAYER = 33          # esm2_t33_650M
MAX_LEN = 1022


def _ensure(pkg, pip_name=None):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        pip_name or pkg], check=True)
        __import__(pkg)


def _load_esm_dbp(path):
    """Load the ESM-DBP checkpoint into a fair-esm model + alphabet."""
    import esm
    try:
        return esm.pretrained.load_model_and_alphabet_local(path)
    except Exception:
        # Fall back: raw checkpoint dict -> core loader.
        ckpt = torch.load(path, map_location="cpu")
        return esm.pretrained.load_model_and_alphabet_core("ESM-DBP", ckpt, None)


@torch.inference_mode()
def _embed(model, alphabet, seqs, batch_size=2):
    bc = alphabet.get_batch_converter()
    model.eval()
    out = np.zeros((len(seqs), model.embed_dim if hasattr(model, "embed_dim") else 1280),
                   dtype=np.float32)
    for start in range(0, len(seqs), batch_size):
        chunk = [(str(i), s[:MAX_LEN]) for i, s in enumerate(seqs[start:start + batch_size])]
        _, _, toks = bc(chunk)
        rep = model(toks, repr_layers=[LAST_LAYER])["representations"][LAST_LAYER]  # [B,T,D]
        for j, (_, s) in enumerate(chunk):
            # fair-esm: token 0 is BOS, then residues 1..L, then EOS -> mean 1..L.
            L = min(len(s), MAX_LEN)
            out[start + j] = rep[j, 1:L + 1].mean(0).numpy()
        print(f"\r  ESM-DBP: {min(start + batch_size, len(seqs))}/{len(seqs)}", end="", flush=True)
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/protein_embeddings_esmdbp.npz")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    _ensure("esm", "fair-esm")
    _ensure("huggingface_hub")
    from huggingface_hub import hf_hub_download

    cfg = load_config(args.config, warn_missing_files=False)
    train_seqs = read_proteins(cfg.paths.train_dbps)
    test_seqs = read_proteins(cfg.paths.test_dbps)

    print(f"Downloading {REPO}/{MODEL_FILE} (2.6 GB, one-time)...")
    path = hf_hub_download(repo_id=REPO, filename=MODEL_FILE)
    print("Loading ESM-DBP...")
    model, alphabet = _load_esm_dbp(path)

    print(f"Embedding {len(train_seqs)} train + {len(test_seqs)} test proteins...")
    train_emb = _embed(model, alphabet, train_seqs)
    test_emb = _embed(model, alphabet, test_seqs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_embedding_cache(args.out, train_emb, test_emb, train_seqs, test_seqs,
                         esm_model="ESM-DBP(esm2_t33_650M-DBP)")
    print(f"Saved -> {args.out}  train={train_emb.shape} test={test_emb.shape}")


if __name__ == "__main__":
    main()
