"""Fast deploy of the core_B+C winner (cross-attention + zero-shot regularization)
on ALL training data, then regenerate the submission.

Trained on ~all proteins (tiny val holdout only for best-epoch selection), for a
small number of epochs near the sweep's best epoch, with reduced probes/epoch so
it finishes inside the ~1h window. Saves artifacts/model.pt and rewrites the 64
DBP score files + submission.zip.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.encoders.protein import load_embedding_cache
from src.train import train

t0 = time.time()
cfg = load_config("config.yaml", warn_missing_files=False)

# --- core_B+C architecture + zero-shot regularization ---
cfg.head.type = "cross_attention"
cache = load_embedding_cache(cfg.paths.embeddings_cache)
cfg.train.protein_noise_std = round(0.1 * float(cache.train_emb.std()), 4)
cfg.train.protein_mask_prob = 0.1

# --- final training on ALL proteins (no holdout), fixed epochs -------------
# Every one of the 387 unique proteins trains. Epoch count is chosen from the
# validated sweep (core peaks ~epoch 7-11). Probes are subsampled per epoch
# (~1400 of 30000) only because cross-attention is too slow to see all probes
# within the time window; each epoch samples a fresh random subset.
cfg.train.epochs = 11                       # ~the sweep's peak epoch for core_B+C
cfg.train.probes_per_protein_per_epoch = 2400
cfg.train.val_protein_fraction = 0.0       # 0 => train on ALL proteins, keep last epoch

print(f"[deploy] core_B+C | noise_std={cfg.train.protein_noise_std} | "
      f"epochs={cfg.train.epochs} probes/epoch={cfg.train.probes_per_protein_per_epoch} "
      f"| ALL proteins")
res = train(cfg)
print(f"[deploy] trained ({(time.time()-t0)/60:.1f} min).")

print("[deploy] regenerating submission...")
subprocess.run([sys.executable, "scripts/predict_all.py"], check=False)
print(f"[deploy] DONE in {(time.time()-t0)/60:.1f} min. model.pt + submission updated.")
