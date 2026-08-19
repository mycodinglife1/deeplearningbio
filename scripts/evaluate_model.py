"""Evaluate the trained model on held-out proteins and report the DISTRIBUTION
of per-protein Pearson (mean / median / spread + a text histogram).

This is the number the course grades on (mean per-protein Pearson), computed on
the protein-disjoint validation split — proteins the model never trained on, so
it is an honest estimate of test performance. Run it any time without having to
re-produce prediction files by hand:

    python scripts/evaluate_model.py                # held-out validation proteins
    python scripts/evaluate_model.py --split train  # proteins the model trained on
    python scripts/evaluate_model.py --split all
    python scripts/evaluate_model.py --per-protein  # also print each protein's score
    python scripts/evaluate_model.py --csv out.csv  # save per-protein table

Note: the real TEST-set labels are not provided, so the distribution here is
over TRAINING proteins (held out or not). The default (--split val) mirrors the
grader's setup: unseen proteins.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.io import read_proteins, read_sequences
from src.data.dna import one_hot_batch
from src.data.dataset import protein_disjoint_split
from src.train import deduplicate_proteins
from src.predict import load_for_prediction
from src.encoders.protein import load_embedding_cache
from src.evaluate import pearson_per_protein, spearman_per_protein


def _histogram(values: np.ndarray, bins=(-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> str:
    """A small text histogram of the per-protein Pearson values."""
    lines = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = int(np.sum((values >= lo) & (values < hi)))
        # include the top edge in the last bucket
        if hi == bins[-1]:
            count = int(np.sum((values >= lo) & (values <= hi)))
        bar = "#" * count
        lines.append(f"  [{lo:+.1f}, {hi:+.1f})  {count:3d} |{bar}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-protein Pearson distribution of the trained model.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="which proteins to evaluate (default: held-out validation).")
    ap.add_argument("--per-protein", action="store_true", help="print each protein's Pearson.")
    ap.add_argument("--csv", default=None, help="optional path to save a per-protein CSV.")
    args = ap.parse_args()

    cfg = load_config(args.config, warn_missing_files=False)
    proteins = read_proteins(cfg.paths.train_dbps)
    seqs = read_sequences(cfg.paths.train_seqs)
    intens = np.load(os.path.join(cfg.paths.artifacts_dir, "intensities.npy"))

    uniq_seqs, uniq_raw, rep_idx = deduplicate_proteins(proteins, intens)
    train_ids, val_ids = protein_disjoint_split(
        len(uniq_seqs), cfg.train.val_protein_fraction, cfg.seed)

    if args.split == "val":
        ids = val_ids
    elif args.split == "train":
        ids = train_ids
    else:
        ids = np.sort(np.concatenate([train_ids, val_ids]))

    cache = load_embedding_cache(cfg.paths.embeddings_cache)
    model, _ = load_for_prediction(cfg)
    onehot = torch.from_numpy(one_hot_batch(seqs))

    print(f"Evaluating {len(ids)} '{args.split}' proteins x {len(seqs)} probes "
          f"(model: {cfg.protein_encoder.esm_model})...")

    rows = []
    with torch.inference_mode():
        for u in ids:
            u = int(u)
            emb = torch.from_numpy(np.asarray(cache.train_emb[rep_idx[u]]))
            pred = model.predict_for_protein(emb, onehot).numpy()
            pr = pearson_per_protein(pred, uniq_raw[u])
            sp = spearman_per_protein(pred, uniq_raw[u])
            rows.append((u, rep_idx[u] + 1, pr, sp))  # +1 -> 1-based training line

    pear = np.array([r[2] for r in rows])
    spear = np.array([r[3] for r in rows])

    print("\n================ per-protein Pearson distribution ================")
    print(f"  proteins           : {len(pear)}")
    print(f"  MEAN   (the score) : {pear.mean():.4f}")
    print(f"  median             : {np.median(pear):.4f}")
    print(f"  std                : {pear.std():.4f}")
    print(f"  min / max          : {pear.min():.4f} / {pear.max():.4f}")
    print(f"  mean Spearman      : {spear.mean():.4f}")
    print(f"  >=0.6 : {(pear>=0.6).sum()}   >=0.5 : {(pear>=0.5).sum()}   "
          f"<0.3 : {(pear<0.3).sum()}")
    print("\n  histogram (per-protein Pearson):")
    print(_histogram(pear))

    if args.per_protein:
        print("\n  per-protein (training line -> Pearson, Spearman):")
        for _, line, pr, sp in sorted(rows, key=lambda r: -r[2]):
            print(f"    DBP@line {line:4d} : Pearson {pr:+.4f}  Spearman {sp:+.4f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["unique_id", "training_line", "pearson", "spearman"])
            for u, line, pr, sp in rows:
                w.writerow([u, line, f"{pr:.6f}", f"{sp:.6f}"])
        print(f"\n  wrote per-protein CSV -> {args.csv}")


if __name__ == "__main__":
    main()
