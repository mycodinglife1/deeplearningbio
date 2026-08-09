"""Thin CLI wrapper around ``src.train.train``.

Usage:
    python scripts/train.py                       # full training from config
    python scripts/train.py --epochs 2 --max-proteins 20   # quick smoke run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the PBM binding predictor.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override config epochs (for quick runs).")
    ap.add_argument("--max-proteins", type=int, default=None,
                    help="Cap number of training proteins (for quick smoke runs).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = train(cfg, epochs=args.epochs, max_train_proteins=args.max_proteins)
    print(f"\nBest validation mean per-protein Pearson: {result['best_val_pearson']:.4f}")
    print(f"(baseline to beat: ~0.208)")


if __name__ == "__main__":
    main()
