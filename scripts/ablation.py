"""Ablation harness: run the default config minus one feature at a time and
tabulate the change in validation mean per-protein Pearson.

Each row trains a variant and reports its best val Pearson and the delta vs the
default. This produces the ablation table for the report directly.

Usage:
    python scripts/ablation.py --epochs 10            # all ablations, 10 epochs each
    python scripts/ablation.py --epochs 6 --only baseline no_reverse_complement
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.train import train


def _variants():
    """Return {name: mutator(cfg)} — each toggles ONE feature off/changes it."""
    def mut(fn):
        return fn

    return {
        "baseline":               mut(lambda c: c),
        "no_reverse_complement":  mut(lambda c: _set(c, "dna_encoder", "use_reverse_complement", False)),
        "head_concat_only":       mut(lambda c: _set(c, "head", "type", "concat")),
        "head_bilinear":          mut(lambda c: _set(c, "head", "type", "bilinear")),
        "no_log1p":               mut(lambda c: _set(c, "target", "log1p", False)),
        "no_zscore":              mut(lambda c: _set(c, "target", "per_protein_zscore", False)),
        "single_kernel_15":       mut(lambda c: _set(c, "dna_encoder", "kernel_sizes", [15])),
        "loss_huber":             mut(lambda c: _set(c, "train", "loss", "huber")),
    }


def _set(cfg, section, field, value):
    setattr(getattr(cfg, section), field, value)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Run feature ablations.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=10,
                    help="Epochs per variant (keep small; this trains many models).")
    ap.add_argument("--max-proteins", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Subset of variant names to run.")
    args = ap.parse_args()

    variants = _variants()
    names = args.only if args.only else list(variants)

    results = {}
    for name in names:
        if name not in variants:
            print(f"[skip] unknown variant {name!r}")
            continue
        cfg = load_config(args.config, warn_missing_files=False)
        cfg = variants[name](cfg)
        # Point each run's checkpoint somewhere disposable so we don't clobber
        # the real model.pt.
        cfg.paths.model_ckpt = os.path.join(cfg.paths.artifacts_dir, f"ablation_{name}.pt")
        print(f"\n===== ablation: {name} =====")
        res = train(cfg, epochs=args.epochs, max_train_proteins=args.max_proteins, quiet=False)
        results[name] = res["best_val_pearson"]

    base = results.get("baseline")
    print("\n================ ABLATION TABLE ================")
    print(f"{'variant':<26}{'val_pearson':>12}{'delta':>10}")
    for name in names:
        if name not in results:
            continue
        val = results[name]
        delta = "" if base is None else f"{val - base:+.4f}"
        print(f"{name:<26}{val:>12.4f}{delta:>10}")


if __name__ == "__main__":
    main()
