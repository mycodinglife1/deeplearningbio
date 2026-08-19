"""Score a predictions file against a ground-truth file (per-protein metrics).

The official metric is Pearson correlation between predicted and true binding
intensities for ONE protein, over all its probes (the course averages this
over the 64 test proteins). This tool scores one protein at a time:

    python scripts/score.py <pred_file> <true_file>

Both files hold one number per line, in the SAME probe order. It prints
Pearson (the graded metric), plus Spearman / MSE / R^2 as extra diagnostics.

IMPORTANT: the TEST-set true intensities are not provided to students — only
the course grader has them. So you can score against your own held-out
*training* probes, but not against the real test labels. The honest estimate
of test performance is the validation Pearson printed during training
(artifacts/train_log.json), which is 0.5906 for the shipped model.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import pearson_per_protein, spearman_per_protein, mse, r2_score


def _read_numbers(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        return np.array([float(x) for x in fh.read().split()], dtype=np.float64)


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python scripts/score.py <pred_file> <true_file>")
        sys.exit(1)
    pred_path, true_path = sys.argv[1], sys.argv[2]
    pred = _read_numbers(pred_path)
    true = _read_numbers(true_path)

    if pred.shape != true.shape:
        print(f"ERROR: length mismatch — pred has {pred.size} numbers, "
              f"true has {true.size}. They must be aligned line-for-line.")
        sys.exit(1)

    print(f"n probes            : {pred.size}")
    print(f"Pearson  (the metric): {pearson_per_protein(pred, true):.4f}")
    print(f"Spearman (rank check): {spearman_per_protein(pred, true):.4f}")
    print(f"MSE                 : {mse(pred, true):.4f}")
    print(f"R^2                 : {r2_score(pred, true):.4f}")


if __name__ == "__main__":
    main()
