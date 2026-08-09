"""Inspect every provided data file and the training_data.zip layout.

Run this FIRST. It prints line counts, length ranges and alphabets for the
sequence files, and unzips + summarizes the intensity matrix so the format is
confirmed before any reader code relies on it.

Usage:
    python scripts/inspect_data.py
"""

from __future__ import annotations

import os
import sys
import zipfile
from collections import Counter

# Make `src` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config


def _summarize_sequences(path: str, label: str) -> None:
    if not os.path.exists(path):
        print(f"  [{label}] MISSING: {path}")
        return
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    lengths = [len(ln) for ln in lines]
    alphabet = sorted(set("".join(lines[:200])))  # sample to keep it cheap
    print(f"  [{label}] {path}")
    print(f"      lines={len(lines)}  len_range=({min(lengths)}, {max(lengths)})  "
          f"unique={len(set(lines))}  alphabet(sample)={''.join(alphabet)}")


def _summarize_intensities(zip_path: str) -> None:
    print(f"\n=== training_data.zip ===")
    if not os.path.exists(zip_path):
        print(f"  MISSING: {zip_path}\n  (Cannot train without intensities — add it to ./data.)")
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        print(f"  archive entries: {names}")
        inner = names[0]
        with zf.open(inner) as fh:
            first = fh.readline().decode().split()
            n_cols_first = len(first)
            n_rows = 1
            col_counts = Counter([n_cols_first])
            for raw in fh:
                n_rows += 1
                col_counts[len(raw.split())] += 1
    print(f"  inner file: {inner}")
    print(f"  rows (lines)     = {n_rows}")
    print(f"  cols per row     = {dict(col_counts)}  (consistent if single key)")
    print(f"  sample values    = {[float(x) for x in first[:5]]}")
    print(f"\n  INTERPRETATION: matrix is [rows x cols] = [n_probes x n_proteins].")
    print(f"  io.load_intensities transposes -> [n_proteins, n_probes] aligned to")
    print(f"  training_DBPs.txt (proteins) and training_seqs.txt (probes).")


def main() -> None:
    cfg = load_config("config.yaml")
    print("=== config loaded OK ===")
    print(f"  protein_encoder.type = {cfg.protein_encoder.type} ({cfg.protein_encoder.esm_model})")
    print(f"  dna kernel_sizes     = {cfg.dna_encoder.kernel_sizes}, RC={cfg.dna_encoder.use_reverse_complement}")
    print(f"  head.type            = {cfg.head.type}")

    print("\n=== sequence files ===")
    _summarize_sequences(cfg.paths.train_dbps, "train_DBPs")
    _summarize_sequences(cfg.paths.train_seqs, "train_seqs")
    _summarize_sequences(cfg.paths.test_dbps, "test_DBPs")
    _summarize_sequences(cfg.paths.test_seqs, "test_seqs")

    _summarize_intensities(cfg.paths.train_intensities)

    baseline = os.path.join(cfg.paths.data_dir, "baseline_results.txt")
    if os.path.exists(baseline):
        with open(baseline) as fh:
            vals = [float(x) for x in fh.read().split()]
        print(f"\n=== baseline_results.txt ===")
        print(f"  n={len(vals)}  mean={sum(vals)/len(vals):.4f}  "
              f"min={min(vals):.3f}  max={max(vals):.3f}   (the bar to beat: ~0.208)")


if __name__ == "__main__":
    main()
