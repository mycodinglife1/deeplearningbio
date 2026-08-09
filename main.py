"""Graded entrypoint.

    python main.py <output_file> <DBP_name> <DNA_probe_file>

  * <DBP_name>      one of DBP1..DBP64 (the 1-based line in test_DBPs.txt).
  * <DNA_probe_file> one 36-bp ACGT sequence per line.
  * <output_file>   one predicted intensity per line, in input order.

Speed contract: this path must be fast (<= 600 s for all 64 DBPs). It loads a
few-MB model + a tiny embedding cache and runs only the DNA tower + head. ESM-2
/ transformers are NOT imported here.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``src`` importable when invoked as ``python main.py ...``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from src.config import load_config
from src.data.io import dbp_name_to_index, read_sequences
from src.predict import load_for_prediction, predict, protein_vector_for_dbp


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict PBM binding intensities for one DBP.")
    ap.add_argument("output_file", help="Where to write one score per line.")
    ap.add_argument("dbp_name", help="DBP name, e.g. DBP1 .. DBP64.")
    ap.add_argument("dna_probe_file", help="File with one 36-bp ACGT sequence per line.")
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: alongside main.py).")
    args = ap.parse_args()

    config_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    cfg = load_config(config_path, warn_missing_files=False)

    # Load model + cached protein vectors (no ESM on this path).
    model, cache = load_for_prediction(cfg)

    dbp_index = dbp_name_to_index(args.dbp_name)
    protein_vec = protein_vector_for_dbp(cache, dbp_index)

    dna_seqs = read_sequences(args.dna_probe_file)
    scores = predict(model, torch.from_numpy(protein_vec), dna_seqs,
                     batch_size=cfg.predict.batch_size)

    with open(args.output_file, "w", encoding="utf-8") as fh:
        fh.write("\n".join(f"{s:.6f}" for s in scores))
        fh.write("\n")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
