"""Produce DBP1.txt .. DBP64.txt for the test probes and zip them.

For each test DBP, score every probe in ``test_seqs.txt`` and write one number
per line (input order), then bundle all 64 files into ``submission.zip`` — the
exact submission artifact the course asks for.

Usage:
    python scripts/predict_all.py [--config config.yaml] [--out-dir submission]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.data.io import read_sequences
from src.predict import load_for_prediction, predict, protein_vector_for_dbp


def main() -> None:
    ap = argparse.ArgumentParser(description="Score all 64 test DBPs and zip the results.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--zip-name", default="submission.zip")
    args = ap.parse_args()

    cfg = load_config(args.config, warn_missing_files=False)
    os.makedirs(args.out_dir, exist_ok=True)

    model, cache = load_for_prediction(cfg)
    probes = read_sequences(cfg.paths.test_seqs)
    n_dbps = cache.test_emb.shape[0]
    print(f"Scoring {n_dbps} DBPs x {len(probes)} probes...")

    written = []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(n_dbps):
            vec = protein_vector_for_dbp(cache, i)
            scores = predict(model, torch.from_numpy(vec), probes,
                             batch_size=cfg.predict.batch_size)
            out_path = os.path.join(args.out_dir, f"DBP{i + 1}.txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(f"{s:.6f}" for s in scores) + "\n")
            written.append(out_path)
    print(f"Wrote {len(written)} files in {time.time() - t0:.1f}s.")

    zip_path = os.path.join(args.out_dir, args.zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, arcname=os.path.basename(path))
    print(f"Bundled submission -> {zip_path}")


if __name__ == "__main__":
    main()
