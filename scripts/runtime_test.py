"""Measure prediction time over all 64 DBPs and report the efficiency score.

The grading efficiency term is ``max(min(1, 2 - time[s]/600), 0)`` over the
prediction time for all 64 DBPs. We replicate the *real* calling pattern by
default — looping and invoking ``main.py`` as a subprocess for each DBP, the
way a grader would — so the measured time includes per-call startup.

``--in-process`` instead loads the model once and scores all 64 DBPs in a
single process (a lower bound that isolates pure compute from startup cost).

Usage:
    python scripts/runtime_test.py                 # subprocess pattern (default)
    python scripts/runtime_test.py --in-process     # single-load lower bound
    python scripts/runtime_test.py --n 5            # quick check on 5 DBPs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config


def efficiency_score(seconds: float) -> float:
    """Grading formula: max(min(1, 2 - t/600), 0)."""
    return max(min(1.0, 2.0 - seconds / 600.0), 0.0)


def run_subprocess(cfg, n_dbps: int, repo_root: str) -> float:
    """Invoke main.py once per DBP (the grader's real pattern); return total s."""
    main_py = os.path.join(repo_root, "main.py")
    dna_file = cfg.paths.test_seqs
    total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(1, n_dbps + 1):
            out_path = os.path.join(tmp, f"DBP{i}.txt")
            t0 = time.time()
            res = subprocess.run(
                [sys.executable, main_py, out_path, f"DBP{i}", dna_file],
                cwd=repo_root, capture_output=True, text=True,
            )
            dt = time.time() - t0
            total += dt
            if res.returncode != 0:
                print(f"DBP{i} FAILED:\n{res.stderr}")
                raise SystemExit(1)
        # Sanity: last output has the right number of lines.
        with open(out_path) as fh:
            n_lines = sum(1 for _ in fh)
    print(f"  (sanity: last DBP wrote {n_lines} lines)")
    return total


def run_in_process(cfg, n_dbps: int) -> float:
    """Load the model once, score all DBPs in-process; return total s."""
    import torch
    from src.data.io import read_sequences
    from src.predict import load_for_prediction, predict, protein_vector_for_dbp

    model, cache = load_for_prediction(cfg)
    probes = read_sequences(cfg.paths.test_seqs)
    n_dbps = min(n_dbps, cache.test_emb.shape[0])

    t0 = time.time()
    with torch.inference_mode():
        for i in range(n_dbps):
            vec = protein_vector_for_dbp(cache, i)
            predict(model, torch.from_numpy(vec), probes, batch_size=cfg.predict.batch_size)
    return time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="Time prediction over 64 DBPs.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=64, help="Number of DBPs to time.")
    ap.add_argument("--in-process", action="store_true",
                    help="Load once and score all DBPs in one process (lower bound).")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(repo_root, args.config), warn_missing_files=False)

    mode = "in-process (single load)" if args.in_process else "subprocess (per-DBP main.py)"
    print(f"Timing {args.n} DBPs — mode: {mode}")

    if args.in_process:
        total = run_in_process(cfg, args.n)
    else:
        total = run_subprocess(cfg, args.n, repo_root)

    # Extrapolate to 64 if a smaller n was used, for the budget comparison.
    scaled_64 = total * (64.0 / args.n)
    print(f"\nTotal time for {args.n} DBPs: {total:.2f} s")
    if args.n != 64:
        print(f"Extrapolated to 64 DBPs:  {scaled_64:.2f} s")
    print(f"Efficiency score (on 64-DBP time): {efficiency_score(scaled_64):.3f}  "
          f"[budget: <=600 s for full marks]")


if __name__ == "__main__":
    main()
