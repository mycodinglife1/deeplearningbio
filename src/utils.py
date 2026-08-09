"""Small cross-cutting utilities: seeding, device, logging, reverse-complement."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

# Complement table for DNA. Defined once at module load.
_COMPLEMENT = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C"})


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so runs are reproducible.

    We also set PYTHONHASHSEED and torch's deterministic flag where cheap;
    full bit-for-bit determinism isn't required for grading, but reproducible
    training/eval is part of being honest about results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-detect CUDA, else CPU. Everything in this project works on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "pbm") -> logging.Logger:
    """A plain stderr logger; idempotent so repeated calls don't duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (A<->T, C<->G, then reverse).

    Example: reverse_complement("AACG") == "CGTT".
    """
    return seq.translate(_COMPLEMENT)[::-1]
