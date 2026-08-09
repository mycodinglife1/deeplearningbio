"""Evaluation metrics. The official metric is mean per-protein Pearson r.

Pearson is shift/scale-invariant, so a per-protein *constant* prediction has
zero variance and an undefined correlation. We treat that case as 0.0 (the
honest score for "no information about the ranking") rather than NaN, matching
how the grader's per-protein average must behave.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Single-vector Pearson r; returns 0.0 if either vector is constant."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in pearson: {x.shape} vs {y.shape}")
    if x.size < 2:
        return 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if denom == 0.0:  # constant vector -> undefined correlation -> 0.0
        return 0.0
    return float(np.sum(xc * yc) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on ranks (average ties)."""
    return _pearson(_rankdata(x), _rankdata(y))


def _rankdata(a: ArrayLike) -> np.ndarray:
    """Rank with ties averaged (mirrors scipy.stats.rankdata, no scipy needed)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1, dtype=np.float64)
    # Average ranks within tied groups.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def pearson_per_protein(pred: Union[Dict, ArrayLike],
                        true: Union[Dict, ArrayLike]) -> float:
    """Mean per-protein Pearson r — the official accuracy metric.

    ``pred``/``true`` may each be:
      * a dict {protein_key: vector} (averaged over shared keys), or
      * a 2-D array [n_proteins, n_probes] (averaged over rows), or
      * a 1-D vector (a single protein).
    """
    if isinstance(pred, dict) or isinstance(true, dict):
        keys = sorted(set(pred) & set(true))
        if not keys:
            return 0.0
        return float(np.mean([_pearson(np.asarray(pred[k]), np.asarray(true[k])) for k in keys]))

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim == 1:
        return _pearson(pred, true)
    return float(np.mean([_pearson(pred[i], true[i]) for i in range(pred.shape[0])]))


def spearman_per_protein(pred, true) -> float:
    """Mean per-protein Spearman r (rank correlation), same input shapes."""
    if isinstance(pred, dict) or isinstance(true, dict):
        keys = sorted(set(pred) & set(true))
        if not keys:
            return 0.0
        return float(np.mean([_spearman(np.asarray(pred[k]), np.asarray(true[k])) for k in keys]))
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.ndim == 1:
        return _spearman(pred, true)
    return float(np.mean([_spearman(pred[i], true[i]) for i in range(pred.shape[0])]))


def mse(pred: ArrayLike, true: ArrayLike) -> float:
    """Mean squared error over all elements."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    return float(np.mean((pred - true) ** 2))


def r2_score(pred: ArrayLike, true: ArrayLike) -> float:
    """Coefficient of determination over all elements (1.0 is perfect)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def full_report(pred, true) -> Dict[str, float]:
    """Return all metrics in one dict — handy for logging a run summary."""
    return {
        "pearson_per_protein": pearson_per_protein(pred, true),
        "spearman_per_protein": spearman_per_protein(pred, true),
        "mse": mse(_flatten(pred), _flatten(true)),
        "r2": r2_score(_flatten(pred), _flatten(true)),
    }


def _flatten(x):
    """Flatten dict-of-vectors or array into a single 1-D array for global metrics."""
    if isinstance(x, dict):
        return np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in x.values()])
    return np.asarray(x, dtype=np.float64).ravel()
