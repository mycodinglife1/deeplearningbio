"""Tests for the evaluation metrics, cross-checked against scipy."""

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from src.evaluate import (
    pearson_per_protein,
    spearman_per_protein,
    mse,
    r2_score,
    full_report,
)


def test_pearson_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        assert pearson_per_protein(x, y) == pytest.approx(pearsonr(x, y)[0], abs=1e-6)


def test_constant_vector_returns_zero_not_nan():
    x = np.ones(10)            # zero variance
    y = np.arange(10.0)
    val = pearson_per_protein(x, y)
    assert val == 0.0
    assert not np.isnan(val)


def test_identical_is_one_and_negated_is_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson_per_protein(x, x) == pytest.approx(1.0, abs=1e-9)
    assert pearson_per_protein(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_averaging_over_proteins():
    # Row 0 perfectly correlated (r=1), row 1 anti-correlated (r=-1) -> mean 0.
    pred = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    true = np.array([[2.0, 4.0, 6.0], [3.0, 2.0, 1.0]])
    assert pearson_per_protein(pred, true) == pytest.approx(0.0, abs=1e-9)


def test_dict_input_averages_shared_keys():
    pred = {"A": [1, 2, 3], "B": [3, 2, 1]}
    true = {"A": [1, 2, 3], "B": [1, 2, 3]}
    # A -> +1, B -> -1, mean 0.
    assert pearson_per_protein(pred, true) == pytest.approx(0.0, abs=1e-9)


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, size=40).astype(float)   # forces ties
    y = rng.normal(size=40)
    assert spearman_per_protein(x, y) == pytest.approx(spearmanr(x, y)[0], abs=1e-6)


def test_mse_and_r2_basic():
    x = np.array([1.0, 2.0, 3.0])
    assert mse(x, x) == 0.0
    assert r2_score(x, x) == pytest.approx(1.0)


def test_full_report_keys():
    rng = np.random.default_rng(2)
    pred = rng.normal(size=(3, 20))
    true = rng.normal(size=(3, 20))
    report = full_report(pred, true)
    assert set(report) == {"pearson_per_protein", "spearman_per_protein", "mse", "r2"}
