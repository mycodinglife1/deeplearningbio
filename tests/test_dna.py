"""Tests for DNA one-hot encoding and reverse complement."""

import numpy as np
import pytest

from src.data.dna import one_hot, one_hot_batch, BASES
from src.utils import reverse_complement


def test_one_hot_shape_and_values():
    seq = "ACGT" * 9  # length 36
    arr = one_hot(seq)
    assert arr.shape == (4, 36)
    # Each column is a single 1.0 (exactly one base set).
    assert np.all(arr.sum(axis=0) == 1.0)
    # First four columns spell A, C, G, T on the diagonal.
    assert np.array_equal(arr[:, :4], np.eye(4, dtype=np.float32))


def test_channel_order_is_ACGT():
    assert BASES == "ACGT"
    a = one_hot("A")
    assert a[0, 0] == 1.0 and a[1:, 0].sum() == 0.0
    t = one_hot("T")
    assert t[3, 0] == 1.0 and t[:3, 0].sum() == 0.0


def test_reverse_complement_known_case():
    assert reverse_complement("AACG") == "CGTT"
    assert reverse_complement("ACGT") == "ACGT"  # palindrome


def test_non_acgt_raises():
    with pytest.raises(ValueError):
        one_hot("ACGN")
    with pytest.raises(ValueError):
        one_hot("acgt")  # lowercase not accepted


def test_expected_len_validation():
    with pytest.raises(ValueError):
        one_hot("ACGT", expected_len=36)


def test_batch_shape():
    seqs = ["ACGT" * 9 for _ in range(5)]
    batch = one_hot_batch(seqs)
    assert batch.shape == (5, 4, 36)
    assert one_hot_batch([]).shape == (0, 4, 0)
