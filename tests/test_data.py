"""Tests for IO parsing, target normalization, and the protein-disjoint split."""

import os
import tempfile
import zipfile

import numpy as np
import pytest

from src.data.io import (
    read_sequences,
    read_proteins,
    dbp_name_to_index,
    load_intensities,
)
from src.data.dataset import (
    normalize_targets,
    protein_disjoint_split,
    make_epoch_subsample,
)


def test_read_sequences_drops_blank_lines():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seqs.txt")
        with open(path, "w") as fh:
            fh.write("ACGT\n\n  \nTTTT\n")
        seqs = read_sequences(path)
    assert seqs == ["ACGT", "TTTT"]


def test_read_proteins_strips():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.txt")
        with open(path, "w") as fh:
            fh.write("  MSSK \nGGGG\n")
        assert read_proteins(path) == ["MSSK", "GGGG"]


def test_dbp_name_to_index():
    assert dbp_name_to_index("DBP1") == 0
    assert dbp_name_to_index("DBP64") == 63
    assert dbp_name_to_index(" DBP10 ") == 9
    with pytest.raises(ValueError):
        dbp_name_to_index("DBP0")
    with pytest.raises(ValueError):
        dbp_name_to_index("XYZ3")


def test_load_intensities_transposes():
    # On disk: [n_probes=3, n_proteins=2]; expect transpose to [2, 3].
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "training_data.txt")
        with open(txt, "w") as fh:
            fh.write("1 2\n3 4\n5 6\n")
        zip_path = os.path.join(d, "training_data.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(txt, arcname="training_data.txt")
        mat = load_intensities(zip_path, n_proteins=2, n_probes=3)
    assert mat.shape == (2, 3)
    assert np.array_equal(mat, np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32))


def test_load_intensities_bad_shape_raises():
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "training_data.txt")
        with open(txt, "w") as fh:
            fh.write("1 2 3\n4 5 6\n")
        with pytest.raises(ValueError):
            load_intensities(txt, n_proteins=5, n_probes=9)


def test_per_protein_zscore_mean0_std1():
    rng = np.random.default_rng(0)
    intensities = rng.exponential(scale=2.0, size=(7, 500)).astype(np.float32)
    norm, stats = normalize_targets(intensities, log1p=True, per_protein_zscore=True)
    assert np.allclose(norm.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(norm.std(axis=1), 1.0, atol=1e-5)
    assert stats.mean.shape == (7,)


def test_normalize_constant_row_is_safe():
    intensities = np.ones((2, 10), dtype=np.float32)
    norm, _ = normalize_targets(intensities, log1p=True, per_protein_zscore=True)
    assert not np.any(np.isnan(norm))


def test_protein_disjoint_split_no_overlap():
    train_ids, val_ids = protein_disjoint_split(100, val_fraction=0.1, seed=42)
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == 100
    assert len(val_ids) == 10


def test_epoch_subsample_covers_all_proteins():
    prot_idx, probe_idx = make_epoch_subsample([0, 1, 2], n_probes=100,
                                               probes_per_protein=10, seed=0)
    assert prot_idx.shape == probe_idx.shape == (30,)
    assert set(prot_idx.tolist()) == {0, 1, 2}
    # No probe repeats within a single protein's sample.
    for pid in (0, 1, 2):
        probes = probe_idx[prot_idx == pid]
        assert len(set(probes.tolist())) == 10
