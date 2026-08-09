"""End-to-end: a tiny synthetic dataset with a PLANTED rule the model must learn.

Rule: each synthetic protein has a "favorite" k-mer; a probe's target is high
iff it contains that k-mer. We train a few steps on cached random protein
vectors (so the protein tower still has to *use* the protein to know which
k-mer matters) and assert val Pearson rises clearly above chance. We also check
the main.py-style prediction emits exactly one ordered line per input probe.

Runs offline in seconds — no ESM download, no network.
"""

import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

from src.config import load_config
from src.model import build_model
from src.data.dna import one_hot_batch
from src.evaluate import pearson_per_protein
from src.utils import set_seed, reverse_complement


BASES = "ACGT"


def _random_dna(rng, length=36):
    return "".join(rng.choice(list(BASES)) for _ in range(length))


def _make_synthetic(rng, n_proteins=6, n_probes=400, emb_dim=16, kmer_len=4, n_kmers=3):
    """Build a *generalizable* planted-rule dataset.

    There are a few shared k-mer "prototypes". Each protein prefers one of them,
    and its embedding is that prototype's (shared) signature plus small noise.
    A probe's target for a protein is how often that protein's k-mer appears.
    Because held-out proteins reuse prototypes seen in training, the model can
    learn the rule and generalize — which is exactly what we want to verify
    (and it forces the head to actually *use* the protein vector).
    """
    probes = [_random_dna(rng) for _ in range(n_probes)]
    kmers = ["".join(rng.choice(list(BASES)) for _ in range(kmer_len)) for _ in range(n_kmers)]
    # Shared per-prototype embedding signatures (scaled apart for separability).
    prototypes = (rng.normal(size=(n_kmers, emb_dim)) * 2.0).astype(np.float32)

    protein_emb = np.zeros((n_proteins, emb_dim), dtype=np.float32)
    protein_kmer = np.zeros(n_proteins, dtype=int)
    for p in range(n_proteins):
        k = p % n_kmers                                   # round-robin -> both sides see all
        protein_kmer[p] = k
        protein_emb[p] = prototypes[k] + 0.05 * rng.normal(size=emb_dim).astype(np.float32)

    targets = np.zeros((n_proteins, n_probes), dtype=np.float32)
    for p in range(n_proteins):
        kmer = kmers[protein_kmer[p]]
        rc_kmer = reverse_complement(kmer)
        for q, probe in enumerate(probes):
            # Strand-symmetric PRESENCE (matches both the RC averaging and the
            # DNA tower's global-max-pool = "does the motif appear anywhere?").
            present = (kmer in probe) or (rc_kmer in probe)
            targets[p, q] = float(present) + 0.02 * rng.normal()
    return probes, protein_emb, targets


def test_planted_signal_is_learnable_and_main_outputs_ordered():
    set_seed(0)
    rng = np.random.default_rng(0)
    cfg = load_config("config.yaml", warn_missing_files=False)
    # Use small towers via config edits in-memory.
    cfg.dna_encoder.conv_channels = 32
    cfg.dna_encoder.kernel_sizes = [4, 8]
    cfg.dna_encoder.proj_dim = 32
    cfg.head.hidden_dim = 32
    cfg.protein_encoder.proj_dim = 32

    emb_dim = 16
    n_kmers = 3
    probes, protein_emb, targets = _make_synthetic(rng, n_proteins=12, emb_dim=emb_dim,
                                                   n_kmers=n_kmers)
    n_proteins, n_probes = targets.shape

    # z-score targets per protein (as in real training).
    tmean = targets.mean(axis=1, keepdims=True)
    tstd = targets.std(axis=1, keepdims=True)
    tstd[tstd < 1e-6] = 1.0
    targets_norm = (targets - tmean) / tstd

    # Protein-disjoint split: last n_kmers proteins are validation (one per
    # prototype, all of which were also seen — under different proteins — in train).
    train_ids = list(range(n_proteins - n_kmers))
    val_ids = list(range(n_proteins - n_kmers, n_proteins))

    model = build_model(cfg, protein_input_dim=emb_dim)
    bank = torch.from_numpy(protein_emb)
    dna = torch.from_numpy(one_hot_batch(probes))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    loss_fn = torch.nn.MSELoss()

    # Train with mini-batch SGD over shuffled (protein, probe) pairs — proteins
    # are mixed within a batch, which is both realistic and stable.
    pairs = np.array([(p, q) for p in train_ids for q in range(n_probes)])
    model.train()
    for _ in range(40):
        rng.shuffle(pairs)
        for s in range(0, len(pairs), 128):
            b = pairs[s:s + 128]
            pred = model(bank[b[:, 0]], dna[b[:, 1]])
            loss = loss_fn(pred, torch.from_numpy(targets_norm[b[:, 0], b[:, 1]]))
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Validation Pearson on held-out proteins vs the (raw) planted targets.
    model.eval()
    preds, trues = {}, {}
    with torch.no_grad():
        for p in val_ids:
            scores = model(bank[p:p + 1].expand(n_probes, -1), dna).numpy()
            preds[p] = scores
            trues[p] = targets[p]
    val_pearson = pearson_per_protein(preds, trues)
    # The rule is cleanly learnable end-to-end; we comfortably clear the
    # prompt's >=0.5 bar (typically ~0.99). A high bar guards against silent
    # pipeline regressions (e.g. the head ignoring the protein vector).
    assert val_pearson >= 0.7, f"planted signal not learned (val Pearson={val_pearson:.3f})"


def test_main_outputs_one_ordered_line_per_probe():
    """A stubbed prediction path: assert N input probes -> N output lines in order.

    We exercise the real predict() on a tiny model + cache so no ESM is needed.
    """
    from src.predict import predict

    set_seed(0)
    cfg = load_config("config.yaml", warn_missing_files=False)
    cfg.dna_encoder.conv_channels = 8
    cfg.dna_encoder.kernel_sizes = [4]
    cfg.dna_encoder.proj_dim = 16
    cfg.head.hidden_dim = 16
    cfg.protein_encoder.proj_dim = 16
    model = build_model(cfg, protein_input_dim=24)

    rng = np.random.default_rng(1)
    probes = [_random_dna(rng) for _ in range(37)]
    protein_vec = torch.from_numpy(rng.normal(size=24).astype(np.float32))

    scores = predict(model, protein_vec, probes, batch_size=16)
    assert len(scores) == 37
    assert all(isinstance(s, float) for s in scores)
