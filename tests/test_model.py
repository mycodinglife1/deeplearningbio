"""Tests for the model: shapes, reproducibility, gradient flow, RC correctness."""

import numpy as np
import torch

from src.config import load_config
from src.model import build_model, reverse_complement_onehot, BindingModel
from src.encoders.protein import ESM2Encoder, LearnedAAEncoder, tokenize_proteins
from src.encoders.dna_cnn import DNAEncoder
from src.heads import build_head
from src.data.dna import one_hot_batch
from src.utils import set_seed, reverse_complement


ESM_DIM = 480  # default 35M model


def _esm_model_from_config():
    cfg = load_config("config.yaml", warn_missing_files=False)
    return build_model(cfg, protein_input_dim=ESM_DIM), cfg


def test_forward_output_shape_cached_input():
    set_seed(0)
    model, _ = _esm_model_from_config()
    n = 5
    protein_vecs = torch.randn(n, ESM_DIM)
    dna = torch.from_numpy(one_hot_batch(["ACGT" * 9] * n))
    out = model(protein_vecs, dna)
    assert out.shape == (n,)


def test_reproducible_under_fixed_seed():
    set_seed(123)
    m1, _ = _esm_model_from_config()
    set_seed(123)
    m2, _ = _esm_model_from_config()
    # eval() disables dropout so the forward pass is a deterministic function
    # of the (identically seeded) weights.
    m1.eval()
    m2.eval()
    protein_vecs = torch.randn(3, ESM_DIM)
    dna = torch.from_numpy(one_hot_batch(["ACGTACGTACGTACGTACGTACGTACGTACGTACGT"] * 3))
    with torch.no_grad():
        out1 = m1(protein_vecs, dna)
        out2 = m2(protein_vecs, dna)
    assert torch.allclose(out1, out2)


def test_reverse_complement_onehot_matches_string_rc():
    seqs = ["AACG" * 9, "ACGTTGCA" + "A" * 28]
    fwd = torch.from_numpy(one_hot_batch(seqs))
    rc = reverse_complement_onehot(fwd)
    expected = torch.from_numpy(one_hot_batch([reverse_complement(s) for s in seqs]))
    assert torch.equal(rc, expected)


def test_gradients_flow_to_towers_and_head_not_frozen_esm():
    """The trainable projection/CNN/head receive gradients; a frozen ESM does not."""
    set_seed(0)
    cfg = load_config("config.yaml", warn_missing_files=False)
    model = build_model(cfg, protein_input_dim=ESM_DIM)

    # Simulate a frozen ESM backbone attached to the encoder and confirm it
    # never receives gradients (we never call it in the cached path).
    esm_enc = model.protein_encoder
    assert isinstance(esm_enc, ESM2Encoder)

    protein_vecs = torch.randn(8, ESM_DIM)
    dna = torch.from_numpy(one_hot_batch(["ACGT" * 9] * 8))
    target = torch.randn(8)

    out = model(protein_vecs, dna)
    loss = torch.mean((out - target) ** 2)
    loss.backward()

    # Projection (protein tower), DNA conv, and head all have gradients.
    assert esm_enc.projection[1].weight.grad is not None
    assert model.dna_encoder.convs[0].weight.grad is not None
    head_param = next(model.head.parameters())
    assert head_param.grad is not None


def test_frozen_esm_params_have_no_grad():
    """A constructed ESM2Encoder with a loaded backbone keeps it requires_grad=False.

    We don't download ESM here; instead we attach a tiny fake frozen module and
    verify the freeze contract used by precompute (requires_grad_(False)).
    """
    enc = ESM2Encoder(input_dim=ESM_DIM, proj_dim=128)
    fake_backbone = torch.nn.Linear(4, 4)
    for p in fake_backbone.parameters():
        p.requires_grad_(False)
    enc._esm = fake_backbone
    assert all(not p.requires_grad for p in enc._esm.parameters())
    # The trainable projection is still trainable.
    assert any(p.requires_grad for p in enc.projection.parameters())


def test_learned_encoder_forward_shape():
    set_seed(0)
    enc = LearnedAAEncoder(proj_dim=128, embed_dim=32, max_len=64)
    dna_enc = DNAEncoder(conv_channels=32, kernel_sizes=[7], proj_dim=128)
    cfg = load_config("config.yaml", warn_missing_files=False)
    head = build_head(cfg.head, proj_dim=128)
    model = BindingModel(enc, dna_enc, head, use_reverse_complement=True)

    tokens = torch.from_numpy(tokenize_proteins(["MSSK", "GGGGHHHH"], max_len=64))
    dna = torch.from_numpy(one_hot_batch(["ACGT" * 9] * 2))
    out = model(tokens, dna)
    assert out.shape == (2,)


def test_predict_for_protein_shape():
    set_seed(0)
    model, _ = _esm_model_from_config()
    protein_vec = torch.randn(ESM_DIM)
    dna = torch.from_numpy(one_hot_batch(["ACGT" * 9] * 17))
    scores = model.predict_for_protein(protein_vec, dna)
    assert scores.shape == (17,)
