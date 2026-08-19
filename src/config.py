"""Typed configuration loaded and validated from ``config.yaml``.

Why a dataclass instead of passing dicts around: a typed object catches
typos and wrong types at load time (not deep inside the training loop), and
it documents exactly which knobs exist. The config is the single source of
truth for paths and hyperparameters — there are no magic numbers in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class PathsConfig:
    data_dir: str
    artifacts_dir: str
    train_dbps: str
    train_seqs: str
    train_intensities: str
    test_dbps: str
    test_seqs: str
    embeddings_cache: str
    model_ckpt: str


@dataclass
class ProteinEncoderConfig:
    type: str = "esm2"
    esm_model: str = "facebook/esm2_t12_35M_UR50D"
    pooling: str = "mean"
    max_len: int = 1022
    freeze: bool = True
    proj_dim: int = 128


@dataclass
class DNAEncoderConfig:
    type: str = "cnn"                  # cnn | cnn_transformer (F) | moe (E)
    use_reverse_complement: bool = True
    rc_combine: str = "mean"
    conv_channels: int = 128
    kernel_sizes: List[int] = field(default_factory=lambda: [15])
    proj_dim: int = 128
    dropout: float = 0.2
    # cnn_transformer (F): small self-attention over the 36 DNA positions.
    transformer_layers: int = 1
    transformer_heads: int = 4
    # moe (E): several CNN experts + a gating network.
    n_experts: int = 4


@dataclass
class HeadConfig:
    type: str = "concat_product"      # concat | concat_product | bilinear | film | cross_attention (B)
    hidden_dim: int = 128
    dropout: float = 0.2
    attn_heads: int = 4               # cross_attention: number of attention heads


@dataclass
class TargetConfig:
    log1p: bool = True
    per_protein_zscore: bool = True


@dataclass
class TrainConfig:
    val_protein_fraction: float = 0.1
    probes_per_protein_per_epoch: int = 3000
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "mse"
    early_stopping_patience: int = 6
    num_workers: int = 0
    # C — zero-shot protein regularization (0.0 = off, preserves old behavior).
    protein_noise_std: float = 0.0    # add N(0, std) to protein vectors while training
    protein_mask_prob: float = 0.0    # randomly zero this fraction of protein dims
    # G — correlation loss needs single-protein batches to compute per-protein r.
    single_protein_batches: bool = False


@dataclass
class PredictConfig:
    batch_size: int = 4096


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    protein_encoder: ProteinEncoderConfig
    dna_encoder: DNAEncoderConfig
    head: HeadConfig
    target: TargetConfig
    train: TrainConfig
    predict: PredictConfig

    # ---- validation helpers ---------------------------------------------

    def validate(self, warn_missing_files: bool = True) -> None:
        """Sanity-check enum-like fields and warn about absent data files.

        We *warn* rather than crash on missing data so the project can still
        be built/imported on a machine that doesn't have the (gitignored)
        data — the prompt explicitly asks for this.
        """
        _check_choice("protein_encoder.type", self.protein_encoder.type, ["esm2", "learned"])
        _check_choice("protein_encoder.pooling", self.protein_encoder.pooling, ["mean", "cls"])
        _check_choice("dna_encoder.rc_combine", self.dna_encoder.rc_combine, ["mean", "max"])
        _check_choice("dna_encoder.type", self.dna_encoder.type, ["cnn", "cnn_transformer", "moe"])
        _check_choice("head.type", self.head.type,
                      ["concat", "concat_product", "bilinear", "film", "cross_attention"])
        _check_choice("train.loss", self.train.loss, ["mse", "huber", "pearson"])

        if not self.dna_encoder.kernel_sizes:
            raise ValueError("dna_encoder.kernel_sizes must be a non-empty list")
        if not (0.0 < self.train.val_protein_fraction < 1.0):
            raise ValueError("train.val_protein_fraction must be in (0, 1)")

        if warn_missing_files:
            for label in ("train_dbps", "train_seqs", "train_intensities",
                          "test_dbps", "test_seqs"):
                path = getattr(self.paths, label)
                if not os.path.exists(path):
                    print(f"[config] WARNING: data file '{label}' not found at {path!r}. "
                          f"Building can continue; this is only needed for that step.")


def _check_choice(name: str, value, allowed: List[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is invalid; expected one of {allowed}")


def load_config(path: str = "config.yaml", warn_missing_files: bool = True) -> Config:
    """Load YAML into a validated ``Config``.

    Unknown keys are ignored defensively (so an older checkpoint's config can
    still load), and missing optional keys fall back to dataclass defaults.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(
        seed=int(raw["seed"]),
        paths=PathsConfig(**raw["paths"]),
        protein_encoder=ProteinEncoderConfig(**_subset(raw.get("protein_encoder", {}), ProteinEncoderConfig)),
        dna_encoder=DNAEncoderConfig(**_subset(raw.get("dna_encoder", {}), DNAEncoderConfig)),
        head=HeadConfig(**_subset(raw.get("head", {}), HeadConfig)),
        target=TargetConfig(**_subset(raw.get("target", {}), TargetConfig)),
        train=TrainConfig(**_subset(raw.get("train", {}), TrainConfig)),
        predict=PredictConfig(**_subset(raw.get("predict", {}), PredictConfig)),
    )
    cfg.validate(warn_missing_files=warn_missing_files)
    return cfg


def _subset(d: dict, dc) -> dict:
    """Keep only keys that are real fields of dataclass ``dc`` (ignore extras)."""
    valid = set(dc.__dataclass_fields__.keys())
    return {k: v for k, v in d.items() if k in valid}
