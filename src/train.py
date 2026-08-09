"""Training loop: protein-disjoint split, per-epoch probe subsampling, early
stopping on the *real* metric (mean per-protein Pearson against raw intensity).

Why evaluate against raw intensity (not the normalized target)? The grader
computes Pearson between predictions and the true raw intensities. z-scoring is
linear (Pearson-invariant) but ``log1p`` is not, so we select/early-stop on
Pearson vs raw to match grading exactly and report an honest number.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data import io
from .data.dna import one_hot_batch
from .data.dataset import (
    FlatPairDataset,
    build_dataloader,
    make_epoch_subsample,
    normalize_targets,
    protein_disjoint_split,
)
from .encoders.protein import load_embedding_cache, tokenize_proteins
from .evaluate import pearson_per_protein, spearman_per_protein
from .model import build_model
from .utils import get_device, get_logger, set_seed

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Data prep helpers
# ---------------------------------------------------------------------------

def load_or_cache_intensities(cfg: Config, n_proteins: int, n_probes: int) -> np.ndarray:
    """Load intensities, caching the parsed matrix to ``.npy`` for fast re-runs.

    Parsing the 108 MB text file takes a while; the ``.npy`` cache makes repeat
    training/ablation runs start instantly.
    """
    npy_cache = os.path.join(cfg.paths.artifacts_dir, "intensities.npy")
    if os.path.exists(npy_cache):
        arr = np.load(npy_cache)
        if arr.shape == (n_proteins, n_probes):
            logger.info(f"Loaded cached intensities {arr.shape} from {npy_cache}")
            return arr
        logger.info("Cached intensities shape mismatch; re-parsing.")
    logger.info("Parsing intensity matrix from source (one-time)...")
    arr = io.load_intensities(cfg.paths.train_intensities, n_proteins, n_probes)
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    np.save(npy_cache, arr)
    logger.info(f"Parsed and cached intensities {arr.shape}.")
    return arr


def deduplicate_proteins(seqs: List[str],
                         intensities: np.ndarray) -> Tuple[List[str], np.ndarray, List[int]]:
    """Collapse identical protein sequences, averaging their intensity rows.

    Averaging replicate measurements is standard for PBM and prevents a
    duplicated protein from being double-weighted (or straddling the split).
    Returns (unique_seqs, unique_intensities, representative_original_indices).
    """
    by_seq: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, s in enumerate(seqs):
        if s not in by_seq:
            by_seq[s] = []
            order.append(s)
        by_seq[s].append(idx)

    n_unique = len(order)
    unique_int = np.zeros((n_unique, intensities.shape[1]), dtype=np.float32)
    rep_idx: List[int] = []
    for u, s in enumerate(order):
        rows = by_seq[s]
        unique_int[u] = intensities[rows].mean(axis=0)
        rep_idx.append(rows[0])
    return order, unique_int, rep_idx


# ---------------------------------------------------------------------------
# Validation (efficient: encode DNA once, reuse across proteins)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_val(model: nn.Module,
                 protein_bank: torch.Tensor,
                 dna_onehot: torch.Tensor,
                 raw_intensities: np.ndarray,
                 val_ids: np.ndarray,
                 device: torch.device,
                 predict_batch_size: int) -> Dict[str, float]:
    """Mean per-protein Pearson on val proteins vs RAW intensities.

    DNA encodings (with RC) are computed once and shared across all proteins —
    the protein only enters via the head — which makes per-epoch validation
    cheap even over all 30k probes.
    """
    model.eval()
    n_probes = dna_onehot.shape[0]

    # Encode all probes once (batched to bound memory).
    d_chunks = []
    for start in range(0, n_probes, predict_batch_size):
        chunk = dna_onehot[start:start + predict_batch_size].to(device)
        d_chunks.append(model.encode_dna(chunk).cpu())
    d_all = torch.cat(d_chunks, dim=0)                       # [n_probes, proj_dim]

    preds: Dict[int, np.ndarray] = {}
    trues: Dict[int, np.ndarray] = {}
    for pid in val_ids:
        p = model.protein_encoder(protein_bank[pid:pid + 1].to(device)).cpu()  # [1, proj]
        p_exp = p.expand(n_probes, -1)
        scores = model.head(p_exp, d_all).numpy()
        preds[int(pid)] = scores
        trues[int(pid)] = raw_intensities[pid]

    return {
        "pearson": pearson_per_protein(preds, trues),
        "spearman": spearman_per_protein(preds, trues),
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def build_loss(name: str):
    """MSE / Huber on z-scored targets (both align with Pearson once z-scored)."""
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss()
    if name == "pearson":
        return _batch_pearson_loss
    raise ValueError(f"Unknown loss {name!r}")


def _batch_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson over the batch (a soft, metric-aligned alternative to MSE)."""
    pred_c = pred - pred.mean()
    tgt_c = target - target.mean()
    denom = torch.sqrt((pred_c ** 2).sum() * (tgt_c ** 2).sum()) + 1e-8
    return 1.0 - (pred_c * tgt_c).sum() / denom


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: Config,
          epochs: Optional[int] = None,
          max_train_proteins: Optional[int] = None,
          quiet: bool = False) -> Dict:
    """Train the model and save the best checkpoint to ``cfg.paths.model_ckpt``.

    Returns a result dict with the best val Pearson and the per-epoch history.
    ``epochs`` / ``max_train_proteins`` override config for quick smoke runs.
    """
    set_seed(cfg.seed)
    device = get_device()
    n_epochs = epochs if epochs is not None else cfg.train.epochs
    logger.info(f"Device: {device}")

    # --- read raw data ---------------------------------------------------
    proteins = io.read_proteins(cfg.paths.train_dbps)
    probes = io.read_sequences(cfg.paths.train_seqs)
    n_proteins, n_probes = len(proteins), len(probes)
    intensities = load_or_cache_intensities(cfg, n_proteins, n_probes)

    # --- dedup + normalize ----------------------------------------------
    uniq_seqs, uniq_raw, rep_idx = deduplicate_proteins(proteins, intensities)
    logger.info(f"Proteins: {n_proteins} -> {len(uniq_seqs)} unique; probes: {n_probes}")
    targets_norm, _ = normalize_targets(uniq_raw, cfg.target.log1p, cfg.target.per_protein_zscore)

    # --- protein features (cached ESM vectors, or learned-encoder tokens) -
    if cfg.protein_encoder.type == "esm2":
        if not os.path.exists(cfg.paths.embeddings_cache):
            raise FileNotFoundError(
                f"Embedding cache {cfg.paths.embeddings_cache!r} not found. "
                f"Run: python scripts/precompute_embeddings.py"
            )
        cache = load_embedding_cache(cfg.paths.embeddings_cache)
        protein_input_dim = cache.esm_dim
        # Gather the embedding for each unique protein (by representative index).
        bank_np = cache.train_emb[np.asarray(rep_idx)]
        protein_bank = torch.from_numpy(bank_np.astype(np.float32))
    else:  # learned encoder: bank is padded token ids
        max_len = min(cfg.protein_encoder.max_len, 512)
        protein_bank = torch.from_numpy(tokenize_proteins(uniq_seqs, max_len))
        protein_input_dim = None

    # --- one-hot all DNA once -------------------------------------------
    dna_onehot = torch.from_numpy(one_hot_batch(probes))      # [n_probes, 4, 36]

    # --- protein-disjoint split -----------------------------------------
    train_ids, val_ids = protein_disjoint_split(len(uniq_seqs),
                                                cfg.train.val_protein_fraction, cfg.seed)
    if max_train_proteins is not None:
        train_ids = train_ids[:max_train_proteins]
        val_ids = val_ids[:max(2, max_train_proteins // 4)]
    logger.info(f"Split: {len(train_ids)} train proteins, {len(val_ids)} val proteins")

    # --- model / optimizer ----------------------------------------------
    model = build_model(cfg, protein_input_dim=protein_input_dim).to(device)
    protein_bank = protein_bank.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
    )
    loss_fn = build_loss(cfg.train.loss)

    # --- training loop ---------------------------------------------------
    best_pearson = -2.0
    best_state = None
    patience_left = cfg.train.early_stopping_patience
    history: List[Dict] = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        prot_idx, probe_idx = make_epoch_subsample(
            train_ids, n_probes, cfg.train.probes_per_protein_per_epoch, cfg.seed + epoch)
        ds = FlatPairDataset(prot_idx, probe_idx, dna_onehot.cpu().numpy()
                             if device.type != "cpu" else dna_onehot.numpy(),
                             targets_norm)
        loader = build_dataloader(ds, cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.train.num_workers)

        running, n_batches = 0.0, 0
        for prot_b, dna_b, target_b in loader:
            protein_vecs = protein_bank[prot_b.to(device)]
            dna_b = dna_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad()
            pred = model(protein_vecs, dna_b)
            loss = loss_fn(pred, target_b)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val = evaluate_val(model, protein_bank, dna_onehot, uniq_raw, val_ids,
                           device, cfg.predict.batch_size)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_pearson": val["pearson"], "val_spearman": val["spearman"]})
        if not quiet:
            logger.info(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                        f"val_pearson {val['pearson']:.4f} | val_spearman {val['spearman']:.4f}")

        # Model selection + early stopping on val Pearson (the real metric).
        if val["pearson"] > best_pearson:
            best_pearson = val["pearson"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.train.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(f"Early stopping at epoch {epoch} (best val Pearson {best_pearson:.4f}).")
                break

    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed:.1f}s. Best val Pearson: {best_pearson:.4f}")

    # --- save best checkpoint -------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    _save_checkpoint(cfg, model, protein_input_dim, best_pearson)
    _save_run_log(cfg, history, best_pearson, elapsed)

    return {"best_val_pearson": best_pearson, "history": history, "elapsed_s": elapsed}


def _save_checkpoint(cfg: Config, model: nn.Module,
                     protein_input_dim: Optional[int], best_pearson: float) -> None:
    """Save model weights + the config needed to rebuild it at predict time."""
    os.makedirs(os.path.dirname(cfg.paths.model_ckpt), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": _config_to_dict(cfg),
        "protein_input_dim": protein_input_dim,
        "esm_model": cfg.protein_encoder.esm_model,
        "best_val_pearson": best_pearson,
    }, cfg.paths.model_ckpt)
    logger.info(f"Saved checkpoint to {cfg.paths.model_ckpt}")


def _save_run_log(cfg: Config, history: List[Dict],
                  best_pearson: float, elapsed: float) -> None:
    """Persist the config + per-epoch metrics so the report writes itself."""
    log_path = os.path.join(cfg.paths.artifacts_dir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump({
            "config": _config_to_dict(cfg),
            "history": history,
            "best_val_pearson": best_pearson,
            "elapsed_s": elapsed,
        }, fh, indent=2)


def _config_to_dict(cfg: Config) -> Dict:
    """Serialize the Config dataclass to a plain dict."""
    return {
        "seed": cfg.seed,
        "protein_encoder": asdict(cfg.protein_encoder),
        "dna_encoder": asdict(cfg.dna_encoder),
        "head": asdict(cfg.head),
        "target": asdict(cfg.target),
        "train": asdict(cfg.train),
        "predict": asdict(cfg.predict),
    }
