"""Reusable doubly-disjoint train+evaluate, for the overnight experiment sweep.

One place that: prepares the shared data once (protein-disjoint AND probe-disjoint
splits, one-hot DNA, train-probe-only target normalization), then trains a model
for a given config and reports the honest metric — mean per-protein Pearson on
**unseen proteins x unseen probes** (mirroring the real test).

Kept separate from src/train.py (which trains the shipped model on all probes);
this module is only for measuring which ideas help.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .data import io
from .data.dna import one_hot_batch
from .data.dataset import protein_disjoint_split, FlatPairDataset, build_dataloader
from .encoders.protein import load_embedding_cache
from .model import build_model
from .train import deduplicate_proteins, build_loss, apply_protein_regularization
from .evaluate import pearson_per_protein, spearman_per_protein
from .utils import set_seed, get_device


@dataclass
class SharedData:
    uniq_seqs: List[str]
    uniq_raw: np.ndarray
    rep_idx: List[int]
    train_ids: np.ndarray
    val_ids: np.ndarray
    train_probe_ids: np.ndarray
    val_probe_ids: np.ndarray
    dna_onehot: torch.Tensor
    targets_norm: np.ndarray


def _normalize_train_stats(uniq_raw, train_probe_ids, log1p) -> np.ndarray:
    x = uniq_raw.astype(np.float64)
    if log1p:
        x = np.log1p(np.clip(x, 0.0, None))
    mean = x[:, train_probe_ids].mean(axis=1)
    std = x[:, train_probe_ids].std(axis=1)
    std = np.where(std > 1e-8, std, 1.0)
    return ((x - mean[:, None]) / std[:, None]).astype(np.float32)


def prepare_shared_data(cfg: Config, probe_holdout: float = 0.1) -> SharedData:
    """Load data once and build the protein- and probe-disjoint splits."""
    proteins = io.read_proteins(cfg.paths.train_dbps)
    probes = io.read_sequences(cfg.paths.train_seqs)
    intensities = np.load(f"{cfg.paths.artifacts_dir}/intensities.npy")
    uniq_seqs, uniq_raw, rep_idx = deduplicate_proteins(proteins, intensities)
    n_probes = len(probes)

    train_ids, val_ids = protein_disjoint_split(
        len(uniq_seqs), cfg.train.val_protein_fraction, cfg.seed)

    rng = np.random.default_rng(cfg.seed + 777)
    perm = rng.permutation(n_probes)
    n_val_probes = int(round(n_probes * probe_holdout))
    val_probe_ids = np.sort(perm[:n_val_probes])
    train_probe_ids = np.sort(perm[n_val_probes:])

    targets_norm = _normalize_train_stats(uniq_raw, train_probe_ids, cfg.target.log1p)
    dna_onehot = torch.from_numpy(one_hot_batch(probes))
    return SharedData(uniq_seqs, uniq_raw, rep_idx, train_ids, val_ids,
                      train_probe_ids, val_probe_ids, dna_onehot, targets_norm)


def bank_from_cache(cache_path: str, rep_idx) -> Tuple[torch.Tensor, int]:
    """Gather per-unique-protein embedding vectors from an .npz cache."""
    cache = load_embedding_cache(cache_path)
    bank = torch.from_numpy(cache.train_emb[np.asarray(rep_idx)].astype(np.float32))
    return bank, cache.esm_dim


def _subsample_train(train_ids, train_probe_ids, k, seed):
    rng = np.random.default_rng(seed)
    k = min(k, len(train_probe_ids))
    prot, probe = [], []
    for pid in train_ids:
        chosen = rng.choice(train_probe_ids, size=k, replace=False)
        prot.extend([int(pid)] * k)
        probe.extend(int(c) for c in chosen)
    return np.asarray(prot, dtype=np.int64), np.asarray(probe, dtype=np.int64)


@torch.inference_mode()
def _eval(model, protein_bank, dna_onehot, uniq_raw, val_ids, probe_ids, device, bs):
    model.eval()
    sub = dna_onehot[probe_ids]
    preds, trues = {}, {}
    # Encode DNA once when the head uses the pooled vector; the cross-attention
    # head needs per-protein attention, so we fall back to per-protein scoring.
    needs_positions = getattr(model.head, "needs_positions", False)
    if not needs_positions:
        d_chunks = []
        for s in range(0, sub.shape[0], bs):
            d_chunks.append(model.encode_dna(sub[s:s + bs].to(device)).cpu())
        d_all = torch.cat(d_chunks, dim=0)
        for pid in val_ids:
            p = model.protein_encoder(protein_bank[pid:pid + 1].to(device)).cpu()
            preds[int(pid)] = model.head(p.expand(d_all.shape[0], -1), d_all).numpy()
            trues[int(pid)] = uniq_raw[pid][probe_ids]
    else:
        for pid in val_ids:
            vec = protein_bank[pid].to(device)
            scores = []
            for s in range(0, sub.shape[0], bs):
                scores.append(model.predict_for_protein(vec, sub[s:s + bs].to(device)).cpu())
            preds[int(pid)] = torch.cat(scores).numpy()
            trues[int(pid)] = uniq_raw[pid][probe_ids]
    return pearson_per_protein(preds, trues), spearman_per_protein(preds, trues)


def run_experiment(cfg: Config,
                   shared: SharedData,
                   protein_bank: torch.Tensor,
                   input_dim: int,
                   epochs: int,
                   device: Optional[torch.device] = None,
                   seed_offset: int = 0,
                   log=print) -> Dict:
    """Train one config and return its doubly-disjoint result dict."""
    device = device or get_device()
    set_seed(cfg.seed + seed_offset)
    protein_bank = protein_bank.to(device)
    dna_onehot = shared.dna_onehot

    model = build_model(cfg, protein_input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = build_loss(cfg.train.loss)

    best_dd, best_epoch, best_state = -2.0, 0, None
    best_po = 0.0
    history = []
    patience = cfg.train.early_stopping_patience
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        prot_idx, probe_idx = _subsample_train(
            shared.train_ids, shared.train_probe_ids,
            cfg.train.probes_per_protein_per_epoch, cfg.seed + seed_offset + epoch)
        ds = FlatPairDataset(prot_idx, probe_idx, dna_onehot.numpy(), shared.targets_norm)
        shuffle = not cfg.train.single_protein_batches
        loader = build_dataloader(ds, cfg.train.batch_size, shuffle=shuffle,
                                  num_workers=cfg.train.num_workers)
        running = n_b = 0
        for prot_b, dna_b, tgt_b in loader:
            pv = protein_bank[prot_b.to(device)]
            pv = apply_protein_regularization(pv, cfg.train.protein_noise_std,
                                              cfg.train.protein_mask_prob, training=True)
            optimizer.zero_grad()
            loss = loss_fn(model(pv, dna_b.to(device)), tgt_b.to(device))
            loss.backward(); optimizer.step()
            running += float(loss.item()); n_b += 1

        dd_p, dd_s = _eval(model, protein_bank, dna_onehot, shared.uniq_raw,
                           shared.val_ids, shared.val_probe_ids, device, cfg.predict.batch_size)
        history.append({"epoch": epoch, "loss": running / max(1, n_b),
                        "double_disjoint_pearson": dd_p, "double_disjoint_spearman": dd_s})
        log(f"    epoch {epoch:2d} | loss {running/max(1,n_b):.4f} | DD pearson {dd_p:.4f}")
        if dd_p > best_dd:
            best_dd, best_epoch, best_po = dd_p, epoch, dd_s
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.train.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                log(f"    early stop at epoch {epoch}")
                break

    return {"best_double_disjoint_pearson": best_dd,
            "best_double_disjoint_spearman": best_po,
            "best_epoch": best_epoch, "history": history,
            "elapsed_s": time.time() - t0, "state_dict": best_state}
