"""Doubly-disjoint validation: hold out BOTH proteins and probes.

This mirrors the real test (and the course baseline): the model is trained on
train_proteins x train_probes, and scored on **val_proteins x val_probes** --
proteins it never saw, evaluated on probes it never saw. It also reports the
protein-only number (val_proteins x train_probes) so the probe-generalization
gap is explicit.

    python scripts/validate_double_disjoint.py [--epochs N] [--probe-holdout 0.1]

Nothing here changes the shipped model; it is a measurement run. Results are
written to artifacts/double_disjoint_log.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import io
from src.data.dna import one_hot_batch
from src.data.dataset import (protein_disjoint_split, FlatPairDataset, build_dataloader)
from src.encoders.protein import load_embedding_cache
from src.model import build_model
from src.train import deduplicate_proteins, build_loss, apply_protein_regularization
from src.evaluate import pearson_per_protein, spearman_per_protein
from src.utils import set_seed, get_device, get_logger

logger = get_logger("double-disjoint")


def _normalize_train_stats(uniq_raw, train_probe_ids, log1p):
    """Per-protein z-score using ONLY train-probe columns (no probe leakage)."""
    x = uniq_raw.astype(np.float64)
    if log1p:
        x = np.log1p(np.clip(x, 0.0, None))
    mean = x[:, train_probe_ids].mean(axis=1)
    std = x[:, train_probe_ids].std(axis=1)
    std = np.where(std > 1e-8, std, 1.0)
    return ((x - mean[:, None]) / std[:, None]).astype(np.float32)


def _subsample_train(train_ids, train_probe_ids, k, seed):
    """K probes per protein, drawn only from the TRAIN probe pool."""
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
    """Mean per-protein Pearson for val proteins over a given probe subset."""
    model.eval()
    sub = dna_onehot[probe_ids]
    d_chunks = []
    for s in range(0, sub.shape[0], bs):
        d_chunks.append(model.encode_dna(sub[s:s + bs].to(device)).cpu())
    d_all = torch.cat(d_chunks, dim=0)
    preds, trues = {}, {}
    for pid in val_ids:
        p = model.protein_encoder(protein_bank[pid:pid + 1].to(device)).cpu()
        scores = model.head(p.expand(d_all.shape[0], -1), d_all).numpy()
        preds[int(pid)] = scores
        trues[int(pid)] = uniq_raw[pid][probe_ids]
    return (pearson_per_protein(preds, trues), spearman_per_protein(preds, trues))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--probe-holdout", type=float, default=0.1,
                    help="fraction of probes held out for evaluation (default 0.1)")
    args = ap.parse_args()

    cfg = load_config(args.config, warn_missing_files=False)
    set_seed(cfg.seed)
    device = get_device()
    n_epochs = args.epochs if args.epochs is not None else cfg.train.epochs
    logger.info(f"Device: {device}")

    proteins = io.read_proteins(cfg.paths.train_dbps)
    probes = io.read_sequences(cfg.paths.train_seqs)
    intensities = np.load(os.path.join(cfg.paths.artifacts_dir, "intensities.npy"))
    uniq_seqs, uniq_raw, rep_idx = deduplicate_proteins(proteins, intensities)
    n_probes = len(probes)

    # --- protein-disjoint split (same as training) ---
    train_ids, val_ids = protein_disjoint_split(len(uniq_seqs),
                                                 cfg.train.val_protein_fraction, cfg.seed)
    # --- probe-disjoint split (NEW) ---
    rng = np.random.default_rng(cfg.seed + 777)
    perm = rng.permutation(n_probes)
    n_val_probes = int(round(n_probes * args.probe_holdout))
    val_probe_ids = np.sort(perm[:n_val_probes])
    train_probe_ids = np.sort(perm[n_val_probes:])
    logger.info(f"Proteins: {len(train_ids)} train / {len(val_ids)} val | "
                f"Probes: {len(train_probe_ids)} train / {len(val_probe_ids)} val")

    targets_norm = _normalize_train_stats(uniq_raw, train_probe_ids, cfg.target.log1p)

    cache = load_embedding_cache(cfg.paths.embeddings_cache)
    protein_bank = torch.from_numpy(cache.train_emb[np.asarray(rep_idx)].astype(np.float32)).to(device)
    dna_onehot = torch.from_numpy(one_hot_batch(probes))

    model = build_model(cfg, protein_input_dim=cache.esm_dim).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = build_loss(cfg.train.loss)

    best_dd, best_state, history = -2.0, None, []
    patience = cfg.train.early_stopping_patience
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        prot_idx, probe_idx = _subsample_train(
            train_ids, train_probe_ids, cfg.train.probes_per_protein_per_epoch, cfg.seed + epoch)
        ds = FlatPairDataset(prot_idx, probe_idx, dna_onehot.numpy(), targets_norm)
        # G: keep protein-contiguous order (shuffle off) so each batch is one
        # protein -> the correlation loss becomes a true per-protein Pearson.
        shuffle = not cfg.train.single_protein_batches
        loader = build_dataloader(ds, cfg.train.batch_size, shuffle=shuffle,
                                  num_workers=cfg.train.num_workers)
        running = n_b = 0
        for prot_b, dna_b, tgt_b in loader:
            protein_vecs = protein_bank[prot_b.to(device)]
            # C: perturb protein vectors during training (no-op if strengths are 0).
            protein_vecs = apply_protein_regularization(
                protein_vecs, cfg.train.protein_noise_std, cfg.train.protein_mask_prob, training=True)
            optimizer.zero_grad()
            loss = loss_fn(model(protein_vecs, dna_b.to(device)), tgt_b.to(device))
            loss.backward(); optimizer.step()
            running += float(loss.item()); n_b += 1

        # The honest number: unseen proteins x UNSEEN probes.
        dd_p, dd_s = _eval(model, protein_bank, dna_onehot, uniq_raw, val_ids,
                           val_probe_ids, device, cfg.predict.batch_size)
        # For the gap: unseen proteins x seen (train) probes.
        po_p, _ = _eval(model, protein_bank, dna_onehot, uniq_raw, val_ids,
                        train_probe_ids, device, cfg.predict.batch_size)
        history.append({"epoch": epoch, "train_loss": running / max(1, n_b),
                        "double_disjoint_pearson": dd_p, "double_disjoint_spearman": dd_s,
                        "protein_only_pearson": po_p})
        logger.info(f"epoch {epoch:3d} | loss {running/max(1,n_b):.4f} | "
                    f"DOUBLE-DISJOINT pearson {dd_p:.4f} | protein-only {po_p:.4f} "
                    f"| probe-gap {po_p-dd_p:+.4f}")

        if dd_p > best_dd:
            best_dd, best_epoch = dd_p, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = cfg.train.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    elapsed = time.time() - t0
    best_row = max(history, key=lambda h: h["double_disjoint_pearson"])
    logger.info(f"DONE in {elapsed:.1f}s")
    logger.info(f"BEST double-disjoint Pearson: {best_dd:.4f} (epoch {best_epoch})")
    logger.info(f"  at that epoch: protein-only {best_row['protein_only_pearson']:.4f}, "
                f"probe-gap {best_row['protein_only_pearson']-best_dd:+.4f}")

    out = {"best_double_disjoint_pearson": best_dd, "best_epoch": best_epoch,
           "history": history, "elapsed_s": elapsed,
           "probe_holdout": args.probe_holdout}
    json.dump(out, open(os.path.join(cfg.paths.artifacts_dir, "double_disjoint_log.json"), "w"), indent=2)
    print(f"\nBEST double-disjoint (unseen proteins x unseen probes) Pearson: {best_dd:.4f}")


if __name__ == "__main__":
    main()
