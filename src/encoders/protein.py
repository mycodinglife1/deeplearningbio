"""Protein tower: a frozen ESM-2 encoder (default) and a learned-from-scratch
alternative, behind one interface, plus the embedding-cache I/O.

Design split (see DESIGN.md §4.1, §7):
  * ESM-2 is run **once, offline** over all proteins; the pooled vectors are
    cached to ``protein_embeddings.npz``. ESM is NEVER imported on the
    prediction path — only the cached vectors + a tiny trainable projection.
  * The model receives, per training example, the **cached ESM vector**
    (``[N, esm_dim]``) and applies only the projection MLP. The training loop
    gathers these vectors from a bank by protein index, so the saved model is
    small and contains no ESM weights.
  * ``LearnedAAEncoder`` is a swappable ablation that learns amino-acid
    embeddings from scratch; it consumes token-id tensors instead of cached
    vectors. ESM is expected to win — running the comparison is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Known ESM-2 output (embedding) dimensions, so we can build the projection
# without loading the (large) model just to read its hidden size.
ESM_OUTPUT_DIMS: Dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
}


class ProteinEncoder(nn.Module):
    """Interface: ``forward(protein_input) -> [N, proj_dim]`` + ``output_dim``."""

    @property
    def output_dim(self) -> int:  # pragma: no cover - trivial
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ESM-2 encoder (default)
# ---------------------------------------------------------------------------

class ESM2Encoder(ProteinEncoder):
    """Projects cached, frozen ESM-2 vectors into the shared space.

    The trainable part is only the small projection MLP. ``precompute`` runs
    ESM offline; ``forward`` consumes already-pooled ``[N, esm_dim]`` vectors.
    """

    def __init__(self,
                 input_dim: int,
                 proj_dim: int = 128,
                 esm_model: str = "facebook/esm2_t12_35M_UR50D",
                 pooling: str = "mean",
                 max_len: int = 1022,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self._proj_dim = proj_dim
        self.esm_model = esm_model
        self.pooling = pooling
        self.max_len = max_len

        # Keep the protein side intentionally small (few hundred proteins) to
        # limit overfitting; LayerNorm stabilizes the raw ESM vector scale.
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Lazily-loaded ESM model/tokenizer (only for precompute / fallback).
        self._tokenizer = None
        self._esm = None

    @property
    def output_dim(self) -> int:
        return self._proj_dim

    def forward(self, cached_vecs: torch.Tensor) -> torch.Tensor:
        """``cached_vecs``: [N, esm_dim] -> [N, proj_dim]."""
        return self.projection(cached_vecs)

    # ---- offline ESM extraction ----------------------------------------

    def _lazy_load_esm(self):
        """Import transformers and load the model only when actually needed."""
        if self._esm is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # local import: off hot path
        self._tokenizer = AutoTokenizer.from_pretrained(self.esm_model)
        self._esm = AutoModel.from_pretrained(self.esm_model)
        self._esm.eval()
        for param in self._esm.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def precompute(self,
                   sequences: Sequence[str],
                   batch_size: int = 8,
                   device: Optional[torch.device] = None,
                   progress: bool = True) -> np.ndarray:
        """Run ESM-2 over ``sequences`` and return pooled ``[N, esm_dim]`` vectors.

        Pooling masks out pad/CLS/EOS so only real residues contribute (mean),
        or takes the CLS token (``pooling='cls'``).
        """
        self._lazy_load_esm()
        device = device or torch.device("cpu")
        self._esm.to(device)

        out = np.zeros((len(sequences), self.input_dim), dtype=np.float32)
        special_ids = {self._tokenizer.cls_token_id,
                       self._tokenizer.eos_token_id,
                       self._tokenizer.pad_token_id}
        special_ids.discard(None)

        for start in range(0, len(sequences), batch_size):
            chunk = list(sequences[start:start + batch_size])
            enc = self._tokenizer(chunk, return_tensors="pt", padding=True,
                                  truncation=True, max_length=self.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = self._esm(**enc).last_hidden_state           # [B, T, D]

            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                # Real-residue mask: attention_mask minus special tokens.
                mask = enc["attention_mask"].clone().float()      # [B, T]
                ids = enc["input_ids"]
                for sid in special_ids:
                    mask[ids == sid] = 0.0
                mask = mask.unsqueeze(-1)                          # [B, T, 1]
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            out[start:start + len(chunk)] = pooled.cpu().numpy()

            if progress:
                done = min(start + batch_size, len(sequences))
                print(f"\r  ESM precompute: {done}/{len(sequences)}", end="", flush=True)
        if progress:
            print()
        return out

    @torch.inference_mode()
    def encode(self, sequences: Sequence[str],
               device: Optional[torch.device] = None) -> torch.Tensor:
        """Fallback path: run ESM then project (slow; avoid on the hot path)."""
        vecs = torch.from_numpy(self.precompute(sequences, device=device, progress=False))
        return self.projection(vecs.to(next(self.projection.parameters()).device))


# ---------------------------------------------------------------------------
# Learned-from-scratch encoder (ablation)
# ---------------------------------------------------------------------------

# Amino-acid vocabulary: PAD=0, then the 20 standard AAs plus X (unknown).
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY" + "X"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}  # 0 reserved for PAD
PAD_ID = 0
VOCAB_SIZE = len(AA_ALPHABET) + 1


def tokenize_protein(seq: str, max_len: int) -> np.ndarray:
    """Map an amino-acid string to padded token ids ``[max_len]`` (PAD=0)."""
    ids = np.zeros(max_len, dtype=np.int64)
    for i, aa in enumerate(seq[:max_len]):
        ids[i] = AA_TO_ID.get(aa, AA_TO_ID["X"])
    return ids


def tokenize_proteins(seqs: Sequence[str], max_len: int) -> np.ndarray:
    """Tokenize a list of proteins to ``[N, max_len]`` token ids."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i] = tokenize_protein(s, max_len)
    return out


class LearnedAAEncoder(ProteinEncoder):
    """Embed amino acids from scratch + positional encoding + conv + mean-pool.

    Consumes token-id tensors ``[N, max_len]`` (PAD=0). This is the simpler
    alternative to ESM-2 for the ablation in the report.
    """

    def __init__(self,
                 proj_dim: int = 128,
                 embed_dim: int = 64,
                 max_len: int = 512,
                 kernel_size: int = 7,
                 dropout: float = 0.2):
        super().__init__()
        self._proj_dim = proj_dim
        self.max_len = max_len
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size,
                              padding=kernel_size // 2)
        self.project = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    @property
    def output_dim(self) -> int:
        return self._proj_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``token_ids``: [N, max_len] long -> [N, proj_dim] (masked mean-pool)."""
        mask = (token_ids != PAD_ID).float()                       # [N, T]
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.embed(token_ids) + self.pos(positions)[None, :, :]  # [N, T, E]
        x = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)  # [N, T, E]
        # Masked mean over real residues.
        summed = (x * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = summed / counts
        return self.project(pooled)


# ---------------------------------------------------------------------------
# Embedding cache I/O
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingCache:
    """Loaded ESM embedding cache: per-protein vectors + sequence fallback map."""
    train_emb: np.ndarray      # [n_train, esm_dim]
    test_emb: np.ndarray       # [n_test, esm_dim]  (row i -> DBP{i+1})
    train_seqs: List[str]
    test_seqs: List[str]
    esm_model: str
    esm_dim: int

    def __post_init__(self):
        # Sequence -> vector fallback (covers train and test).
        self._by_seq: Dict[str, np.ndarray] = {}
        for s, v in zip(self.train_seqs, self.train_emb):
            self._by_seq.setdefault(s, v)
        for s, v in zip(self.test_seqs, self.test_emb):
            self._by_seq.setdefault(s, v)

    def by_sequence(self, seq: str) -> Optional[np.ndarray]:
        """Return the cached vector for an exact sequence match, or None."""
        return self._by_seq.get(seq)


def save_embedding_cache(path: str,
                         train_emb: np.ndarray,
                         test_emb: np.ndarray,
                         train_seqs: Sequence[str],
                         test_seqs: Sequence[str],
                         esm_model: str) -> None:
    """Persist embeddings + the sequences (keys) + metadata to a ``.npz``."""
    np.savez_compressed(
        path,
        train_emb=train_emb.astype(np.float32),
        test_emb=test_emb.astype(np.float32),
        train_seqs=np.array(list(train_seqs), dtype=object),
        test_seqs=np.array(list(test_seqs), dtype=object),
        esm_model=np.array(esm_model),
        esm_dim=np.array(train_emb.shape[1]),
    )


def load_embedding_cache(path: str) -> EmbeddingCache:
    """Load a cache saved by :func:`save_embedding_cache`."""
    data = np.load(path, allow_pickle=True)
    return EmbeddingCache(
        train_emb=data["train_emb"],
        test_emb=data["test_emb"],
        train_seqs=list(data["train_seqs"]),
        test_seqs=list(data["test_seqs"]),
        esm_model=str(data["esm_model"]),
        esm_dim=int(data["esm_dim"]),
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_protein_encoder(cfg, input_dim: Optional[int] = None) -> ProteinEncoder:
    """Build the protein encoder from the ``protein_encoder`` config section.

    ``input_dim`` is the cached ESM dimension (required for ``esm2``). If not
    provided, it is looked up from the known ESM model dims.
    """
    if cfg.type == "esm2":
        if input_dim is None:
            input_dim = ESM_OUTPUT_DIMS.get(cfg.esm_model)
            if input_dim is None:
                raise ValueError(
                    f"Unknown ESM dim for {cfg.esm_model!r}; pass input_dim explicitly."
                )
        return ESM2Encoder(
            input_dim=input_dim,
            proj_dim=cfg.proj_dim,
            esm_model=cfg.esm_model,
            pooling=cfg.pooling,
            max_len=cfg.max_len,
        )
    if cfg.type == "learned":
        return LearnedAAEncoder(proj_dim=cfg.proj_dim, max_len=min(cfg.max_len, 512))
    raise ValueError(f"Unknown protein_encoder.type={cfg.type!r}")
