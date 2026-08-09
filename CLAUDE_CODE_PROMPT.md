# Claude Code Build Prompt — PBM Binding Predictor

> Paste this whole file into Claude Code (in VS Code) as the initial instruction. It is self-contained. Work in the numbered **phases** and **stop to verify** at each checkpoint before continuing. Favor clarity over cleverness — this is a course project that will be read and graded by a human, and the author has just learned about transformers.

---

## 0. Mission

Build a **modular, config-driven PyTorch project** that predicts **protein–DNA binding intensity** for *unseen* proteins and *unseen* DNA probes (a Protein Binding Microarray / PBM task). The model is a **two-tower recommender**: a pretrained protein language model (ESM-2, frozen) encodes proteins; a small 1D CNN encodes DNA probes; a small head combines them into one score.

The evaluation metric is **Pearson correlation computed per protein, averaged over proteins**. The baseline to beat averages **0.208**. Aim well above that.

**Hard interface requirement (the grader calls this exactly):**
```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```
- `<DBP_name>` ∈ {`DBP1`,…,`DBP64`}; the number is the 1-based line in `test_DBPs.txt`.
- `<DNA_probe_file>`: one 36-bp `ACGT` sequence per line.
- `<output_file>`: one predicted intensity per line, **same order as the input**.

**Runtime requirement:** prediction over all 64 proteins must finish in **≤ 600 s** (efficiency score `max(min(1, 2 − t/600), 0)`). Therefore **ESM-2 must NOT run on the prediction path** — precompute and cache protein embeddings offline.

---

## 1. Verified data facts (don't re-derive, but DO verify the zip)

| File | Count | Notes |
|---|---|---|
| `training_DBPs.txt` | 400 proteins (387 unique) | 1/line; len 74–832; alphabet 20 AA + `X` |
| `training_seqs.txt` | 30,000 DNA | len 36, `ACGT`, lexicographically sorted |
| `training_data.zip` | binding intensities | **format UNCONFIRMED — inspect first** (likely 1 file per protein, 30,000 values aligned to `training_seqs.txt`) |
| `test_DBPs.txt` | 64 proteins | len 43–455 |
| `test_seqs.txt` | 11,728 DNA | len 36, `ACGT` |
| `baseline_results.txt` | 20 Pearson scores | mean 0.208 (the bar) |

No protein and no DNA overlap between train and test → genuine zero-shot. Many test proteins are homeodomains (`WFQNRR`) / zinc fingers → ESM-2 transfer is expected to work well.

Assume the data lives in `./data/` (gitignored). If a file is missing (especially `training_data.zip`), **print a clear message and continue building everything else** — don't block.

---

## 2. Tech stack & global constraints

- Python 3.10+, **PyTorch**, `transformers` (for ESM-2 via `facebook/esm2_t12_35M_UR50D`), `numpy`, `scipy` (only in tests/eval, for Pearson cross-check), `pyyaml`, `pytest`. Pin them in `requirements.txt`. Avoid heavy extra deps.
- **Config-driven:** every path and hyperparameter and component choice lives in `config.yaml`, loaded into a validated dataclass. No magic numbers in code.
- **Interfaces over hardcoding:** encoders and head are swappable behind small interfaces, selected by config via simple factory functions.
- **Determinism:** a `set_seed(seed)` util seeds python/numpy/torch.
- **Device:** auto-detect CUDA, fall back to CPU; everything must work on CPU.
- **Style:** type hints, short docstrings explaining *why*, no clever one-liners. Each module independently importable.
- **KISS guardrails (do NOT do these):** no graph neural nets, no custom CUDA, no giant config frameworks (hydra), no distributed training, no web UI, no premature abstraction. One conv layer is fine to start. If unsure, pick the simpler option and leave a `# TODO experiment:` note.

---

## 3. Directory layout to create

```
pbm-binding-predictor/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── dna.py
│   │   └── dataset.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── protein.py
│   │   └── dna_cnn.py
│   ├── heads.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── scripts/
│   ├── inspect_data.py
│   ├── precompute_embeddings.py
│   ├── train.py
│   ├── predict_all.py
│   └── runtime_test.py
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   ├── test_dna.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_end_to_end.py
├── artifacts/        # .gitkeep; model.pt + protein_embeddings.npz land here
└── data/             # gitignored; provided files go here
```

---

## 4. Module specifications (implement to these interfaces)

### `config.yaml` (and `src/config.py`)
A dataclass `Config` with nested sections, loaded+validated from YAML. Include at least:
```yaml
seed: 42
paths:
  data_dir: ./data
  artifacts_dir: ./artifacts
  train_dbps: ./data/training_DBPs.txt
  train_seqs: ./data/training_seqs.txt
  train_intensities: ./data/training_data.zip   # adjust after inspection
  test_dbps: ./data/test_DBPs.txt
  test_seqs: ./data/test_seqs.txt
  embeddings_cache: ./artifacts/protein_embeddings.npz
  model_ckpt: ./artifacts/model.pt
protein_encoder:
  type: esm2                 # esm2 | learned
  esm_model: facebook/esm2_t12_35M_UR50D
  pooling: mean              # mean | cls
  max_len: 1022
  freeze: true
  proj_dim: 128
dna_encoder:
  use_reverse_complement: true
  rc_combine: mean           # mean | max
  conv_channels: 128
  kernel_sizes: [15]         # list → parallel multi-width convs
  proj_dim: 128
head:
  type: concat_product       # concat | concat_product | bilinear | film
  hidden_dim: 128
  dropout: 0.2
target:
  log1p: true
  per_protein_zscore: true
train:
  val_protein_fraction: 0.1  # protein-disjoint split
  probes_per_protein_per_epoch: 3000
  batch_size: 512
  epochs: 30
  lr: 0.001
  weight_decay: 0.0001
  loss: mse                  # mse | huber | pearson
  early_stopping_patience: 5
predict:
  batch_size: 4096
```
`load_config(path) -> Config` validates types and that referenced files exist (warn, don't crash, if data files are absent).

### `src/utils.py`
`set_seed(seed)`, `get_device()`, a simple logger, and `reverse_complement(seq: str) -> str`.

### `src/data/io.py`
- `read_sequences(path) -> list[str]` (strip, drop blank lines).
- `read_proteins(path) -> list[str]`.
- `load_intensities(path, n_proteins, n_probes) -> np.ndarray [n_proteins, n_probes]`:
  - If `path` is a zip/dir, **inspect contents and adapt** (the most likely layout is one file per protein with one intensity per line, aligned to `training_seqs.txt`). Print what was found. Raise a clear error if it can't be aligned.
- `dbp_name_to_index(name: str) -> int` (`"DBP1" -> 0`).

### `src/data/dna.py`
- `one_hot(seq: str) -> np.ndarray [4, 36]`, channel order `A,C,G,T`. Validate length and alphabet.
- `one_hot_batch(seqs) -> np.ndarray [N, 4, L]`.
- Reverse complement helper (reuse `utils.reverse_complement`).

### `src/data/dataset.py`
- `PairDataset(torch.utils.data.Dataset)`: holds protein **indices**, one-hot DNA (or raw strings → one-hot in `__getitem__`), and z-scored targets. `__getitem__` returns `(protein_idx, dna_onehot, target)`.
- `normalize_targets(intensities, log1p, per_protein_zscore) -> (norm, stats)` — z-score **per protein (per row)**; return stats for optional inversion.
- `protein_disjoint_split(n_proteins, val_fraction, seed) -> (train_protein_ids, val_protein_ids)` — **no protein appears in both**.
- `make_epoch_subsample(...)` — sample K probes per protein per epoch.
- A `collate`/DataLoader builder.

### `src/encoders/protein.py`
- Abstract `ProteinEncoder` with `encode(sequences: list[str]) -> Tensor [N, proj_dim]` and a property `output_dim`.
- `ESM2Encoder(ProteinEncoder)`:
  - lazy-loads `AutoTokenizer`/`AutoModel` for `esm_model`; **frozen** (`eval()`, `requires_grad_(False)`) when `freeze`.
  - tokenizes (truncate to `max_len`), forward, **mean-pool over real residue tokens** (mask out pad/CLS/EOS), then a trainable `Linear(esm_dim → proj_dim)`.
  - **Caching:** `precompute(sequences) -> np.ndarray` returns pooled (pre-projection) ESM vectors so they can be saved; at train/predict time we load cached vectors and only learn the projection. Design so ESM itself is **never needed on the predict path** when the cache exists.
- `LearnedAAEncoder(ProteinEncoder)` (the simpler alternative for ablation): `nn.Embedding(26, d)` over amino acids + positional encoding + mean-pool (a 1-layer GRU/conv is fine) → `Linear → proj_dim`. Same interface.
- `build_protein_encoder(cfg) -> ProteinEncoder` factory.
- A module-level helper to **load/save the `.npz` embedding cache** keyed by a stable protein id (use the protein's own index within its file: train indices and test `DBP{i}`), plus a sequence→vector map fallback.

### `src/encoders/dna_cnn.py`
- `DNAEncoder(nn.Module)`: parallel `Conv1d(4, conv_channels, k, padding=k//2)` for each `k` in `kernel_sizes` → `ReLU` → `AdaptiveMaxPool1d(1)`, concatenate across kernels → `Linear(→ proj_dim)`. `forward(onehot [N,4,L]) -> [N, proj_dim]`.
- Reverse-complement handling lives here or in the model: run forward & RC through the **same** weights and combine by `rc_combine`.

### `src/heads.py`
- `InteractionHead(nn.Module)` variants selected by `head.type`:
  - `concat`: `[p, d] → MLP → 1`.
  - `concat_product` (**default**): `[p, d, p*d] → MLP → 1`.
  - `bilinear`: `pᵀ W d (+ small MLP)`.
  - `film`: `p` predicts scale/shift applied to `d`, then `→ 1`.
- `build_head(cfg, proj_dim) -> InteractionHead`. Output shape `[N]`.

### `src/model.py` — the orchestrator
- `BindingModel(nn.Module)` composes `protein_encoder`, `dna_encoder`, `head` (all from config).
- Two forward modes:
  - training: `forward(protein_emb [N,Dp_cached] OR seqs, dna_onehot [N,4,L]) -> [N]`. Accept **cached protein vectors** (then apply only the projection) to keep training fast; ESM stays frozen/optional.
  - predict: a convenience `predict_for_protein(protein_vec, dna_onehot_batch) -> scores`.
- Handles reverse-complement combination and the projection of cached ESM vectors.

### `src/train.py`
- `train(cfg)`:
  1. read proteins/seqs/intensities; de-duplicate proteins; normalize targets.
  2. load (or build) protein embedding cache.
  3. protein-disjoint split.
  4. loop epochs with per-epoch probe subsampling; optimizer per cfg; chosen loss.
  5. **after each epoch, compute mean per-protein Pearson on validation proteins**; checkpoint best; early-stop on patience.
  6. save `model.pt` (weights + the config used).
- Print a per-epoch line: epoch, train loss, **val mean Pearson**.

### `src/predict.py`
- `load_for_prediction(cfg) -> (model, embedding_cache)` — loads `model.pt` + `.npz`; **no `transformers` import unless a needed vector is missing** (then warn + lazy-load ESM as fallback).
- `predict(model, protein_vec, dna_seqs, batch_size) -> list[float]` — one-hot (+RC) → batched forward → scores in input order.

### `src/evaluate.py`
- `pearson_per_protein(pred: dict|array, true: dict|array) -> float` — mean over proteins; **constant vectors return 0.0**, never NaN.
- Also `spearman`, `mse`, `r2`. A `full_report(...) -> dict` returning all metrics.

### `main.py` (the graded entrypoint)
```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```
- parse 3 positional args (argparse);
- `load_config()` (default `./config.yaml`);
- `load_for_prediction(cfg)`; map `DBP_name`→index→cached protein vector;
- read DNA file; `predict(...)`; write one number per line in order;
- be fast and quiet (no heavy imports on the hot path; `torch.inference_mode()`).

---

## 5. Scripts

- `scripts/inspect_data.py` — **run this first.** Print line counts, length distributions, alphabets for every file, and **unzip + print the structure of `training_data.zip`** (file names, line counts, sample values). This is how we learn the intensity format.
- `scripts/precompute_embeddings.py` — run ESM-2 over **all** train+test proteins once; save `artifacts/protein_embeddings.npz` (keyed by train index and `DBP{i}`). Batched, CPU-friendly, shows progress.
- `scripts/train.py` — thin CLI over `src/train.py` (reads config, trains, saves).
- `scripts/predict_all.py` — for `DBP1..DBP64`, run prediction on `test_seqs.txt`, write `DBPk.txt`, then zip all into `submission.zip`.
- `scripts/runtime_test.py` — **replicate the real calling pattern**: invoke prediction for all 64 DBPs the way the grader would (e.g., loop calling `main.py` as a subprocess, or document clearly if measuring in-process), print **total seconds** and the implied efficiency score.

---

## 6. Tests (pytest) — must all pass

- `test_metrics.py`: `pearson_per_protein` matches `scipy.stats.pearsonr` (random data, tol 1e-6); constant input → 0.0 (not NaN); identical → 1.0; negated → −1.0; averaging correct.
- `test_dna.py`: one-hot shape `(4,36)` & values; `reverse_complement("AACG")=="CGTT"`; non-`ACGT` raises; batch shape.
- `test_data.py`: parsing drops blanks; per-protein z-score → row mean≈0, std≈1; `protein_disjoint_split` shares no protein id; `dbp_name_to_index("DBP1")==0`.
- `test_model.py`: `BindingModel` forward output shape `[N]`; reproducible under fixed seed; **no grad on frozen ESM params, grad present on towers/head**; works with a cached-embedding input.
- `test_end_to_end.py`: build a **tiny synthetic dataset with a planted rule** (e.g., target high iff probe contains a fixed k-mer); train a few steps; assert val Pearson rises above a threshold (≥ 0.5); assert `main.py`-style prediction outputs exactly N lines in order. Use a tiny `LearnedAAEncoder` or random cached vectors so the test needs **no ESM download** and runs in seconds.

Make tests runnable offline and fast (no network, no ESM weights).

---

## 7. Build order (phased — verify at each checkpoint)

**Phase 1 — scaffold & data understanding**
- Create the tree, `requirements.txt`, `README.md` skeleton, `config.yaml`, `src/config.py`, `src/utils.py`.
- Implement `scripts/inspect_data.py` and **run it**. Report the `training_data.zip` structure and **adjust `io.load_intensities` to match what you actually find.**
- ✅ *Checkpoint:* config loads; inspection prints sane shapes; intensity format confirmed (or a clear message if the zip is absent).

**Phase 2 — data pipeline**
- `data/io.py`, `data/dna.py`, `data/dataset.py`.
- Write & pass `test_dna.py`, `test_data.py`, `test_metrics.py` (+ `evaluate.py`).
- ✅ *Checkpoint:* these tests green.

**Phase 3 — model**
- `encoders/dna_cnn.py`, `encoders/protein.py` (both encoders + cache I/O), `heads.py`, `model.py`.
- Write & pass `test_model.py`.
- ✅ *Checkpoint:* forward shapes correct; frozen-ESM grads verified; reproducible.

**Phase 4 — training & embeddings**
- `scripts/precompute_embeddings.py`, `src/train.py`, `scripts/train.py`.
- Train on a **small subset** first (few proteins/probes, 1–2 epochs) to confirm loss decreases and val Pearson is computed.
- Write & pass `test_end_to_end.py` (synthetic, no ESM).
- ✅ *Checkpoint:* end-to-end green; tiny real run shows val Pearson rising.

**Phase 5 — prediction, runtime, submission**
- `src/predict.py`, `main.py`, `scripts/predict_all.py`, `scripts/runtime_test.py`.
- Verify `python main.py out.txt DBP1 ./data/test_seqs.txt` writes 11,728 ordered lines with **no ESM on the path**.
- Run `runtime_test.py`; confirm ≤ 600 s.
- ✅ *Checkpoint:* correct output; within time budget.

**Phase 6 — polish**
- Finish `README.md` (setup, how to precompute/train/predict/test, how to reproduce). Make sure each `# TODO experiment:` from §8 of the design is easy to toggle via config.
- ✅ *Checkpoint:* a newcomer can run everything from the README.

---

## 8. Definition of done

1. `pytest` is fully green, offline, in seconds.
2. `python main.py <out> DBP1 ./data/test_seqs.txt` produces 11,728 ordered numbers, no ESM import on the path.
3. `scripts/runtime_test.py` reports total 64-DBP prediction time and it is ≤ 600 s (CPU acceptable).
4. Training prints **val mean per-protein Pearson** per epoch and saves the best checkpoint; the number clearly exceeds **0.208** on the protein-disjoint validation set (with real intensities).
5. Swapping any component (ESM size, RC on/off, head type, learned-vs-ESM encoder, target transform) is a **config edit** only.
6. `README.md` lets a newcomer reproduce precompute → train → predict → evaluate.

## 9. Reporting hooks (so the final report writes itself)
While building, log to `artifacts/`: the config used, per-epoch val Pearson, final val Pearson + Spearman + MSE, prediction time, and peak memory. Add a `scripts/ablation.py` stub that runs the default config minus one feature at a time (RC, product head, log1p, etc.) and tabulates Δ val Pearson.

---

**Begin with Phase 1. After running `inspect_data.py`, summarize the `training_data.zip` format and your chosen `load_intensities` approach, then proceed.** Keep everything simple, documented, and readable.
