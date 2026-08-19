# Design — Predicting PBM Binding from Amino-Acid Sequence

*Deep Learning in Computational Biology — final project design document*

---

## 0. TL;DR

We build a two-tower recommender model. One tower turns a protein into a vector using a pretrained protein language model (ESM-2). The other tower turns a 36 bp DNA probe into a vector using a small 1D CNN. A small head combines the two vectors into one number: the predicted binding intensity.

- **Why this design:** the test proteins and test DNA are unseen during training, so we need to learn a general rule.
- **The bar:** the provided baseline averages **Pearson r ≈ 0.208** across proteins. It is a naive nearest-neighbor method. We expect a frozen-ESM two-tower model to land meaningfully higher.
- **KISS promise:** nothing here is exotic. Pretrained embeddings + a CNN + an MLP. If you understand "transformer gives me a vector" and "convolution slides a filter," you understand the whole model.

---

## 1. The problem, in plain words

**Input at prediction time** (this is the exact contract we must satisfy):

```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```

- `<DBP_name>` is `DBP1` … `DBP64`. The number is the line number in `test_DBPs.txt` (line 1 → `DBP1`). That line is the protein's amino-acid sequence.
- `<DNA_probe_file>` has one DNA sequence per line (length 36, alphabet `ACGT`).
- `<output_file>` must contain one number per line, same order as the DNA file — the predicted binding intensity of that protein for that probe.

**Goal:** for each of the 64 test proteins, predict a binding intensity for every test probe such that our predictions correlate with the true intensities.

**Metric:** Pearson correlation, computed per protein, then averaged over the 64 proteins.

```
pearson(x, y) = cov(x, y) / (std(x) · std(y))
```

> **The single most important consequence of using Pearson:** it is invariant to shifting and scaling. If the truth is `y`, then predicting `2y + 100` scores a perfect 1.0. So we do not need to predict the absolute intensity correctly — we only need to get the relative ranking of probes right, for each protein separately. This drives two design choices later: (a) we normalize each protein's targets to mean 0 / std 1 before training, and (b) a model that just outputs a per-protein constant scores 0 (becauase constants have zero variance), so the model is forced to actually use the DNA.


## 2. What the data actually looks like

| File | Count | Notes |
|---|---|---|
| `training_DBPs.txt` | 400 proteins (387 unique) | 1 per line; lengths 74–832, median ~106; alphabet = 20 standard AAs + `X` |
| `training_seqs.txt` | 30,000 DNA probes | length 36, `ACGT` only, sorted lexicographically |
| `training_data.zip` | the binding intensity of each of the 400 proteins for each of the 30,000 probes. |
| `test_DBPs.txt` | 64 proteins | lengths 43–455, median ~101 |
| `test_seqs.txt` | 11,728 DNA probes | length 36, `ACGT` only |
| `baseline_results.txt` | 20 Pearson scores | mean **0.208**, min 0.034, max 0.380 — the bar to beat |

---

## 3. The core idea: a two-tower recommender

Think of it like Netflix. Netflix predicts how much a user will like a movie from features of each. We predict how much a protein will bind a DNA probe from features of each. Binding intensity is the "rating."

```
                       ┌─────────────────────────────┐
  protein sequence ───►│  PROTEIN TOWER               │──► p  (vector, e.g. 128-d)
  (amino acids)        │  ESM-2 (frozen) → mean-pool  │
                       │  → small MLP projection      │
                       └─────────────────────────────┘
                                                          ┐
                                                          ├─► INTERACTION HEAD ──► binding
                                                          │   combine(p, d) → MLP    intensity
                       ┌─────────────────────────────┐    ┘        (one number)
  DNA probe       ────►│  DNA TOWER                   │──► d  (vector, e.g. 128-d)
  (36 bp, ACGT)        │  one-hot + reverse-comp      │
                       │  → 1D CNN → global max-pool  │
                       │  → small MLP projection      │
                       └─────────────────────────────┘
```

**Why two towers?**

- **Generalization:** the protein tower learns "what kind of binder is this," the DNA tower learns "what does this site look like," and the head learns "do they match." Each piece is reusable and the rule transfers to unseen proteins.
- **Speed:** protein embeddings can be **precomputed once** (there are only 464 proteins total). At prediction time we only run the cheap DNA tower + head. This is the key to the runtime budget.
- **Clarity:** each tower is independently testable and swappable.

---

## 4. Architecture in detail

### 4.1 Protein tower — pretrained embeddings (ESM-2)

**What it is.** ESM-2 is a transformer trained on ~tens of millions of natural protein sequences. Feed it a sequence of amino acids and it returns one vector per residue that encodes structural/functional context. We mean-pool those per-residue vectors into a single fixed-length protein vector, then pass it through a small trainable MLP to project into our shared space.

**Default checkpoint:** `facebook/esm2_t12_35M_UR50D` (35M params, **480-dim** output). Verified options for the config:

| Checkpoint | Layers | Params | Embedding dim | Use when |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 6 | 8M | 320 | fastest, weakest — quick experiments |
| `esm2_t12_35M_UR50D` | 12 | 35M | **480** | **default** — best speed/quality trade-off |
| `esm2_t30_150M_UR50D` | 30 | 150M | 640 | if accuracy needs a push and you have a GPU |

**Why pretrained instead of learning amino-acid embeddings from scratch?** With only ~400 training proteins, a from-scratch encoder has far too little data to learn what makes proteins similar, and it would generalize poorly to the unseen test proteins. ESM-2 has already learned that from millions of proteins for free.

**Frozen by default.** We do not fine-tune ESM-2 at first. We run it once, cache the 464 protein vectors to disk, and train only the small towers/head on top. This is faster, simpler, needs no GPU for training, and is already strong. *Fine-tuning ESM* is listed as an experiment in §10 (higher ceiling, much slower, easy to overfit 400 proteins).

### 4.2 DNA tower — a learnable motif scanner (1D CNN)

**What it is.** We one-hot encode the 36 bp probe into a `4 × 36` matrix (rows = A,C,G,T). A 1D convolution slides filters of width ~11–15 across it; each filter activates when it sees a particular short pattern. We take ReLU, then global max-pool over the length (β: "does this motif appear *anywhere* in the probe, and how strongly?"), giving one number per filter. A small MLP projects the filter activations to the shared DNA vector `d`.

**Reverse complement.** A protein can bind either DNA strand, and the array only reports one strand. So a motif on the reverse-complement strand should count too. We handle it cleanly:

- Compute the reverse complement of each probe (`A↔T`, `C↔G`, then reverse the string).
- Run both the forward and the RC probe through the same CNN (shared weights), and average (or max) the two DNA vectors.

This roughly doubles the effective signal for strand-symmetric motifs at almost no cost, and it's a config toggle (`use_reverse_complement: true`).


### 4.3 Interaction head — how the two vectors meet

Both towers output the same dimension `D` (e.g., 128). The head must let the protein and DNA vectors interact (not just sit side by side), otherwise the model can cheat by emitting a per-protein constant (which Pearson scores as 0).

**Default head (simple but forces interaction):**

```
combine(p, d) = concat([ p, d, p ⊙ d ])      # ⊙ = element-wise product
                → Linear(3D → D) → ReLU → Dropout → Linear(D → 1)
```

The `p ⊙ d` term is the "do these match" signal: it's large in dimensions where the protein "wants" something and the DNA "has" it. 


### 4.4 The model module = the orchestrator

A single `BindingModel(nn.Module)` **composes** `ProteinEncoder + DNAEncoder + InteractionHead`, all chosen from config. Swapping any component (different ESM size, learned-from-scratch protein encoder, bilinear head, RC on/off) is a **config change**, not a rewrite. That is the "main orchestrator for easy future changes" you asked for.

---

## 5. Data handling & preprocessing

### 5.1 Reading the data
A small `io` module reads proteins (one per line), DNA (one per line), and the intensities from `training_data.zip`. 

### 5.2 Target normalization (important, because of Pearson)
PBM intensities are right-skewed with heavy tails. Per protein, we:
1. `log1p` transform (tames the tail), then
2. z-score (subtract that protein's mean, divide by its std).

We train the model to predict the z-scored intensity. Because Pearson is scale/shift-invariant per protein, the z-scored target is the ideal learning target, by equalizing their influence on the loss.

### 5.3 Train/validation split — protein-disjoint (mirror the real test)
To pick hyperparameters honestly, the validation set must look like the real test: **proteins the model never trained on.** So we split by **protein**, not by random pairs — e.g., **360 train proteins / 40 validation proteins**, all 30,000 probes each. We report **mean per-protein Pearson on the held-out proteins** and use it for early stopping and model selection.

### 5.4 Handling ~11.6M pairs
387 proteins × 30,000 probes ≈ 11.6M training pairs — plenty. We don't need all of them every epoch. Each epoch, **subsample K probes per protein** (e.g., K=3,000 → ~1.16M pairs/epoch) so an epoch is fast. Protein vectors are precomputed and just looked up by index, so each training example is cheap.

---

## 6. Training

- **Loss:** `MSELoss` (or `SmoothL1`/Huber for robustness to outliers) on z-scored targets. Simple and aligns with Pearson once targets are z-scored.
- **Optimizer:** Adam/AdamW, lr ~1e-3, weight decay ~1e-4.
- **Batch size:** 256–1024 (these examples are tiny).
- **Early stopping / model selection:** on **mean validation Pearson** (the real metric), not on loss.
- **Regularization:** dropout in the head, weight decay, and the protein-disjoint split keep us honest. With only 400 proteins, overfitting the *protein* side is the main risk → another reason to freeze ESM by default.
- **Reproducibility:** fixed seeds; log config + metrics per run.
- **A correlation-style loss** (e.g., `1 − Pearson` computed within batches grouped by protein, or a listwise ranking loss) is an optional experiment — it optimizes the metric directly but needs protein-grouped batches. Keep MSE-on-zscore as the dependable default.

---

## 7. Inference & the runtime budget

**The rule:** prediction over all 64 DBPs must be fast. Loading ESM-2 inside `main.py` would likely blow the budget (large import + model load, possibly repeated per call). So ESM-2 must not run on the prediction path.

**The plan:**
1. **Offline (once, during training prep):** run ESM-2 over all 400 train + 64 test proteins and save vectors to `artifacts/protein_embeddings.npz`, keyed by `DBP` index. This is a fixed feature-extraction step (no labels involved) — standard and fair.
2. **At prediction time, `main.py`:**
   - loads the small trained model (`artifacts/model.pt`, a few MB) and the embedding cache (tiny);
   - looks up the requested protein's cached vector (no ESM, no `transformers` import on the hot path — import lazily only as a fallback if a vector is missing);
   - reads the DNA file, one-hot encodes (+ reverse complement), runs the DNA tower + head in batches, writes one score per line in input order.

**Back-of-envelope:** 11,728 probes through a tiny CNN is milliseconds on GPU and a few seconds on CPU; ×64 proteins is comfortably under 600 s even on CPU, with no ESM cost. We use `torch.inference_mode()` and a sane batch size. A `runtime_test.py` script replicates the real 64-call pattern and prints total seconds.

**Determinism:** inference is deterministic; we still set seeds and document CPU/GPU/memory in the report.

---

## 8. Evaluation & testing

### 8.1 Metrics module
- **`pearson_per_protein`** — the official metric. Average of per-protein Pearson over the evaluated proteins. Unit-tested against `scipy.stats.pearsonr` on random data.
- **Also report:** Spearman (rank correlation — robust sanity check), MSE/RMSE, and R². These help diagnose why a model is good or bad (e.g., high Spearman but low Pearson → a monotonic-but-nonlinear miss).
- **Edge cases handled & tested:** a constant prediction (zero variance → Pearson undefined) returns a defined sentinel (e.g., 0.0) rather than `NaN`/crash; identical inputs → 1.0; reversed inputs → −1.0.

### 8.2 Test suite
- `test_metrics.py` — Pearson matches scipy; constant/NaN edge cases; averaging logic.
- `test_dna.py` — one-hot shape `(4,36)` and correctness; reverse-complement correctness (`AACG → CGTT`); only `ACGT` accepted.
- `test_data.py` — protein/DNA file parsing; per-protein z-score has mean≈0/std≈1; protein-disjoint split shares no protein between train and val.
- `test_model.py` — forward pass returns the right shape; a fixed seed is reproducible; gradients flow to the towers but not into frozen ESM.
- `test_end_to_end.py` — a tiny synthetic dataset (a handful of fake proteins/probes with a planted rule) that the model can actually learn, proving the whole pipeline (train → predict → score) runs and that Pearson goes up. Also asserts `main.py` output has exactly one line per input probe, in order.
- A planted-signal sanity test: make a fake protein whose "binding" is high exactly when the probe contains `ACGTACGT`; the model should reach high Pearson on it. If it can't learn an obvious rule, something is wired wrong.

---

## 9. Project structure (modular, config-driven)

```
pbm-binding-predictor/
├── README.md                     # what it is, how to set up, how to run, how to reproduce
├── requirements.txt
├── config.yaml                   # ALL paths + hyperparameters + component choices
├── main.py                       # REQUIRED entrypoint: python main.py <ofile> <DBP> <DNA>
│
├── src/
│   ├── config.py                 # load + validate config into a dataclass
│   ├── data/
│   │   ├── io.py                 # read DBPs / seqs / intensities (+ unzip training_data.zip)
│   │   ├── dna.py                # one-hot + reverse complement
│   │   └── dataset.py            # Dataset/DataLoader, per-protein normalization, protein-disjoint split, per-epoch subsampling
│   ├── encoders/
│   │   ├── protein.py            # ProteinEncoder interface; ESM2Encoder (default) + LearnedAAEncoder (your idea); embedding cache
│   │   └── dna_cnn.py            # DNAEncoder (1D CNN motif scanner)
│   ├── heads.py                  # InteractionHead: concat+product (default), concat, bilinear, FiLM
│   ├── model.py                  # BindingModel — the ORCHESTRATOR that composes encoders + head from config
│   ├── train.py                  # training loop, early stopping on val Pearson, checkpointing
│   ├── predict.py                # encode one protein's probes → scores (used by main.py)
│   ├── evaluate.py               # Pearson-per-protein + Spearman + MSE/R²
│   └── utils.py                  # seeds, device, logging
│
├── scripts/
│   ├── inspect_data.py           # FIRST: print shapes/format of every file incl. training_data.zip
│   ├── precompute_embeddings.py  # run ESM-2 once → artifacts/protein_embeddings.npz
│   ├── train.py                  # CLI wrapper around src/train.py
│   ├── predict_all.py            # produce DBP1.txt … DBP64.txt and zip them (submission)
│   └── runtime_test.py           # time the real 64-call prediction pattern; print total seconds + efficiency score
│
├── tests/                        # the suite from §8
│
├── artifacts/                    # model.pt, protein_embeddings.npz, run logs   (shipped/needed at predict time)
└── data/                         # provided files incl. training_data.zip       (gitignored)
```

**Design principles:**
- **Config is the single source of truth.** Architecture choices live in `config.yaml`. Changing the model = editing config, occasionally adding one class behind an existing interface. No surgery across files.
- **Clean interfaces.** `ProteinEncoder.encode(seqs) -> [N, Dp]`, `DNAEncoder(onehot) -> [N, Dd]`, `InteractionHead(p, d) -> [N]`, all wired by `BindingModel`. New ideas implement an interface and register in a small factory.
- **One orchestrator at the model level** (`BindingModel`) and **one at the pipeline level** (`main.py` / `predict.py`). Training, evaluation, and prediction all reuse the same components.
- **Separation of slow/offline (ESM, training) from fast/online (prediction).**

---

## 10. Experiments and results (in order)

We started from the two-tower baseline above and improved it step by step. All
comparisons use the **doubly-disjoint** evaluation — unseen proteins **and**
unseen probes — which mirrors the real test set.

1. **Baseline two-tower** (ESM-2 35M frozen, CNN motif scanner, concat-product
   head, log1p + per-protein z-score, reverse-complement averaging). Mean
   per-protein Pearson **0.520**, already well above the 0.208 provided baseline.

2. **Larger protein model (ESM-2 650M) — no improvement.** A ~18× larger protein
   encoder (1280-d embeddings) did **not** help (0.586 vs 0.591 on the
   protein-disjoint setting) and is much heavier, so we kept the 35M encoder.
   Encoder size does not affect prediction runtime (ESM is offline and cached).

3. **Cross-attention head + zero-shot regularization — our main improvement.**
   The protein attends to the per-position DNA features (instead of a pooled
   product), and we regularize the protein vectors with Gaussian noise + feature
   masking during training to prevent memorizing the few training proteins. This
   raised mean per-protein Pearson **0.520 → 0.560 (+0.040)**, with prediction
   still well inside the runtime budget.

4. **Final training on all data.** For the submitted model we retrained the
   winning configuration on **all 387 unique proteins with no validation split**,
   using the epoch count chosen from the validated sweep — so every protein
   contributes to the final model.

The winning configuration (cross-attention head + zero-shot regularization,
35M ESM) is what we ship.