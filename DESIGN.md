# Design — Predicting PBM Binding from Amino-Acid Sequence

*Deep Learning in Computational Biology — final project design document*

---

## 0. TL;DR

We build a **two-tower recommender model**. One tower turns a **protein** into a vector using a pretrained protein language model (ESM-2). The other tower turns a **36 bp DNA probe** into a vector using a small 1D CNN (a learnable motif scanner). A small head combines the two vectors into one number: the predicted binding intensity.

- **Why this design:** the test proteins and test DNA are *completely unseen* during training, so we cannot memorize. We must learn a *general* "protein ↔ DNA" rule. A recommender/two-tower model is the standard, well-understood way to do that.
- **The bar:** the provided baseline averages **Pearson r ≈ 0.208** across proteins. It is a naive nearest-neighbor method. We expect a frozen-ESM two-tower model to land meaningfully higher (commonly **~0.35–0.55** average Pearson on cross-protein PBM tasks; the validation harness will tell us the truth on *this* data).
- **KISS promise:** nothing here is exotic. Pretrained embeddings + a CNN + an MLP. If you understand "transformer gives me a vector" and "convolution slides a filter," you understand the whole model.

---

## 1. The problem, in plain words

**Input at prediction time** (this is the exact contract we must satisfy):

```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```

- `<DBP_name>` is `DBP1` … `DBP64`. The number is the **line number** in `test_DBPs.txt` (line 1 → `DBP1`). That line is the protein's amino-acid sequence.
- `<DNA_probe_file>` has one DNA sequence per line (length 36, alphabet `ACGT`).
- `<output_file>` must contain **one number per line, same order as the DNA file** — the predicted binding intensity of that protein for that probe.

**Goal:** for each of the 64 test proteins, predict a binding intensity for every test probe such that our predictions *correlate* with the true intensities.

**Metric:** **Pearson correlation, computed per protein**, then averaged over the 64 proteins.

```
pearson(x, y) = cov(x, y) / (std(x) · std(y))
```

> **The single most important consequence of using Pearson:** it is **invariant to shifting and scaling**. If the truth is `y`, then predicting `2y + 100` scores a *perfect* 1.0. So we do **not** need to predict the absolute intensity correctly — we only need to get the **relative ranking / spread** of probes right, *for each protein separately*. This drives two design choices later: (a) we normalize each protein's targets to mean 0 / std 1 before training, and (b) a model that just outputs a per-protein constant scores **0** (constants have zero variance), so the model is forced to actually use the DNA.

### What the grading rewards (and what it implies for us)

From the submission slides, the project is 70% of the course; the breakdown that affects *engineering decisions* is:

| Component | Formula | What it means for design |
|---|---|---|
| Accuracy (30%) | `score / max_score`, score = mean Pearson over 64 DBPs | Maximize average per-protein Pearson. This is the main objective. |
| Rank (10%) | `r / k` among `k` projects | Same objective — be near the top. |
| Runtime (10%) | `max(min(1, 2 − time[s]/600), 0)` | **Prediction on all 64 DBPs must finish in ≤ 600 s for full marks.** At 900 s → 0.5; at 1200 s → 0. This is generous, but it forbids running ESM-2 at prediction time (see §7). |
| Implementation (10%) | modularity, clarity, docs | This is why the project is structured and config-driven (§9). |
| Report (10%) | the design doc / final report | This document is the seed of it. |

---

## 2. What the data actually looks like (verified)

I inspected the files directly:

| File | Count | Notes |
|---|---|---|
| `training_DBPs.txt` | 400 proteins (387 unique) | 1 per line; lengths 74–832, median ~106; alphabet = 20 standard AAs + `X` |
| `training_seqs.txt` | 30,000 DNA probes | length 36, `ACGT` only, sorted lexicographically |
| `training_data.zip` | **(binding intensities)** | ⚠️ **NOT in the uploads — you must add it.** Expected: the binding intensity of each of the 400 proteins for each of the 30,000 probes. |
| `test_DBPs.txt` | 64 proteins | lengths 43–455, median ~101 |
| `test_seqs.txt` | 11,728 DNA probes | length 36, `ACGT` only |
| `baseline_results.txt` | 20 Pearson scores | mean **0.208**, min 0.034, max 0.380 — the bar to beat |

**Critical findings:**

1. **No overlap.** Zero proteins and zero DNA probes are shared between train and test. This is a genuine **zero-shot generalization** task on both axes. (This is why per-protein models like DeepBind don't apply directly — we need a protein *encoder*.)
2. **Family structure.** Many test proteins contain `WFQNRR` (homeodomain signature) or `CxxC…HxxxH` patterns (zinc fingers). Unseen test proteins are *homologs* of training proteins. This is precisely the regime where a pretrained protein language model shines: similar sequences → similar embeddings → transferable binding rules. (This is the same insight behind ProbeRating and Affinity Regression.)
3. **All proteins fit ESM-2.** Max length 832 < ESM-2's 1022-token limit, so no truncation is needed (we handle it defensively anyway).
4. **13 duplicate proteins** in training — worth de-duplicating so the loss isn't double-weighted.

---

## 3. The core idea: a two-tower recommender

Think of it like Netflix. Netflix predicts how much a **user** will like a **movie** from features of each. We predict how much a **protein** will bind a **DNA probe** from features of each. Binding intensity is the "rating."

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

**Why two towers (and not one big network eating both)?**

- **Generalization:** the protein tower learns "what kind of binder is this," the DNA tower learns "what does this site look like," and the head learns "do they match." Each piece is reusable and the rule transfers to unseen proteins.
- **Speed:** protein embeddings can be **precomputed once** (there are only 464 proteins total). At prediction time we only run the cheap DNA tower + head. This is the key to the runtime budget.
- **Clarity:** each tower is independently testable and swappable.

---

## 4. Architecture in detail

### 4.1 Protein tower — pretrained embeddings (ESM-2)

**What it is.** ESM-2 is a transformer trained on ~tens of millions of natural protein sequences. Feed it a sequence of amino acids and it returns one vector per residue that encodes structural/functional context. We **mean-pool** those per-residue vectors into a single fixed-length protein vector, then pass it through a small trainable MLP to project into our shared space.

**Default checkpoint:** `facebook/esm2_t12_35M_UR50D` (35M params, **480-dim** output). Verified options for the config:

| Checkpoint | Layers | Params | Embedding dim | Use when |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D` | 6 | 8M | 320 | fastest, weakest — quick experiments |
| `esm2_t12_35M_UR50D` | 12 | 35M | **480** | **default** — best speed/quality trade-off |
| `esm2_t30_150M_UR50D` | 30 | 150M | 640 | if accuracy needs a push and you have a GPU |

**Why pretrained instead of learning amino-acid embeddings from scratch?** (This addresses your "torch.nn embedding with updating coding + positional vector" idea.) With only ~400 training proteins, a from-scratch encoder has far too little data to learn what makes proteins similar, and it would generalize poorly to the unseen test proteins. ESM-2 has already learned that from millions of proteins for free. **We keep your idea as a swappable alternative encoder** (a learnable `nn.Embedding(25, d)` over amino acids + sinusoidal/learned positional encodings + a tiny transformer/CNN, mean-pooled). It plugs into the same `ProteinEncoder` interface so you can A/B test "ESM-2 vs learned-from-scratch" and *see* the gap yourself. Expectation: ESM-2 wins clearly, but running the comparison is instructive and makes a great paragraph in the report.

**Frozen by default.** We do **not** fine-tune ESM-2 at first. We run it once, cache the 464 protein vectors to disk, and train only the small towers/head on top. This is faster, simpler, needs no GPU for training, and is already strong. *Fine-tuning ESM* is listed as an experiment in §10 (higher ceiling, much slower, easy to overfit 400 proteins).

### 4.2 DNA tower — a learnable motif scanner (1D CNN)

**What it is.** We one-hot encode the 36 bp probe into a `4 × 36` matrix (rows = A,C,G,T). A 1D convolution slides filters of width ~11–15 across it; each filter activates when it sees a particular short pattern. We take ReLU, then **global max-pool** over the length (β: "does this motif appear *anywhere* in the probe, and how strongly?"), giving one number per filter. A small MLP projects the filter activations to the shared DNA vector `d`.

> **Direct tie to your course slides.** The slides ask: *"How can we use a NN for a PWM model?"* A 1D convolution filter over one-hot DNA **is literally a Position Weight Matrix scanner** — the filter weights are the per-position, per-base scores, and the convolution computes the PWM match score at every offset. Global max-pool = "best match anywhere." So the DNA CNN is the neural-network version of the exact PWM picture in the lecture. This is the DeepBind (Alipanahi et al., 2015) idea. Saying this in your report shows you connected the dots.

**Reverse complement (your idea — and it's a good one).** A protein can bind either DNA strand, and the array only reports one strand. So a motif on the reverse-complement strand should count too. We handle it cleanly:

- Compute the reverse complement of each probe (`A↔T`, `C↔G`, then reverse the string).
- Run **both** the forward and the RC probe through the **same** CNN (shared weights), and **average** (or max) the two DNA vectors.

This roughly doubles the effective signal for strand-symmetric motifs at almost no cost, and it's a config toggle (`use_reverse_complement: true`). It is also a clean ablation for the report (with vs without).

**Default CNN shape (tunable in config):**
- `Conv1d(in=4, out=128, kernel_size=15, padding=7)` → `ReLU` → `AdaptiveMaxPool1d(1)` → 128 numbers
- (optional second conv for a touch more capacity)
- `Linear(128 → 128)` projection → `d`

Kernel ~11–15 matches typical TF motif lengths (6–20 bp). A nice cheap upgrade is **multiple kernel widths in parallel** (e.g., 9, 15, 21) concatenated — captures short and long motifs. Start with one width; add this as an experiment.

### 4.3 Interaction head — how the two vectors meet

Both towers output the same dimension `D` (e.g., 128). The head must let the protein and DNA vectors *interact* (not just sit side by side), otherwise the model can cheat by emitting a per-protein constant (which Pearson scores as 0).

**Default head (simple but forces interaction):**

```
combine(p, d) = concat([ p, d, p ⊙ d ])      # ⊙ = element-wise product
                → Linear(3D → D) → ReLU → Dropout → Linear(D → 1)
```

The `p ⊙ d` term is the "do these match" signal: it's large in dimensions where the protein "wants" something and the DNA "has" it. This is the factorization-machine flavor of a recommender, kept to two extra lines of code.

**Alternatives to test (§10):**
- **Pure concat** `[p, d]` (even simpler; good ablation to show the product helps).
- **Bilinear / dot product** `pᵀ W d` (closest to Affinity Regression's protein-kmer × DNA-kmer interaction).
- **FiLM:** let `p` produce scale+shift parameters that modulate the DNA features.

### 4.4 The model module = the orchestrator

A single `BindingModel(nn.Module)` **composes** `ProteinEncoder + DNAEncoder + InteractionHead`, all chosen from config. Swapping any component (different ESM size, learned-from-scratch protein encoder, bilinear head, RC on/off) is a **config change**, not a rewrite. That is the "main orchestrator for easy future changes" you asked for.

---

## 5. Data handling & preprocessing

### 5.1 Reading the data
A small `io` module reads proteins (one per line), DNA (one per line), and the intensities from `training_data.zip`. **⚠️ The exact structure of the zip is unconfirmed** (it isn't in the uploads). The most likely layout is one file per protein with 30,000 intensities aligned to `training_seqs.txt` order — but **the first build step is to unzip it and print its structure**, then make the reader match reality. The reader yields aligned `(protein_index, probe_index, intensity)` triples or a `[n_proteins × n_probes]` matrix.

### 5.2 Target normalization (important, because of Pearson)
PBM intensities are right-skewed with heavy tails. Per protein, we:
1. `log1p` transform (tames the tail), then
2. **z-score** (subtract that protein's mean, divide by its std).

We **train the model to predict the z-scored intensity.** Because Pearson is scale/shift-invariant per protein, the z-scored target is the *ideal* learning target, and it puts every protein on the same footing for the loss. (Both steps are config toggles so you can ablate `raw` vs `log1p` vs `log1p+zscore`.)

### 5.3 Train/validation split — protein-disjoint (mirror the real test)
To pick hyperparameters honestly, the validation set must look like the real test: **proteins the model never trained on.** So we split by **protein**, not by random pairs — e.g., **360 train proteins / 40 validation proteins**, all 30,000 probes each. We report **mean per-protein Pearson on the held-out proteins** and use it for early stopping and model selection. (The provided baseline did a simpler "first 380 proteins / first 27,000 probes" split; a clean random protein split is better practice and we'll note that.)

### 5.4 Handling ~11.6M pairs
387 proteins × 30,000 probes ≈ 11.6M training pairs — plenty. We don't need all of them every epoch. Each epoch, **subsample K probes per protein** (e.g., K=3,000 → ~1.16M pairs/epoch) so an epoch is fast. Protein vectors are precomputed and just looked up by index, so each training example is cheap (look up `p`, one-hot a 36 bp string, forward a tiny net).

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

## 7. Inference & the runtime budget (this is where 10% of the grade lives)

**The rule:** prediction over all 64 DBPs must be fast. Loading ESM-2 inside `main.py` would likely blow the budget (large import + model load, possibly repeated per call). **So ESM-2 must not run on the prediction path.**

**The plan:**
1. **Offline (once, during training prep):** run ESM-2 over all 400 train + 64 test proteins and save vectors to `artifacts/protein_embeddings.npz`, keyed by `DBP` index. This is a fixed feature-extraction step (no labels involved) — standard and fair.
2. **At prediction time, `main.py`:**
   - loads the small trained model (`artifacts/model.pt`, a few MB) and the embedding cache (tiny);
   - looks up the requested protein's cached vector (no ESM, no `transformers` import on the hot path — import lazily only as a fallback if a vector is missing);
   - reads the DNA file, one-hot encodes (+ reverse complement), runs the **DNA tower + head in batches**, writes one score per line in input order.

**Back-of-envelope:** 11,728 probes through a tiny CNN is milliseconds on GPU and a few seconds on CPU; ×64 proteins is comfortably under 600 s even on CPU, with no ESM cost. We use `torch.inference_mode()` and a sane batch size. A `runtime_test.py` script **replicates the real 64-call pattern** and prints total seconds so you can verify the efficiency score before submitting.

**Determinism:** inference is deterministic; we still set seeds and document CPU/GPU/memory in the report (required).

---

## 8. Evaluation & testing

### 8.1 Metrics module
- **`pearson_per_protein`** — the official metric. Average of per-protein Pearson over the evaluated proteins. **Unit-tested against `scipy.stats.pearsonr`** on random data.
- **Also report:** Spearman (rank correlation — robust sanity check), MSE/RMSE, and R². These help diagnose *why* a model is good or bad (e.g., high Spearman but low Pearson → a monotonic-but-nonlinear miss).
- **Edge cases handled & tested:** a constant prediction (zero variance → Pearson undefined) returns a defined sentinel (e.g., 0.0) rather than `NaN`/crash; identical inputs → 1.0; reversed inputs → −1.0.

### 8.2 Test suite (mirrors how *they* will grade + standard checks)
- `test_metrics.py` — Pearson matches scipy; constant/NaN edge cases; averaging logic.
- `test_dna.py` — one-hot shape `(4,36)` and correctness; reverse-complement correctness (`AACG → CGTT`); only `ACGT` accepted.
- `test_data.py` — protein/DNA file parsing; per-protein z-score has mean≈0/std≈1; protein-disjoint split shares no protein between train and val.
- `test_model.py` — forward pass returns the right shape; a fixed seed is reproducible; gradients flow to the towers but **not** into frozen ESM.
- `test_end_to_end.py` — a tiny **synthetic** dataset (a handful of fake proteins/probes with a planted rule) that the model can actually learn, proving the whole pipeline (train → predict → score) runs and that Pearson goes up. Also asserts `main.py` output has exactly one line per input probe, in order.
- A **planted-signal sanity test**: make a fake protein whose "binding" is high exactly when the probe contains `ACGTACGT`; the model should reach high Pearson on it. If it can't learn an obvious rule, something is wired wrong.

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

**Design principles baked in:**
- **Config is the single source of truth.** Architecture choices live in `config.yaml`. Changing the model = editing config, occasionally adding one class behind an existing interface. No surgery across files.
- **Clean interfaces.** `ProteinEncoder.encode(seqs) -> [N, Dp]`, `DNAEncoder(onehot) -> [N, Dd]`, `InteractionHead(p, d) -> [N]`, all wired by `BindingModel`. New ideas implement an interface and register in a small factory.
- **One orchestrator at the model level** (`BindingModel`) and **one at the pipeline level** (`main.py` / `predict.py`). Training, evaluation, and prediction all reuse the same components.
- **Separation of slow/offline (ESM, training) from fast/online (prediction).**

---

## 10. Ideas to test (prioritized experiment backlog)

Each is a small, isolated change with a clear hypothesis and a way to measure it (mean validation Pearson on held-out proteins). Run them one at a time, fill in the table, and the winners go in the report. **Start at the top; stop when gains flatten.**

| # | Idea | Hypothesis | Effort | Risk |
|---|---|---|---|---|
| 1 | **Reverse-complement averaging** (on/off) | Captures motifs on either strand → ↑ Pearson | tiny | none |
| 2 | **Target transform**: raw vs `log1p` vs `log1p+zscore` | z-score aligns with Pearson → ↑ & more stable | tiny | none |
| 3 | **Interaction head**: concat vs concat+product vs bilinear | product/bilinear forces real interaction → ↑ | tiny | none |
| 4 | **ESM-2 size**: 8M → 35M → 150M | bigger protein model → ↑ (diminishing) | small | slower precompute |
| 5 | **Multi-width CNN** (kernels 9/15/21 in parallel) | multi-scale motifs → ↑ | small | mild overfit |
| 6 | **CNN depth / filters** (1 vs 2 conv layers, 64 vs 128) | more capacity → ↑ up to a point | small | overfit |
| 7 | **Pearson / ranking loss** vs MSE | optimize the metric directly → ↑ | medium | needs protein-grouped batches |
| 8 | **Ensemble** (avg predictions of 3 seeds/configs) | variance reduction → small reliable ↑ | small | slower predict (watch budget) |
| 9 | **Learned-from-scratch protein encoder** (your `nn.Embedding`+positional idea) vs ESM-2 | ESM-2 should win; quantify the gap | medium | likely worse (that's the point) |
| 10 | **Fine-tune ESM-2** (unfreeze top layers) | higher ceiling | large | overfit 400 proteins; slow; may break runtime if used online |

A clean **ablation table** (each row = "default minus one feature → Δ Pearson") is one of the most convincing things you can put in the report, and these experiments produce it directly.

---

## 11. Risks & gotchas

- **`training_data.zip` is missing from the uploads.** Without intensities there is nothing to train on. *First action:* obtain it and run `inspect_data.py` to learn its exact format before writing the reader.
- **Don't run ESM on the prediction path.** It risks the 10% runtime score. Precompute + cache (§7).
- **Per-protein-constant trap.** If validation Pearson sits near 0, the head is emitting a per-protein constant — confirm the interaction term/DNA tower is actually used (z-scored targets + the `p⊙d` head prevent this).
- **Long proteins.** Max 832 < 1022, so fine — but clamp to ESM's limit defensively.
- **Duplicate proteins (13).** De-duplicate (or weight) so the loss isn't skewed.
- **CPU vs GPU.** Training is small enough for CPU; a GPU just makes ESM precompute and training faster. Prediction is CPU-friendly by design.
- **Determinism for the runtime test.** Set seeds; replicate the real 64-call pattern in `runtime_test.py`, not a single in-process loop, so the measured time matches grading.

---

## 12. Submission checklist (maps to grading)

- [ ] `training_data.zip` obtained and `inspect_data.py` output reviewed.
- [ ] `artifacts/protein_embeddings.npz` built (train + test proteins) and `artifacts/model.pt` trained.
- [ ] `python main.py out.txt DBP1 test_seqs.txt` runs and writes 11,728 lines in order.
- [ ] `scripts/predict_all.py` produces `DBP1.txt … DBP64.txt` and a single zip.
- [ ] `scripts/runtime_test.py` reports total prediction time **≤ 600 s** (target full efficiency mark).
- [ ] Validation **mean Pearson ≫ 0.208** (beats baseline) and ablation table filled.
- [ ] All tests in `tests/` pass.
- [ ] Design/final report (3–5 pages, this doc as seed): goal, I/O, architecture, parameters, pre/post-processing, parameter search, results on training data, performance (time/memory/CPU).
- [ ] Code submitted (modular, documented) + Python runtime-test script.

---

### One-line mental model to keep
> *Pretrained protein vector (ESM-2) **×** learnable DNA motif vector (CNN) → small MLP → binding score; protein vectors cached so prediction is fast; everything chosen from one config file.*
