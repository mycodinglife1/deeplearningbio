# Final Report — Predicting PBM Binding from Amino-Acid Sequence

*Deep Learning in Computational Biology — final project.*
*(Companion to [DESIGN.md](DESIGN.md), which holds the full rationale and the
experiment backlog. This report focuses on implementation, parameters, and
results.)*

---

## 1. Goal, input, output

**Goal.** For an unseen DNA-binding protein, predict the binding intensity of
each 36-bp DNA probe so that predictions *correlate* (per protein) with the true
PBM intensities. Proteins and probes in the test set are disjoint from training,
so this is a genuine zero-shot task.

**Interface.**
```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```
`<DBP_name>` is `DBP1`..`DBP64` (the 1-based line in `test_DBPs.txt`);
`<DNA_probe_file>` has one 36-bp `ACGT` sequence per line; `<output_file>` gets
one predicted intensity per line, in input order.

**Metric.** Mean per-protein Pearson correlation. The provided nearest-neighbor
baseline averages **0.208**.

---

## 2. Architecture

A **two-tower recommender** (DESIGN.md §3–4):

* **Protein tower** — `facebook/esm2_t12_35M_UR50D` (35M params, 480-d output),
  **frozen**. Per-residue embeddings are **mean-pooled over real residues**
  (CLS/EOS/pad masked out) to one vector per protein, run **once offline** and
  cached. A trainable `LayerNorm → Linear(480→128) → ReLU → Dropout` projects
  into the shared 128-d space. Keeping the protein side tiny limits overfitting
  the ~387 unique training proteins.
* **DNA tower** — one-hot `[4, 36]` → parallel `Conv1d` branches of widths
  `{9, 15, 21}` (each 128 channels) → ReLU → global max-pool → concat →
  `Linear → ReLU → Dropout` to 128-d. A 1-D conv over one-hot DNA *is* a
  learnable PWM scanner; max-pool = "best motif match anywhere" (DeepBind).
  Forward and reverse-complement strands pass through the **same** weights and
  are **averaged** — a protein may bind either strand. (We verified at full
  scale that RC helps here: 0.5906 with vs 0.5761 without; see §6.)
* **Interaction head** — the final model uses a **cross-attention head**: the
  protein vector is the query and the per-position DNA features are the
  keys/values, so the protein selectively attends to the sites it cares about
  (following TransBind). The attended DNA summary is combined with the protein
  via `[p, d, p⊙d] → MLP → 1`. The `p⊙d` term forces a real interaction so the
  model cannot cheat with a per-protein constant (which scores 0 under Pearson).
  A simpler `concat_product` head is also available and was our starting point
  (see §6).
* **Zero-shot regularization.** With only 387 training proteins, the protein
  side is the main overfitting risk, so during training we perturb each protein
  vector with Gaussian noise and random feature masking. This stops the model
  memorizing specific proteins and improves generalization to unseen ones.

`BindingModel` (`src/model.py`) composes the towers + head from config; every
choice above is a single `config.yaml` edit.

---

## 3. Pre- and post-processing

* **Intensities.** `training_data.zip → training_data.txt` is `[30000 probes ×
  400 proteins]`; we transpose to `[n_proteins, n_probes]`. The parsed matrix is
  cached to `artifacts/intensities.npy` for fast re-runs.
* **Duplicate proteins.** 400 → **387 unique**; replicate intensity rows are
  **averaged** (standard for PBM) so the loss isn't double-weighted and no
  protein straddles the train/val split.
* **Target transform.** Per protein: `log1p` (tames the heavy right tail) then
  **z-score** (mean 0, std 1). Because Pearson is per-protein and
  shift/scale-invariant, the z-scored target is the ideal learning signal.
* **No post-processing** is needed at predict time — Pearson is invariant to the
  monotone-linear scaling, so raw model outputs are written directly.

---

## 4. Training & parameter search

* **Split.** Protein-disjoint (90/10): the validation proteins are never seen
  in training, mirroring the real zero-shot test. We early-stop and select on
  **mean per-protein Pearson vs raw intensity**, not on loss.
* **Per epoch** we subsample 3,000 of the 30,000 probes per protein (~1.04M
  pairs/epoch) so epochs are fast while covering every protein.
* **Optimizer / loss.** AdamW, lr 1e-3, weight decay 1e-4; MSE on z-scored
  targets; batch size 512; up to 30 epochs, early-stopping patience 6.
* **Parameter search.** Component choices were validated by the ablation harness
  (`scripts/ablation.py`), each toggling one feature and reporting Δ val Pearson
  — see §6.

---

## 5. Results on training data

We evaluate in two settings. The **doubly-disjoint** setting — unseen proteins
**and** unseen probes — mirrors the real test exactly (and matches how the
provided baseline was built), so it is our headline metric.

| Model (evaluated on unseen proteins **and** unseen probes) | mean per-protein Pearson |
|---|---|
| Provided baseline | 0.208 |
| Our two-tower baseline (CNN + concat-product head) | 0.520 |
| **Final model (cross-attention head + zero-shot regularization)** | **0.560** |

The final model reaches **0.560** — **~2.7× the 0.208 baseline** — and improves
our own two-tower baseline by **+0.040**. On the easier protein-disjoint setting
(unseen proteins, training probes) the two-tower model reaches **0.591** mean
per-protein Pearson (Spearman 0.623). Per-epoch curves are logged in
`artifacts/train_log.json`.

---

## 6. Ablations (does each design choice help?)

Run with `python scripts/ablation.py --epochs <k>`; each row toggles one
feature off the default and reports Δ val Pearson. The table below used a
reduced setting (8 epochs, 150 train proteins) so all six variants train in
about an hour; the *relative* deltas are what matter.

| Variant | val Pearson | Δ vs default |
|---|---|---|
| default (RC + multi-kernel + product head + log1p+zscore) | 0.5295 | — |
| − reverse complement | 0.5606 | +0.0311 |
| head: concat only (no product) | 0.5193 | −0.0102 |
| − log1p | 0.5166 | −0.0129 |
| − per-protein z-score | 0.5054 | −0.0241 |
| single kernel [15] | 0.5149 | −0.0146 |

**Reading the table.** The product head (`p⊙d`), `log1p`, per-protein z-score,
and the multi-width CNN each *help* (removing any one lowers Pearson) — z-score
matters most. These are kept in the final model.

**A cautionary result on reverse complement.** At this *reduced* setting,
removing RC looked like a clear win (+0.0311). We did not take that at face
value: we retrained **both** RC variants at full scale (348 proteins, 30
epochs, early stopping) and the result **flipped**:

| Full-scale model | best val Pearson |
|---|---|
| **RC on (shipped)** | **0.5906** |
| RC off | 0.5761 |

At full training scale RC averaging behaves like useful data augmentation and
**improves** generalization to unseen proteins, so we keep
`use_reverse_complement: true`.

### 6.1 What we tried next, in order

**1. A larger protein model (ESM-2 650M) — did not help.** We swapped the 35M
encoder for `esm2_t33_650M` (1280-d embeddings, ~18× the parameters), recomputed
all embeddings, and retrained. Accuracy did **not** improve (0.586 vs 0.591 mean
per-protein Pearson on the protein-disjoint set) while the model is far heavier,
so we kept the 35M encoder. Because ESM runs offline and is cached, encoder size
does **not** affect prediction runtime — a bigger model is a free experiment on
the runtime score, it simply did not pay off on accuracy here.

**2. Cross-attention head + zero-shot regularization (B+C) — our main
improvement.** We replaced the concat-product head with a **cross-attention
head** (the protein attends to the per-position DNA features) and added
**zero-shot regularization** on the protein vectors (Gaussian noise + feature
masking during training). On the honest doubly-disjoint evaluation (unseen
proteins **and** unseen probes) this raised mean per-protein Pearson from
**0.520 → 0.560 (+0.040)**. Prediction stays comfortably within budget
(64 DBPs in ~152 s in-process, efficiency 1.00), so the accuracy gain is free on
the runtime score. This is the configuration we ship.

**3. Final training on all data (no validation split).** For the submitted
model we retrained the winning configuration on **all 387 unique proteins with
no validation holdout**, using the number of epochs identified by the validated
sweep. Every protein contributes signal to the final model; because the epoch
count is fixed ahead of time from prior validation, no held-out set is needed to
decide when to stop.

---

## 7. Performance (time, memory, CPU)

* **Prediction (the graded path).** All 64 DBPs over the 11,728 test probes with
  the final cross-attention model: **~152 s** in-process → efficiency
  `max(min(1, 2 − t/600), 0) = 1.000`. The simpler pooled-head model runs in
  ~75 s; both are well inside the 600 s budget. Cross-attention is ~2× the
  pooled head but still comfortably within budget.
* **No ESM at predict time** — `main.py` loads a few-MB model + a tiny embedding
  cache and runs only the DNA tower + head under `torch.inference_mode()`.
* **Offline ESM precompute.** All 464 proteins encoded once in ~124 s on CPU.
* **Hardware.** Everything runs on **CPU** (no GPU required); CUDA is used
  automatically if present. Peak memory is modest (the intensity matrix and
  one-hot probes dominate at a few hundred MB during training).

---

## 8. Reproducing

See [README.md](README.md): `inspect_data → precompute_embeddings → train →
main.py / predict_all → runtime_test`, plus `pytest` (offline, no ESM).
