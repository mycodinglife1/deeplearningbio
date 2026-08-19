# PBM Binding Predictor — two-tower recommender for protein–DNA binding

Our mission is to predict the **binding intensity** of a DNA-binding protein (DBP) for 36-bp DNA
probes, for **unseen proteins and unseen probes**.
To do so, we'll use a two-tower recommender (one tower for the protein and one for the DNA)

```
protein (amino acids) ─► ESM-2 ─► mean-pool ─► projection ─► p ┐
                                                                        ├─► interaction head ─► binding score
DNA probe (36 bp)     ─► one-hot (+RC)  ─► 1-D CNN   ─► projection ─► d ┘
```

* **Protein tower:** for this tower we'll just use the pretrained protein language model (ESM-2),
  and then standardize and adapt the reults using average pooling and a projection.
* **DNA tower:** for this tower we'll use one-hot encoding and a CNN,
  making sure to use reverse-complement averaging to account for both strands.
* **Head:** and then we'll just merge the reults and use a multilayer perceptron layer to output a binding score.

The evaluation metric is **mean per-protein Pearson correlation** with provided
baseline averages **0.208** and we're aiming to beat it.

> **Final model:** a cross-attention head (the protein attends to the DNA) plus
> zero-shot regularization on the protein vectors, trained on all 387 proteins.
> It reaches **0.560** mean per-protein Pearson on unseen proteins **and** unseen
> probes — ~2.7× the 0.208 baseline. See [REPORT.md](REPORT.md) §5–6 for the full
> results and the experiments that got us there.

---

## Why this design

* Test proteins/probes are completely unseen, so we must learn a general rule,
  not memorize. A two-tower recommender is the standard way.
* Pearson is shift/scale-invariant per protein, so we **z-score each protein's
  targets** (after `log1p`) to standardize values.
* To force the model to use the DNA we do element-wise multiplication `p⊙d`.
* The runtime is in budget because ESM-2 is already computed,
  and running the DNA path is not so expensive.

Full rationale, data facts, and the experiment backlog are in
[DESIGN.md](DESIGN.md).

---

## Setup

```bash
pip install -r requirements.txt
```

Place the provided data in `./data/` (gitignored):

```
data/training_DBPs.txt      400 proteins (one per line)
data/training_seqs.txt      30,000 DNA probes (length 36)
data/training_data.zip      intensities: training_data.txt = [30000 probes x 400 proteins]
data/test_DBPs.txt          64 proteins
data/test_seqs.txt          11,728 DNA probes
data/baseline_results.txt   baseline Pearson scores (mean 0.208)
```

> The intensity matrix is stored **probe-major** (one line per probe, one value
> per protein); `src/data/io.load_intensities` transposes it to
> `[n_proteins, n_probes]`. Run `python scripts/inspect_data.py` to confirm.
> (Data placement is only needed to *retrain*; it is **not** needed for grading.)

---

## Running the model (for grading — no training needed)

**The trained model (`artifacts/model.pt`) and the precomputed protein embeddings
(`artifacts/protein_embeddings.npz`) are already included.** There is nothing to
train, download, or precompute — prediction reads these files directly. ESM-2 is
**never** loaded at prediction time (no `transformers` import, no internet, no
GPU required).

The graded entrypoint scores one protein against a DNA probe file:

```bash
pip install -r requirements.txt          # one-time: torch, numpy, pyyaml

# Score protein DBP1 for every probe in a DNA file (one score per line, in order)
python main.py <output_file> <DBP_name> <DNA_probe_file>
# e.g.
python main.py DBP1.txt DBP1 data/test_seqs.txt
```

- `<DBP_name>` ∈ `DBP1`..`DBP64` (the 1-based line number in `test_DBPs.txt`).
- Produce **all 64** score files + `submission.zip` at once: `python scripts/predict_all.py`
- Time the full 64-DBP prediction (efficiency score): `python scripts/runtime_test.py`

That is all the graders need to run. The section below is only for reproducing
the model *from scratch* — **not required for grading.**

---

## Reproduce from scratch (optional — not needed for grading)

```bash
# 0. Inspect the data and confirm the intensity format.
python scripts/inspect_data.py

# 1. Precompute ESM-2 protein embeddings ONCE (downloads ESM weights; CPU OK).
python scripts/precompute_embeddings.py            # -> artifacts/protein_embeddings.npz

# 2. Train (protein-disjoint split; early-stops on val mean per-protein Pearson).
python scripts/train.py                            # -> artifacts/model.pt + train_log.json
#    Quick smoke run:  python scripts/train.py --epochs 2 --max-proteins 30

# 3. Predict for one DBP (the graded interface).
python main.py out.txt DBP1 data/test_seqs.txt     # writes 11,728 ordered numbers

# 4. Produce all 64 score files + submission.zip.
python scripts/predict_all.py                      # -> submission/DBP1.txt..DBP64.txt + submission.zip

# 5. Time the 64-DBP prediction and print the efficiency score.
python scripts/runtime_test.py                     # real per-call pattern
python scripts/runtime_test.py --in-process        # single-load lower bound

# 6. (Optional) Ablation table for the report.
python scripts/ablation.py --epochs 10
```

Run the test suite (offline, no ESM download needed):

```bash
pytest
```

---

## Entrypoint

```
python main.py <output_file> <DBP_name> <DNA_probe_file>
```

* `<DBP_name>` ∈ `DBP1`..`DBP64` — the 1-based line number in `test_DBPs.txt`.
* `<DNA_probe_file>` — one 36-bp `ACGT` sequence per line.
* `<output_file>` — one predicted intensity per line, **same order** as input.

`main.py` loads the small `model.pt` + the cached protein vectors and runs only
the DNA tower + head (`torch.inference_mode()`); it does **not** import
`transformers`.

---

## Configuration — everything is a config edit

All paths, hyperparameters, and architecture choices live in
[config.yaml](config.yaml) and load into a validated `Config` dataclass
(`src/config.py`). Swapping a component is a config change, not a code change:

| Knob | Where | Options |
|---|---|---|
| Protein encoder | `protein_encoder.type` | `esm2` (default), `learned` |
| ESM size | `protein_encoder.esm_model` | `esm2_t6_8M` / `t12_35M` (default) / `t30_150M` |
| Reverse complement | `dna_encoder.use_reverse_complement` | `true` / `false` |
| CNN kernels | `dna_encoder.kernel_sizes` | e.g. `[15]` or `[9, 15, 21]` |
| DNA encoder | `dna_encoder.type` | `cnn` (default), `cnn_transformer`, `moe` |
| Interaction head | `head.type` | `concat`, `concat_product`, `bilinear`, `film`, `cross_attention` (final model) |
| Zero-shot regularization | `train.protein_noise_std`, `train.protein_mask_prob` | e.g. `0.024`, `0.1` |
| Target transform | `target.log1p`, `target.per_protein_zscore` | `true` / `false` |
| Loss | `train.loss` | `mse`, `huber`, `pearson` |

---

## Project layout

```
main.py                     graded entrypoint (fast; no ESM)
config.yaml                 single source of truth
src/
  config.py                 typed config + validation
  utils.py                  seeds, device, logging, reverse-complement
  data/{io,dna,dataset}.py  reading, one-hot, normalization + protein-disjoint split
  encoders/protein.py       ESM2Encoder + LearnedAAEncoder + embedding cache I/O
  encoders/dna_cnn.py       1-D CNN motif scanner
  heads.py                  concat / concat_product / bilinear / FiLM
  model.py                  BindingModel orchestrator (RC handling)
  train.py                  training loop, early stopping on val Pearson
  predict.py                fast prediction path (used by main.py)
  evaluate.py               Pearson-per-protein + Spearman + MSE/R²
scripts/                    inspect_data, precompute_embeddings, train,
                            predict_all, runtime_test, ablation
tests/                      offline pytest suite
artifacts/                  model.pt, protein_embeddings.npz, run logs
```

## Performance logging

Training writes `artifacts/train_log.json` (config + per-epoch val Pearson +
best score + elapsed time), and `runtime_test.py` reports the 64-DBP prediction
time and efficiency score — everything needed for the report's results section.
