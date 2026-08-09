# PBM Binding Predictor — two-tower recommender for protein–DNA binding

Predict the **binding intensity** of a DNA-binding protein (DBP) for 36-bp DNA
probes, for **unseen proteins and unseen probes** (a zero-shot Protein Binding
Microarray task). The model is a two-tower recommender:

```
protein (amino acids) ─► ESM-2 (frozen) ─► mean-pool ─► projection ─► p ┐
                                                                        ├─► interaction head ─► binding score
DNA probe (36 bp)     ─► one-hot (+RC)  ─► 1-D CNN   ─► projection ─► d ┘
```

* **Protein tower:** a pretrained protein language model (ESM-2), run **once
  offline** and cached, plus a small trainable projection.
* **DNA tower:** a 1-D CNN motif scanner (a learnable PWM; DeepBind-style),
  with reverse-complement averaging.
* **Head:** combines `[p, d, p⊙d]` → MLP → one number.

The evaluation metric is **mean per-protein Pearson correlation**; the provided
baseline averages **0.208** and we aim well above it.

---

## Why this design (the short version)

* Test proteins/probes are completely unseen, so we must learn a *general*
  protein↔DNA rule, not memorize. A two-tower recommender is the standard way.
* Pearson is shift/scale-invariant per protein, so we **z-score each protein's
  targets** (after `log1p`) — the ideal learning signal — and the `p⊙d` head
  forces the model to use the DNA (a per-protein constant scores 0).
* ESM-2 is **frozen and precomputed**, so prediction runs only the cheap DNA
  tower + head → well within the 600 s runtime budget. **ESM never runs on the
  prediction path.**

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

---

## Reproduce end-to-end

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

## The graded entrypoint

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
| Interaction head | `head.type` | `concat`, `concat_product` (default), `bilinear`, `film` |
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
