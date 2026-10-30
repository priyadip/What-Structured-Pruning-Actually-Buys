# Reproduction Guide

Three tiers, in increasing cost. Tier 1 checks the paper's arithmetic and needs
no GPU. Tier 2 re-derives the mechanism from weights. Tier 3 reruns the
experiments themselves.

`reproduce.sh` runs each tier in dependency order.

---

## Tier 1, arithmetic. No GPU, no datasets, no weights. About one minute.

```bash
./reproduce.sh tables
```

Regenerates every table and figure in the paper from the recorded runs shipped in
this artifact.

**What to check.** The file `analysis/results/paper_numbers.json` should contain:

| Field | Expected | Where it appears in the paper |
| --- | --- | --- |
| `n_checkpoints` | 25 | abstract, Section 5 |
| `frontier_refit_merge` | 44.109 | abstract, Table 3 |
| `frontier_refit_magn` | 32.745 | Section 5 |
| `frontier_sel_span` | 0.515 | abstract, Table 3 |
| `ratio` | 86.0 | quoted as "a factor of 85" |
| `omp_not_winner` | 15 | abstract, "15 of 25 checkpoints" |

These were verified on a clean copy of this tree.

**What it writes.** Nine `.tex` fragments to `analysis/tables/`, eight PDF and
PNG pairs to `analysis/figures/`, thirteen JSON files to `analysis/results/`.

---

## Tier 2, mechanism. GPU. About 35 minutes.

This is the stronger form of verification, because it recomputes Section 4 from
weights and involves no accuracy evaluation anywhere.

```bash
./reproduce.sh mechanism     # Theorem 1 and Corollary 2, from checkpoints
./reproduce.sh spectra       # Section 6 spectral tails, downloads DistilBERT
```

**Prerequisite.** The eight fully connected checkpoints listed in
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv` with
`used_by_section4_residuals = True`. They total 101 MiB and are not shipped.
Retrain them in about 70 minutes:

```bash
python -m trainer.train --dataset MNIST        --model mlp2_bn
python -m trainer.train --dataset MNIST        --model mlp3_bn
python -m trainer.train --dataset FashionMNIST --model mlp2_bn
python -m trainer.train --dataset FashionMNIST --model mlp3_bn
python -m trainer.train --dataset CIFAR10      --model mlp3_quad_bn
python -m trainer.train --dataset CIFAR10      --model mlp3_oct_bn
python -m trainer.train --dataset CIFAR100     --model mlp3_quad_bn
python -m trainer.train --dataset CIFAR100     --model mlp3_oct_bn
```

**What to check.**

| Quantity | Expected |
| --- | --- |
| Maximum relative violation of the Theorem 1 identity | about 5.7e-16 |
| Agreement between the two routes to `R_fit` | about 2.4e-8 |
| Violations of the Corollary 2 bound | 0 of 154 |
| Correction / selection residual ratio | 7.63 rising to 28.82 |

A retrained checkpoint will not match the original bit for bit. That does not
matter: every claim is a paired contrast within one model at one budget, so the
contrast survives a fraction of a point of drift in dense accuracy.

---

## Tier 3, experiments. Weeks of GPU time.

Only necessary to challenge the recorded runs rather than the analysis of them.

```bash
# Study 1, fully connected
python compression_mlp/run_autonomous_compression.py --model mnist_mlp2_bn

# Study 2, convolutional
python compression_cnn/hidden_layer_compression/run_cnn_compression.py \
       --model cifar10_vgg11_bn

# Study 3, Transformer
python compression_transformer/scripts/phase3_sweep.py       # budget ladder
python compression_transformer/scripts/phase8_attention.py   # Section 6

# Targeted ablations
python ablations/e3_thinet.py                                # Table 5
```

Approximate cost: 1 to 3 hours per MLP sweep, 3 to 12 hours per CNN sweep, about
40 hours for the full Transformer study, about 31 GPU hours to retrain all 24
vision checkpoints.

---

## Mapping paper artifacts to commands

| Paper artifact | Command | Tier |
| --- | --- | --- |
| Table 1, the two residuals | `./reproduce.sh mechanism` then `tables` | 2 |
| Table 2, master per checkpoint | `./reproduce.sh tables` | 1 |
| Table 3, contrasts at the operating point | `./reproduce.sh tables` | 1 |
| Table 4, both effects by band | `./reproduce.sh tables` | 1 |
| Table 5, published methods | `./reproduce.sh tables` | 1 |
| Table 6, the attention reversal | `./reproduce.sh tables` | 1 |
| Figure 1, the decomposition | `./reproduce.sh tables` | 1 |
| Figure 2, Corollary 2 tested | `./reproduce.sh tables` | 1 |
| Figure 3, effects against budget | `./reproduce.sh tables` | 1 |
| Figure 4, per architecture family | `./reproduce.sh tables` | 1 |
| Section 4 residual verification | `./reproduce.sh mechanism` | 2 |
| Section 6 spectral tails | `./reproduce.sh spectra` | 2 |

Every table and figure in the paper is reachable from tier 1, because the
analysis outputs that tier 2 would regenerate are shipped as JSON.

---

## Self-tests

Three checks ship with the code and are worth running before trusting anything.

```bash
python compression_transformer/scripts/selftest.py    # 4 assertions, needs GPU
python compression_cnn/common/selftest.py             # conv patch round-trip
python compression_transformer/scripts/verify_report.py  # re-derives 118 numbers
```

The first proves that the Transformer selection reproduces the fully connected
study's `omp_select` index for index, that the refit matches a dense `lstsq` to
8e-08, that keeping every unit is a no-op, that no padding row enters the design
matrix, and that the under-determination guard fires.

---

## Troubleshooting

**`ModuleNotFoundError: models` or `utils`.** Each driver inserts its own project
root into `sys.path` based on its file location. Do not move scripts between
directories or flatten the tree; the layout is the interface.

**`FileNotFoundError` on a `.safetensors` file.** Expected. Weights are not
shipped. See tier 2 above, or verify your own against the SHA-256 digests in
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv`.

**LaTeX reports `??` for `tab:master`.** `make_tables.py` did not finish. The
master table needs `load_all` over all three studies; if one CSV is missing the
table is skipped.

**Numbers differ in the last digits.** Expected. Runs are seeded but not
bit-deterministic; cuDNN autotuning and reduction order remain. Contrasts should
match, accuracies may not.

