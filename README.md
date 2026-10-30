# What Structured Pruning Actually Buys: Why Selection Criteria Stop Mattering

Code and recorded runs for the paper. Everything the paper prints is regenerated
from the runs shipped here by one command, on a CPU, in about a minute.

```bash
pip install -r requirements.txt
./reproduce.sh tables
```

---

## 1. What the paper shows

Structured pruning's two components are worth wildly different amounts, and the
field has been measuring them in a way that hides this. The layer-wise error
admits an exact orthogonal decomposition: the repair applied to the consuming
layer recovers precisely the part of the discarded contribution the surviving
units can express, while changing which units survive can move the residual only
by the spectral tail of the design matrix beyond the retained rank. The first
quantity is O(1) in the budget and the second vanishes, so selection criteria
must converge while the repair does not.

Across 3922 recorded compression runs on 25 checkpoints spanning fully
connected, convolutional and Transformer models, the repair is worth +44.1
accuracy points at the deployed operating point and wins 24 of 24 paired
comparisons, while replacing a target-blind criterion by a target-aware one is
worth +0.52. The theory predicts its own exception, on units whose design matrix
has no spectral tail, and that exception is confirmed on attention heads
at +14.4.

**This is a re-analysis paper.** It introduces no new compression operator and
trains no model to make its point. Three consequences shape this repository: the
accuracy evidence is a re-reading of CSVs written by the original experiment
drivers; the mechanism evidence is recomputed from checkpoints with no accuracy
evaluation in it at all; and checking the paper's arithmetic requires no GPU.

---

## 2. Every table and figure in the paper

`./reproduce.sh tables` writes all of these. The generated file names predate the
paper's numbering, so the mapping is given explicitly.

| Paper item | Generated file | Analysis behind it |
| --- | --- | --- |
| Table 1, the two residuals over 154 cells | `analysis/tables/t3_mechanism.tex` | `a7_mechanism.py`, `a8_theory.py` |
| Table 2, every checkpoint at its operating point | `analysis/tables/t7_master.tex` | `a6_at_the_frontier.py` |
| Table 3, every contrast at the operating point | `analysis/tables/t2_frontier.tex` | `a6_at_the_frontier.py` |
| Table 4, both effects against the budget | `analysis/tables/t1_regime.tex` | `a2_regime.py`, `a3_component_regime.py` |
| Table 5, published methods at matched budget | `analysis/tables/t9_comparison.tex` | `a4_who_wins_at_the_frontier.py`, `ablations/e3_thinet.py` |
| Table 6, the predicted reversal on heads | `analysis/tables/t5_attention.tex` | `compression_transformer/scripts/phase8_attention.py` |
| Figure 1, the decomposition | `analysis/figures/fig1_mechanism.pdf` | `make_fig_mechanism.py` |
| Figure 2, Corollary 1 tested directly | `analysis/figures/fig4_tail.pdf` | `make_fig_extra.py` |
| Figure 3, repair against criterion by budget | `analysis/figures/fig3_budget.pdf` | `make_fig_extra.py` |
| Figure 4, the same ordering per family | `analysis/figures/fig5_arch.pdf` | `make_fig_extra.py` |
| Section 6 spectral tails, quoted in prose | `analysis/tables/t8_spectra.tex` | `a9_spectra.py` |
| Every number quoted in the text | `analysis/results/paper_numbers.json` | `make_tables.py` |

`make_tables.py` also writes `t4_perckpt.tex` and `t6_frontiers.tex`, which the
paper does not print. `t4_perckpt` holds the per-checkpoint contrasts that
Table 2 carries as columns.

---

## 3. Installation

Python 3.10 or later and PyTorch 2.1 or later. A GPU is optional: the tier that
regenerates every table and figure runs on a CPU. The paper was produced with
Python 3.13 and PyTorch 2.7.1 on CUDA 12.6, pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

or `conda env create -f environment.yml && conda activate structured-pruning-decomposition`.

To regenerate only the tables and figures, `pip install numpy matplotlib` is
enough. The analysis layer imports the standard library plus `matplotlib`, and
deliberately does not use `pandas`, so it cannot silently reindex or coerce a
column. `torch` is needed only to touch checkpoints, and `transformers` only for
the Transformer study.

**Check the install.**

```bash
cd analysis/scripts && python3 load_all.py
```

```
records loaded: {'mlp': 2760, 'cnn': 677, 'transformer': 485} total 3922
distinct models: 25
```

If those three numbers appear, the whole CPU-only path will work.

---

## 4. Reproduction

Three tiers. Tier 1 checks the paper's arithmetic, tier 2 re-derives the
mechanism from weights, tier 3 reruns the experiments.

### Tier 1, arithmetic. No GPU, no datasets, no weights. About one minute.

```bash
./reproduce.sh tables
```

`analysis/results/paper_numbers.json` should then contain:

| Field | Expected | Where it appears in the paper |
| --- | --- | --- |
| `n_checkpoints` | 25 | abstract, Section 5 |
| `frontier_refit_merge` | 44.109 | abstract, Table 3 |
| `frontier_refit_magn` | 32.745 | Section 5 |
| `frontier_sel_span` | 0.515 | abstract, Table 3 |
| `ratio` | 86.0 | quoted as "a factor of 85" |
| `omp_not_winner` | 15 | abstract, "15 of 25 checkpoints" |

### Tier 2, mechanism. GPU. About 35 minutes.

The stronger verification, because it recomputes Section 4 from weights with no
accuracy evaluation anywhere.

```bash
./reproduce.sh mechanism     # Theorem 1 and Corollary 2, from checkpoints
./reproduce.sh spectra       # Section 6 spectral tails, downloads DistilBERT
```

This needs the eight fully connected checkpoints marked
`used_by_section4_residuals = True` in
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv`. They total 101 MiB, are not
shipped, and retrain in about 70 minutes:

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

Expected: maximum relative violation of the Theorem 1 identity about 5.7e-16,
agreement between the two routes to `R_fit` about 2.4e-8, zero violations of the
Corollary 2 bound in 154 cells, and a correction over selection residual ratio
rising from 7.63 to 28.82.

A retrained checkpoint will not match the original bit for bit, and does not need
to: every claim is a paired contrast within one model at one budget, so it
survives a fraction of a point of drift in dense accuracy.

### Tier 3, experiments. Weeks of GPU time.

```bash
python compression_mlp/run_autonomous_compression.py --model mnist_mlp2_bn
python compression_cnn/hidden_layer_compression/run_cnn_compression.py --model cifar10_vgg11_bn
python compression_transformer/scripts/phase3_sweep.py       # budget ladder
python compression_transformer/scripts/phase8_attention.py   # Section 6
python ablations/e3_thinet.py                                # Table 5
```

About 1 to 3 hours per fully connected sweep, 3 to 12 hours per convolutional
sweep, about 40 hours for the full Transformer study, and about 31 GPU hours to
retrain all 24 vision checkpoints.

### Self-tests

```bash
python compression_transformer/scripts/selftest.py    # needs GPU
python compression_cnn/common/selftest.py             # conv patch round-trip
```

The first proves that the Transformer selection reproduces the fully connected
study's `omp_select` index for index, that the refit matches a dense `lstsq` to
8e-08, that keeping every unit is a no-op, that no padding row enters the design
matrix, and that the under-determination guard fires.

---

## 5. Layout

```
.
├── README.md              this file
├── reproduce.sh           staged entry point
├── requirements.txt       pinned to the versions the paper used
├── environment.yml        conda equivalent
├── CITATION.cff
├── LICENSE
│
├── models/                network definitions: MLP, CNN, VGG, ResNet
├── utils/                 paths, config, data loading, metrics
├── intrinsic_rank/        GRIDE intrinsic-dimension estimator
├── trainer/               training loop; checkpoint manifest, not weights
│
├── compression_mlp/           STUDY 1  fully connected      2760 runs
├── compression_cnn/           STUDY 2  convolutional          677 runs
├── compression_transformer/   STUDY 3  DistilBERT MNLI        485 runs
├── ablations/                 targeted experiments e1 to e4
│
└── analysis/              every table, figure and number in the paper
    ├── scripts/           a1 to a10, make_tables, make_fig_mechanism, make_fig_extra
    ├── results/           analysis outputs as JSON
    ├── tables/            generated LaTeX fragments
    └── figures/           generated PDF and PNG
```

Each study directory keeps its own `results/` holding the CSVs its driver wrote.
Those CSVs are the input to `analysis/`, and they are what makes the tables
reproducible without a GPU.

**The directory depth is load-bearing.** Every script resolves its paths with
`Path(__file__).resolve().parents[N]` and manipulates `sys.path` relative to its
own file location, so the structure is the interface between modules. Moving
`analysis/` inside a `src/` folder, for example, would change `parents[2]` from
the project root to `src/` and break resolution silently rather than loudly. If
you reorganise, rerun `analysis/scripts/load_all.py` and confirm it still reports
3922 records over 25 models.

### What each directory holds

| Directory | Contents |
| --- | --- |
| `models/` | MLP, CNN, VGG, ResNet definitions and `model_factory.build_model`. Imported by the trainer and by every compression driver to rebuild a checkpoint. |
| `utils/` | `config.py` is the single source of paths and training defaults, honouring `OT_NEURON_COMPRESSION_ROOT`. `data_loader.py` builds transforms and loaders. `hf_datasets.py` pulls CIFAR from a parquet mirror. `imagenet32.py` reads the pickled ImageNet32 batches. `metrics.py` is accuracy. |
| `intrinsic_rank/` | The GRIDE estimator that sets the budget ladder. `id_estimator.py` is imported by `compression_transformer/scripts/core.py`. |
| `trainer/` | `train.py` produced the 24 vision checkpoints. `checkpoint/` holds the manifest with SHA-256 digests. `one_off/` holds the runner that trained `imagenet32_resnet50` under non-default settings. |
| `compression_mlp/` | Study 1. `run_autonomous_compression.py` is both the driver and, because `analysis/scripts/a7_mechanism.py` imports it, the reference definition of checkpoint loading and BatchNorm folding used in Section 4. Operators live in `novel_strategies/` and are loaded at runtime. |
| `compression_cnn/` | Study 2. `common/` holds the shared conv machinery: patch extraction, BN folding, layer discovery, surgery, rebuild, conv GRIDE, linear algebra. `hidden_layer_compression/` holds the driver and its operators. The Transformer study imports `common/linalg.py` from here. |
| `compression_transformer/` | Study 3. The phase scripts and a shared `core.py` carrying all 17 operators, the row sampler that excludes padding, the unit abstraction and the over-determination guard. `phase8_attention.py` produces the Section 6 reversal. |
| `ablations/` | Four targeted experiments run against the paper's claims after they had formed. `e1` seed variance, which is the noise band in Figure 3; `e2` calibration size; `e3` ThiNet reimplemented verbatim, which becomes Table 5; `e4` the Gauss-Newton metric with the refit disabled. |
| `analysis/` | The analysis layer. Everything the paper prints is generated here from the study CSVs. |

### Analysis scripts

| Script | Purpose | Inputs |
| --- | --- | --- |
| `load_all.py` | Normalises three CSV schemas into one `Run` record. The step that makes cross-architecture contrasts possible. | study CSVs |
| `a1_selection_vs_correction.py` | Pooled contrast before regime stratification | CSVs |
| `a2_regime.py` | How contrasts move with distance from dense | CSVs |
| `a3_component_regime.py` | Per component; gives Table 4 | CSVs |
| `a4_who_wins_at_the_frontier.py` | Which operator wins where; spread within the family | CSVs |
| `a5_collapse_law.py` | Accuracy collapse as the budget tightens | CSVs |
| `a6_at_the_frontier.py` | **The central result.** Every contrast at each checkpoint's own frontier | CSVs |
| `a7_mechanism.py` | Exact residuals on 154 layer-budget cells | 8 checkpoints, GPU |
| `a8_theory.py` | Numerical verification of Theorem 1 and Corollary 2 | `a7` |
| `a9_spectra.py` | Spectral tails, feed-forward against attention | DistilBERT, GPU |
| `a10_dim_vs_rank.py` | GRIDE dimension against spectral effective rank | `a7` output |
| `make_tables.py` | The LaTeX tables plus `paper_numbers.json` | the above |
| `make_fig_mechanism.py` | Figure 1 | `a8` output |
| `make_fig_extra.py` | Figures 2, 3 and 4 | `a6`, `a7`, `a8` outputs |

---

## 6. Configuration and protocol

There is no YAML or JSON configuration layer, by design. `utils/config.py` holds
paths, `SEED = 42` and `TRAIN_DEFAULTS`: batch 128, 30 epochs, lr 1e-3, early
stopping patience 5. Every path derives from `PROJECT_ROOT`, which honours the
`OT_NEURON_COMPRESSION_ROOT` environment variable, so the tree can be relocated
without editing code.

Compression settings are command-line arguments with per-driver defaults:
`--calib`, `--ridge` (1e-4), `--seed` (0), `--alphas`, `--target`.

Calibration uses unlabelled training inputs only, between 1,153 and 32,768
examples, sized so every least-squares system is at least twofold
over-determined; fits below that are refused at runtime. BatchNorm is folded into
the preceding linear map, so a unit is the pair `(w, b)`, and the ridge is
`1e-4` relative to the mean diagonal of `UᵀU`. Accuracy is measured on the
held-out test split.

**Seeds.** Training uses seed 42, compression drivers default to seed 0, and the
stability study uses seeds 0, 1 and 2 with the GRIDE dimensions re-measured per
seed, so the widths themselves move with the seed. Seeds control calibration
subsampling, GRIDE neighbour subsampling and k-means initialisation. They do not
make runs bit-identical: cuDNN autotuning and non-deterministic reduction order
remain. This is why the paper reports seed variance as a noise band rather than
claiming exact reproduction, and why every contrast is paired within a cell so
the shared noise cancels. Expect to reproduce the contrasts, not the accuracies.

**Hardware.** The recorded runs were produced on a single NVIDIA L40S.

---

## 7. Datasets and checkpoints

No dataset is needed to reproduce the tables and figures; they come from the
shipped CSVs. Datasets are needed only to retrain checkpoints or rerun a sweep.

| Dataset | Size | Obtained | Used by |
| --- | --- | --- | --- |
| MNIST | 12 MB | automatic, torchvision | 2 checkpoints |
| FashionMNIST | 30 MB | automatic, torchvision | 6 checkpoints |
| CIFAR-10 | 170 MB | automatic, Hugging Face parquet mirror | 5 checkpoints |
| CIFAR-100 | 170 MB | automatic, Hugging Face parquet mirror | 6 checkpoints |
| ImageNet32 | 4 GB | **manual**, needs an ImageNet account | 4 checkpoints |
| MNLI | 300 MB | automatic, Hugging Face `nyu-mll/glue` | Transformer study |

Everything except ImageNet32 downloads on first use. ImageNet32 ships as pickled
batches that must be placed at `data/raw/ImageNet32/`. Without them those four
checkpoints cannot be retrained, but every recorded run involving them is still
in the CSVs and every table and figure still reproduces.

Trained weights are **not** distributed: the 24 vision checkpoints total 1066
MiB, 101 for the fully connected, 397 for VGG and 568 for ResNet.
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv` lists each with its dataset,
architecture, family, byte size, SHA-256 digest, and a flag marking the eight
Section 4 requires. The DistilBERT MNLI checkpoint comes from the Hugging Face
Hub under its own terms and is not redistributed here.

---

## 8. Troubleshooting

**`ModuleNotFoundError: models` or `utils`.** Each driver inserts its own project
root into `sys.path` based on its file location. Do not move scripts between
directories or flatten the tree.

**`FileNotFoundError` on a `.safetensors` file.** Expected. Weights are not
shipped. See tier 2 above, or check your own against the SHA-256 digests in the
checkpoint manifest.

**Numbers differ in the last digits.** Expected. Runs are seeded but not
bit-deterministic. Contrasts should match, accuracies may not.

---

## 9. Citation

```bibtex
@inproceedings{anonymous2027structured,
  title     = {What Structured Pruning Actually Buys: Why Selection
               Criteria Stop Mattering},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
  note      = {Under review}
}
```

See also `CITATION.cff`. Licensed MIT; see `LICENSE`.
