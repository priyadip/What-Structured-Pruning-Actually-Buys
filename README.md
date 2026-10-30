# What Structured Pruning Actually Buys

*A Decomposition, and Why Selection Criteria Stop Mattering*

Code, recorded runs and the experimental record for the paper. The repository
root holds the code and the analysis pipeline, each study keeps the result CSVs
its driver wrote, and `docs/` holds the written record: the evidence map, the
hyperparameters, the training and evaluation protocol, the hardware, and what
each ablation changed.

| | |
| --- | --- |
| Install | `INSTALL.md` |
| Reproduce | `REPRODUCE.md` |
| Evidence map | `docs/CLAIM_TO_EVIDENCE.md` |
| Experimental record | `docs/` |

**Quick start.** `pip install -r requirements.txt && ./reproduce.sh tables`
regenerates every table and figure in the paper in about a minute, with no GPU,
no datasets and no trained weights.

### Where to start

**Five minutes.** Open `docs/CLAIM_TO_EVIDENCE.md`. It lists
every quantitative claim in the paper next to the file that carries it and the
script that produced it, including the one claim it finds weaker than the paper
states.

**Twenty minutes.** `./reproduce.sh tables` regenerates every table and figure
from the recorded runs and writes `analysis/results/paper_numbers.json`, whose
fields correspond one to one with the numbers quoted in the text. Verified on a
clean copy of this tree: `n_checkpoints = 25`, `frontier_refit_merge = 44.109`,
`frontier_refit_magn = 32.745`, `frontier_sel_span = 0.515`, `ratio = 86.0`,
`omp_not_winner = 15`.

**The mechanism rather than the arithmetic.** The central theoretical claim is
checked without evaluating any network: exact Frobenius residuals in double
precision on 154 layer-budget cells, recomputed from checkpoints, so it cannot
inherit an artefact of an accuracy pipeline. `./reproduce.sh mechanism` needs a
GPU and the eight fully connected checkpoints. Expect a maximum relative
violation of the Theorem 1 identity near 5.7e-16 and zero violations of the
Corollary 2 bound in 154 cells.

---

## 1. Overview

The paper shows that structured pruning's two components are worth wildly
different amounts, and that the field has been measuring them in a way that hides
this. The layer-wise error admits an exact orthogonal decomposition: the repair
applied to the consuming layer recovers precisely the part of the discarded
contribution the surviving units can express, while changing which units survive
can move the residual only by the spectral tail of the design matrix beyond the
retained rank. The first quantity is O(1) in the budget and the second vanishes,
so selection criteria must converge while the repair does not.

Across 3922 recorded compression runs on 25 checkpoints spanning fully connected,
convolutional and Transformer models, the repair is worth +44.1 accuracy points
at the deployed operating point and wins 24 of 24 paired comparisons, while
replacing a target-blind criterion by a target-aware one is worth +0.52. The
theory predicts its own exception, on units whose design matrix has no spectral
tail, and that exception is confirmed on attention heads at +14.4.

**This is a re-analysis paper.** It introduces no new compression operator and
trains no model to make its point. Three consequences shape this artifact: the
accuracy evidence is a re-reading of CSVs written by the original experiment
drivers; the mechanism evidence is recomputed from checkpoints with no accuracy
evaluation in it at all; and checking the paper's arithmetic requires no GPU.

---

## 2. Layout

```
.
├── README.md              this file
├── INSTALL.md             environment setup
├── REPRODUCE.md           the three reproduction tiers
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
├── compression_input_space/   supports one Section 7 claim only
├── ablations/                 targeted experiments e1 to e7
│
├── analysis/              every table, figure and number in the paper
│   ├── scripts/           a1 to a10, make_tables, make_fig_*
│   ├── results/           analysis outputs as JSON
│   ├── tables/            generated LaTeX fragments
│   └── figures/           generated PDF and PNG
│
└── docs/                  evidence map, experimental setup, hyperparameters,
                           training, evaluation, hardware, ablation outcomes
```

Each study directory keeps its own `results/` holding the CSVs its driver wrote.
Those CSVs are the input to `analysis/`, and they are what makes the tables
reproducible without a GPU.

The paper source is not included. The manuscript is the submission itself, and
the AAAI class files it needs are the venue's to distribute, not ours. What this
artifact provides is everything that *feeds* the manuscript: the tables and
figures in `analysis/tables/` and `analysis/figures/` are the exact fragments the
paper inputs, regenerated by `./reproduce.sh tables`.

### A note on the layout

Every script resolves its paths with `Path(__file__).resolve().parents[N]` and
manipulates `sys.path` relative to its own file location. **The directory
structure is the interface between modules.** Directories were renamed for this
artifact and every affected reference was patched, but the *depth* of each
directory is load-bearing: moving `analysis/` inside a `src/` folder, for
example, would change `parents[2]` from the project root to `src/` and break
resolution silently rather than loudly.

If you reorganise further, rerun `analysis/scripts/load_all.py` and confirm it
still reports 3922 records over 25 models.

---

## 3. What each directory holds

| Directory | Contents |
| --- | --- |
| `models/` | MLP, CNN, VGG, ResNet definitions and `model_factory.build_model`. Imported by the trainer and by every compression driver to rebuild a checkpoint. |
| `utils/` | `config.py` is the single source of paths and training defaults, honouring `OT_NEURON_COMPRESSION_ROOT`. `data_loader.py` builds transforms and loaders. `hf_datasets.py` pulls CIFAR from a parquet mirror. `imagenet32.py` reads the pickled ImageNet32 batches. `metrics.py` is accuracy. |
| `intrinsic_rank/` | The GRIDE estimator. `id_estimator.py` is imported by `compression_transformer/scripts/core.py`; the other estimators are archived. |
| `trainer/` | `train.py` produced the 24 vision checkpoints. `checkpoint/` holds the manifest with SHA-256 digests. `one_off/` holds the runner that trained `imagenet32_resnet50` under non-default settings. |
| `compression_mlp/` | Study 1. `run_autonomous_compression.py` is both the driver and, because `analysis/scripts/a7_mechanism.py` imports it, the reference definition of checkpoint loading and BatchNorm folding used in Section 4. |
| `compression_cnn/` | Study 2. `common/` holds shared conv machinery: patch extraction, BN folding, layer discovery, surgery, rebuild, conv GRIDE, linear algebra. `hidden_layer_compression/` holds the driver and its operators. `data_reduction_and_fit/` holds the patch-rank input stage that `ablations/e5` and `e7` build on; its code and arch specs are here, its saved weights are not. The Transformer study imports `common.linalg` from here. |
| `compression_transformer/` | Study 3. Nine phase scripts, a shared `core.py` carrying all 17 operators, a self-test, and `verify_report.py`, which re-derives every headline number in `REPORT.md` from the raw CSVs. |
| `compression_input_space/` | Input-space reduction. Supports **only** the frozen-network half of the Section 7 metric claim. Included because that number comes from nowhere else; see the evidence map. |
| `ablations/` | Seven targeted experiments run to attack the paper's claims after they had formed. `e3` became Table 5. `e4` did not confirm what it was built to confirm, which the evidence map records. |
| `analysis/` | The analysis layer. Everything the paper prints is generated here from the study CSVs. |
| `docs/` | The written record: `CLAIM_TO_EVIDENCE.md` maps every quantitative claim to the artefact that carries it, alongside the experimental setup, hyperparameters, training and evaluation protocol, hardware, and what each ablation changed. |

---

## 4. Key scripts

### Analysis, `analysis/scripts/`

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
| `make_tables.py` | All nine LaTeX tables plus `paper_numbers.json` | the above |
| `make_fig_mechanism.py` | Figure 1 | `a8` output |
| `make_fig_extra.py` | Figures 2, 3 and 4 | `a6`, `a7`, `a8` outputs |
| `compress_figures.py` | Compressed PNG copies, about 70 percent smaller | figures |

### Study drivers

| Script | Purpose |
| --- | --- |
| `compression_mlp/run_autonomous_compression.py` | Study 1 driver; sweeps operators against the GRIDE budget ladder |
| `compression_cnn/hidden_layer_compression/run_cnn_compression.py` | Study 2 driver; treats convolution as a linear map on extracted patches |
| `compression_transformer/scripts/core.py` | Study 3 core: all 17 operators, the row sampler that excludes padding, the unit abstraction, the over-determination guard |
| `compression_transformer/scripts/phase8_attention.py` | Produces the Section 6 reversal |
| `ablations/e3_thinet.py` | ThiNet Algorithm 1 reimplemented verbatim, then decomposed; Table 5 |

### Self-tests

| Script | What it proves |
| --- | --- |
| `compression_transformer/scripts/selftest.py` | Transformer selection reproduces the fully connected study's `omp_select` index for index; the refit matches a dense `lstsq` to 8e-08; keeping everything is a no-op; no padding row enters the design matrix; the under-determination guard fires |
| `compression_cnn/common/selftest.py` | Conv patch extraction and surgery round-trip |
| `compression_transformer/scripts/verify_report.py` | Re-derives all 118 numbers in the Transformer study report |

---

## 5. Configuration

There is no YAML or JSON configuration layer, by design.

`utils/config.py` holds paths, `SEED = 42` and `TRAIN_DEFAULTS` (batch 128,
30 epochs, lr 1e-3, patience 5). Every path derives from `PROJECT_ROOT`, which
honours the `OT_NEURON_COMPRESSION_ROOT` environment variable, so the tree can be
relocated without editing code.

Compression settings are command-line arguments with per-driver defaults:
`--calib`, `--ridge` (1e-4), `--seed` (0), `--alphas`, `--target`,
`--patch-budget`. The complete table is in
`docs/hyperparameters.md`.

---

## 6. Datasets

**No dataset is needed to reproduce the tables and figures.** They come from the
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
batches that must be placed at `data/raw/ImageNet32/`. If they are absent, those
four checkpoints cannot be retrained, but every recorded run involving them is
still in the CSVs and every table and figure still reproduces.

---

## 7. Checkpoints

Trained weights are **not** distributed: the 24 vision checkpoints total 1066
MiB. `trainer/checkpoint/CHECKPOINT_MANIFEST.csv` lists each with its dataset,
architecture, family, byte size, SHA-256 digest, and a flag marking the eight
that Section 4 requires.

| Group | Count | Size |
| --- | --- | --- |
| Fully connected | 8 | 101 MiB |
| VGG | 8 | 397 MiB |
| ResNet | 8 | 568 MiB |

Retrain the Section 4 subset in about 70 minutes; see `REPRODUCE.md` tier 2.

---

## 8. Random seeds and determinism

Training uses seed 42. Compression drivers default to seed 0. The stability study
uses seeds 0, 1 and 2, with the GRIDE dimensions re-measured per seed, so the
widths themselves move with the seed.

Seeds control calibration subsampling, GRIDE neighbour subsampling and k-means
initialisation. They do **not** make runs bit-identical: cuDNN autotuning and
non-deterministic reduction order remain. This is why the paper reports seed
variance as a noise band rather than claiming exact reproduction, and why every
contrast is paired within a cell so the shared noise cancels. Expect to reproduce
the contrasts, not the accuracies.

---


## 9. Citation

```bibtex
@inproceedings{anonymous2027structured,
  title     = {What Structured Pruning Actually Buys: A Decomposition,
               and Why Selection Criteria Stop Mattering},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027},
  note      = {Under review}
}
```

See also `CITATION.cff`. Licensed MIT; see `LICENSE`. The DistilBERT MNLI
checkpoint is obtained from the Hugging Face Hub under its own terms and is not
redistributed here.
