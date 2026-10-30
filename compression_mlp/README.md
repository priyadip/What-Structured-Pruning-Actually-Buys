# `compression_mlp/` — autonomous zero-shot neuron compression at the GRIDE budget

Self-contained research folder. Everything the experiment needs — driver,
strategies, logs, reports, saved models, notes — lives here.

## The aim

Take a converged MLP from `../trainer/checkpoint/`, cut its hidden layers, and
land within **±2.5 points** of the dense model's test accuracy using
**no fine-tuning at all** — zero backward passes, zero `optimizer.step()`. Only
the architecture, the saved weights, and forward passes over unlabelled
calibration inputs are used.

## The golden constraint

Every compressed layer width comes from that layer's GRIDE intrinsic dimension
and from nothing else:

```
M_l = round(multiplier * d_l)
```

`d_l` is measured at runtime by `intrinsic_rank.id_estimator.gride` on the
actual activations of the loaded checkpoint. `multiplier` is swept over a grid
derived at runtime from the discovered widths. No hardcoded widths, no
PCA-99% rank, no arbitrary sparsity ratios. This is the paper's thesis —
Zador's bound `E_m ~ m^{-1/d}` makes the achievable width a function of the
intrinsic dimension — so breaking it would invalidate the claim.

`widths_from_gride()` is the single place a width can be produced, and
`compress_stack()` asserts that every strategy returns exactly `target_m`
neurons.

## Layout

```
compression_mlp/
├── run_autonomous_compression.py   the driver + the 10 seed strategies
├── novel_strategies/               hot-loaded in the adaptive phase
│   ├── innovations.py              round 1 — selection + corrected least squares
│   ├── innovations2.py             round 2 — augmented dictionary, forward-backward
│   └── innovations3.py             round 3 — output-Jacobian (Fisher) metric
├── verify_compressed.py            rebuild each saved model from disk and re-evaluate
├── summarize.py                    regenerate notes/02_results.md from the CSV
├── notes/
│   ├── 00_design_decisions.md      why the code is shaped this way, negatives, bugs
│   ├── 01_method.md                the method and its three effect sizes
│   ├── 02_results.md               all result tables (generated — do not hand-edit)
│   └── 03_references.md            papers the methods relate to; nothing is vendored
├── results/                        compression_results.csv, run logs, per-model reports,
│                                   *_compressed / *_frontier .safetensors + .arch.json
├── final_report.md                 ranked table + GRIDE foundation, regenerated per run
└── adaptive_research_request.txt   diagnostics written when phase 1 misses
```

## Headline result

The aim is met on **all 8 MLP checkpoints**, training-free, at **63–82%
parameter reduction** — see `notes/02_results.md` for the full tables and
`notes/01_method.md` for why. Over 96 (checkpoint × GRIDE-budget) cells the
corrected-least-squares family beats every merging / pruning / OT method in
**96/96**, mean **+16.1 points**; no seed strategy wins a single cell.

## Running it

```bash
cd <PROJECT_ROOT>

# default: discover every MLP checkpoint, take the largest, sweep, report
python3 compression_mlp/run_autonomous_compression.py

# a specific checkpoint, full sweep (no early stop) — what the paper tables use
python3 compression_mlp/run_autonomous_compression.py \
    --checkpoint cifar10_mlp3_quad_bn --exhaustive --calib 16384

# every MLP found
python3 compression_mlp/run_autonomous_compression.py --all-mlps
```

Useful flags: `--tol` (target window), `--id-space {neuron,activation}`,
`--calib`, `--multipliers` / `--strategies` (restrict the sweep), `--rounds`
(adaptive rounds), `--alloc-search` (phase 3), `--no-refine` (skip phase 4),
`--only STRATEGY@MULT` (re-run and save one configuration),
`--wait-for-operator` (pause each adaptive round so new strategy code can be
pasted into `novel_strategies/`).

## What the script does

1. **Discovers** every `.safetensors` in `../trainer/checkpoint/` whose name
   denotes an MLP, ordered by size. It identifies the dataset by trying each
   entry of `utils.config.DATASETS` and requiring a *strict* `load_state_dict` —
   a wrong dataset gives a shape mismatch, so this is a real identification test
   rather than a filename lookup.
2. **Establishes the target**: evaluates the dense model, sets the window.
3. **Measures GRIDE** per hidden layer, on the neuron cloud and (as a
   cross-check) the activation manifold, then derives the multiplier grid.
4. **Phase 1** sweeps the 10 seed strategies across the grid, logging every run
   to `results/compression_results.csv`.
5. **Phase 2** (if the aim is unmet) writes `adaptive_research_request.txt` with
   diagnostics — best-per-strategy, best-per-multiplier, per-layer least-squares
   residuals — then hot-loads and sweeps `novel_strategies/*.py`, up to 3 rounds.
6. **Phase 3** frees each layer's multiplier independently at a fixed parameter
   budget (still GRIDE-legal — every width stays `round(mult_l · d_l)`) and runs
   coordinate descent, to test whether the uniform Zador shape is the right
   allocation. It is.
7. **Phase 4** bisects between the last miss and the first hit to locate the
   *frontier* — the tightest GRIDE budget that still clears the window — because
   the geometric ladder steps 12 → 16 and is far too coarse to find it.
8. **Reports** to `final_report.md` (and a per-model copy under `results/`),
   saving two artefacts per checkpoint: `*_compressed` (most accurate) and
   `*_frontier` (smallest still inside the window), each `.safetensors` with a
   companion `.arch.json` recording widths, GRIDE dimensions and the strategy.

Re-check any saved model with `python3 compression_mlp/verify_compressed.py`: it rebuilds
each one from its `.safetensors` + `.arch.json` alone, re-evaluates it *and* its
dense source, and reports whether the recorded numbers reproduce. Currently
16/16.

## The 10 seed strategies

| # | name | idea |
|---|---|---|
| 1 | `01_direct_ls_svd` | rank-*M* leverage selection + least-squares consuming layer |
| 2 | `02_topm_l2_norm` | magnitude pruning baseline |
| 3 | `03_activation_pattern_kmeans` | k-means on binarised ReLU masks, merge |
| 4 | `04_cosine_ot_intrafusion` | Intra-Fusion: keep top-`|w|_1`, EMD-transport the rest |
| 5 | `05_balanced_signature` | k-means on `[incoming | bias | outgoing]`, merge |
| 6 | `06_greedy_wasserstein1` | exact 1-D W1 between activation laws + average linkage |
| 7 | `07_mixed_importance` | protect the top-20% variance neurons, merge the rest, then LS |
| 8 | `08_ot_then_ls` | OT merge, then re-solve the consuming layer by LS |
| 9 | `09_kernel_kmeans_rbf` | RBF kernel k-means on signatures |
| 10 | `10_iterative_refinement` | two-stage geometric schedule with LS at each stage |

## The novel strategies

`N08_augdict_fwdbwd` (round 2) is the best method overall, winning 31/96 cells;
round 1's `N02_omp_ls` is the clearest statement of the core idea. Round 3's
Fisher metric is a genuine negative result. Note that an earlier version of these
notes recorded rounds 2 and 3 as negative *in general* — that conclusion came
from a single CIFAR-10 checkpoint and was wrong; see the correction in
`notes/00_design_decisions.md`.

## Constraints the code actually enforces

- **No fine-tuning.** There is no `.backward()`, no optimizer, and no `.grad`
  anywhere in this folder. Every correction is a closed-form linear solve.
- **No labels.** Calibration uses inputs only (`for x, _ in loader`).
- **GRIDE or nothing.** If `intrinsic_rank.id_estimator` cannot be imported the
  driver raises at import time rather than falling back to another width rule.
- **Checkpoints are `.safetensors`**, in and out, with an `.arch.json` companion
  recording the widths, the GRIDE dimensions, and the strategy that produced
  them.
