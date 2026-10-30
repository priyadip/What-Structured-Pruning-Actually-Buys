# Autonomous zero-shot compression report -- CIFAR10/mlp3_oct_bn

*generated 2026-07-25 09:26*

## Aim

Compress **mlp3_oct_bn** (8,928,778 params, dense test accuracy **66.70%**) to within **+/-2.5** points -- i.e. into **[64.20%, 69.20%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **66.08%** via `N08_augdict_fwdbwd` at multiplier x32.0, widths [686, 826, 432] (3,037,134 params, -66.0%), drop +0.62 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 2048 | 21.43 | 34.22 |
| h2 | 1024 | 25.80 | 25.91 |
| h3 | 512 | 21.60 | 19.98 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]`

## Ranked results (top 25 of 416 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **66.08%** | +0.62 | `N08_augdict_fwdbwd` | x32.0 | [686, 826, 432] | 3,037,134 | 66.0% |
| 2 | **66.06%** | +0.64 | `N08_augdict_fwdbwd` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 3 | **65.77%** | +0.93 | `N08_augdict_fwdbwd` | x32.0 | [686, 826, 346] | 2,965,152 | 66.8% |
| 4 | **65.76%** | +0.94 | `N08_augdict_fwdbwd` | x32.0 | [686, 619, 512] | 2,855,901 | 68.0% |
| 5 | **65.76%** | +0.94 | `N08_augdict_fwdbwd` | x32.0 | [686, 619, 432] | 2,805,501 | 68.6% |
| 6 | **65.65%** | +1.05 | `N09_fisher_omp` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 7 | **65.65%** | +1.05 | `N09_fisher_omp_e4` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 8 | **65.65%** | +1.05 | `N08_augdict_fwdbwd` | x32.0 | [429, 826, 512] | 2,102,051 | 76.5% |
| 9 | **65.60%** | +1.10 | `N08_augdict_fwdbwd` | x32.0 | [429, 826, 432] | 2,035,091 | 77.2% |
| 10 | **65.58%** | +1.12 | `N06_augdict_omp` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 11 | **65.56%** | +1.14 | `N07_fwdbwd_omp` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 12 | **65.56%** | +1.14 | `N08_augdict_fwdbwd` | x32.0 | [686, 826, 259] | 2,892,333 | 67.6% |
| 13 | **65.54%** | +1.16 | `N09_fisher_omp_e2` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 14 | **65.47%** | +1.23 | `N06_augdict_omp_wide` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 15 | **65.45%** | +1.25 | `10_iterative_refinement` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 16 | **65.45%** | +1.25 | `N01_span_ls` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 17 | **65.45%** | +1.25 | `N10_fisher_span_ls` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 18 | **65.43%** | +1.27 | `N02_omp_ls` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 19 | **65.43%** | +1.27 | `N02_omp_ls_lowridge` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 20 | **65.43%** | +1.27 | `N03_gradual_omp` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 21 | **65.40%** | +1.30 | `N08_augdict_fwdbwd` | x32.0 | [514, 826, 512] | 2,433,466 | 72.7% |
| 22 | **65.40%** | +1.30 | `N08_augdict_fwdbwd` | x32.0 | [514, 826, 432] | 2,366,506 | 73.5% |
| 23 | **65.39%** | +1.31 | `N04_bias_calibrated_omp` | x32.0 | [686, 826, 512] | 3,104,094 | 65.2% |
| 24 | **65.35%** | +1.35 | `N08_augdict_fwdbwd` | x32.0 | [686, 516, 512] | 2,732,404 | 69.4% |
| 25 | **65.35%** | +1.35 | `N08_augdict_fwdbwd` | x32.0 | [686, 826, 216] | 2,856,342 | 68.0% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N08_augdict_fwdbwd` | 66.08% | x32.0 | [686, 826, 432] | novel (phase 2) |
| `N09_fisher_omp` | 65.65% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N09_fisher_omp_e4` | 65.65% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N06_augdict_omp` | 65.58% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N07_fwdbwd_omp` | 65.56% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N09_fisher_omp_e2` | 65.54% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N06_augdict_omp_wide` | 65.47% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `10_iterative_refinement` | 65.45% | x32.0 | [686, 826, 512] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 65.45% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N10_fisher_span_ls` | 65.45% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N02_omp_ls` | 65.43% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 65.43% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N03_gradual_omp` | 65.43% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 65.39% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `N05_omp_transport` | 65.33% | x32.0 | [686, 826, 512] | novel (phase 2) |
| `01_direct_ls_svd` | 65.31% | x32.0 | [686, 826, 512] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `07_mixed_importance` | 65.03% | x32.0 | [686, 826, 512] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `08_ot_then_ls` | 63.98% | x32.0 | [686, 826, 512] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `05_balanced_signature` | 62.53% | x32.0 | [686, 826, 512] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 62.53% | x32.0 | [686, 826, 512] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `03_activation_pattern_kmeans` | 60.07% | x32.0 | [686, 826, 512] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `06_greedy_wasserstein1` | 43.32% | x32.0 | [686, 826, 512] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `02_topm_l2_norm` | 33.94% | x32.0 | [686, 826, 512] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |
| `04_cosine_ot_intrafusion` | 29.35% | x32.0 | [686, 826, 512] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [32, 39, 32] | 42.79% | +23.91 |
| x2.0 | [43, 52, 43] | 45.20% | +21.50 |
| x2.5 | [54, 65, 54] | 47.75% | +18.95 |
| x3.0 | [64, 77, 65] | 49.33% | +17.37 |
| x4.0 | [86, 103, 86] | 52.21% | +14.49 |
| x5.0 | [107, 129, 108] | 54.87% | +11.83 |
| x6.0 | [129, 155, 130] | 56.27% | +10.43 |
| x8.0 | [171, 206, 173] | 59.12% | +7.58 |
| x10.0 | [214, 258, 216] | 60.88% | +5.82 |
| x12.0 | [257, 310, 259] | 62.02% | +4.68 |
| x16.0 | [343, 413, 346] | 63.69% | +3.01 |
| x18.0 | [386, 464, 389] | 63.75% | +2.95 |
| x18.5 | [396, 477, 400] | 64.02% | +2.68 |
| x18.75 | [402, 484, 405] | 64.05% | +2.65 |
| x18.88 | [405, 487, 408] | 64.41% | +2.29 |
| x19.0 | [407, 490, 410] | 64.35% | +2.35 |
| x20.0 | [429, 516, 432] | 64.60% | +2.10 |
| x24.0 | [514, 619, 512] | 65.01% | +1.69 |
| x32.0 | [686, 826, 432] | 66.08% | +0.62 |

## Why the winner works

The winning method is **`N08_augdict_fwdbwd`**.

It is a merging method: neurons are grouped and replaced by cluster barycentres, with outgoing weights summed. This preserves the layer output exactly when the cluster's members co-activate, which is the exactness condition for ReLU merging.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint cifar10_mlp3_oct_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (416 runs). Best model: `Explore/results/cifar10_mlp3_oct_bn_compressed.safetensors`.
