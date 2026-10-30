# Autonomous zero-shot compression report -- MNIST/mlp2_bn

*generated 2026-07-25 14:26*

## Aim

Compress **mlp2_bn** (235,914 params, dense test accuracy **99.27%**) to within **+/-2.5** points -- i.e. into **[96.77%, 101.77%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **98.40%** via `N08_augdict_fwdbwd` at multiplier x6.0, widths [104, 52] (87,630 params, -62.9%), drop +0.87 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 8192 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 256 | 17.32 | 11.91 |
| h2 | 128 | 8.62 | 10.64 |

Multiplier grid swept (derived from the discovered widths): `[3.0, 6.0]`

## Ranked results (top 25 of 65 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **98.40%** | +0.87 | `N08_augdict_fwdbwd` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 2 | **98.36%** | +0.91 | `N09_fisher_omp_e2` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 3 | **98.36%** | +0.91 | `N09_fisher_omp` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 4 | **98.36%** | +0.91 | `N09_fisher_omp_e4` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 5 | **98.35%** | +0.92 | `N07_fwdbwd_omp` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 6 | **98.29%** | +0.98 | `N06_augdict_omp_wide` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 7 | **98.28%** | +0.99 | `N02_omp_ls` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 8 | **98.28%** | +0.99 | `N02_omp_ls_lowridge` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 9 | **98.28%** | +0.99 | `N03_gradual_omp` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 10 | **98.18%** | +1.09 | `N04_bias_calibrated_omp` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 11 | **98.18%** | +1.09 | `N06_augdict_omp` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 12 | **98.12%** | +1.15 | `N05_omp_transport` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 13 | **98.11%** | +1.16 | `N08_augdict_fwdbwd` | x6.0 | [104, 26] | 84,640 | 64.1% |
| 14 | **97.85%** | +1.42 | `07_mixed_importance` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 15 | **97.79%** | +1.48 | `N08_augdict_fwdbwd` | x4.5 | [78, 39] | 64,711 | 72.6% |
| 16 | **97.64%** | +1.63 | `N09_fisher_omp_e2` | x4.5 | [78, 39] | 64,711 | 72.6% |
| 17 | **97.64%** | +1.63 | `N09_fisher_omp` | x4.5 | [78, 39] | 64,711 | 72.6% |
| 18 | **97.53%** | +1.74 | `10_iterative_refinement` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 19 | **97.53%** | +1.74 | `N01_span_ls` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 20 | **97.53%** | +1.74 | `N10_fisher_span_ls` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 21 | **97.20%** | +2.07 | `N08_augdict_fwdbwd` | x3.75 | [65, 32] | 53,467 | 77.3% |
| 22 | **97.07%** | +2.20 | `N09_fisher_omp_e2` | x3.75 | [65, 32] | 53,467 | 77.3% |
| 23 | **97.07%** | +2.20 | `N09_fisher_omp` | x3.75 | [65, 32] | 53,467 | 77.3% |
| 24 | **96.88%** | +2.39 | `08_ot_then_ls` | x6.0 | [104, 52] | 87,630 | 62.9% |
| 25 | **96.82%** | +2.45 | `N08_augdict_fwdbwd` | x6.0 | [52, 52] | 44,106 | 81.3% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N08_augdict_fwdbwd` | 98.40% | x6.0 | [104, 52] | novel (phase 2) |
| `N09_fisher_omp_e2` | 98.36% | x6.0 | [104, 52] | novel (phase 2) |
| `N09_fisher_omp` | 98.36% | x6.0 | [104, 52] | novel (phase 2) |
| `N09_fisher_omp_e4` | 98.36% | x6.0 | [104, 52] | novel (phase 2) |
| `N07_fwdbwd_omp` | 98.35% | x6.0 | [104, 52] | novel (phase 2) |
| `N06_augdict_omp_wide` | 98.29% | x6.0 | [104, 52] | novel (phase 2) |
| `N02_omp_ls` | 98.28% | x6.0 | [104, 52] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 98.28% | x6.0 | [104, 52] | novel (phase 2) |
| `N03_gradual_omp` | 98.28% | x6.0 | [104, 52] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 98.18% | x6.0 | [104, 52] | novel (phase 2) |
| `N06_augdict_omp` | 98.18% | x6.0 | [104, 52] | novel (phase 2) |
| `N05_omp_transport` | 98.12% | x6.0 | [104, 52] | novel (phase 2) |
| `07_mixed_importance` | 97.85% | x6.0 | [104, 52] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `10_iterative_refinement` | 97.53% | x6.0 | [104, 52] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 97.53% | x6.0 | [104, 52] | novel (phase 2) |
| `N10_fisher_span_ls` | 97.53% | x6.0 | [104, 52] | novel (phase 2) |
| `08_ot_then_ls` | 96.88% | x6.0 | [104, 52] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `01_direct_ls_svd` | 96.76% | x6.0 | [104, 52] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `03_activation_pattern_kmeans` | 94.91% | x6.0 | [104, 52] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `05_balanced_signature` | 81.52% | x6.0 | [104, 52] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 81.52% | x6.0 | [104, 52] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `02_topm_l2_norm` | 60.64% | x6.0 | [104, 52] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |
| `06_greedy_wasserstein1` | 21.27% | x6.0 | [104, 52] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `04_cosine_ot_intrafusion` | 17.28% | x6.0 | [104, 52] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x3.0 | [52, 26] | 96.72% | +2.55 |
| x3.19 | [55, 27] | 96.36% | +2.91 |
| x3.29 | [57, 28] | 96.64% | +2.63 |
| x3.38 | [59, 29] | 96.78% | +2.49 |
| x3.75 | [65, 32] | 97.20% | +2.07 |
| x4.5 | [78, 39] | 97.79% | +1.48 |
| x6.0 | [104, 52] | 98.40% | +0.87 |

## Why the winner works

The winning method is **`N08_augdict_fwdbwd`**.

It is a merging method: neurons are grouped and replaced by cluster barycentres, with outgoing weights summed. This preserves the layer output exactly when the cluster's members co-activate, which is the exactness condition for ReLU merging.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint mnist_mlp2_bn --id-space neuron --calib 8192
```

Raw log: `Explore/results/compression_results.csv` (65 runs). Best model: `Explore/results/mnist_mlp2_bn_compressed.safetensors`.
