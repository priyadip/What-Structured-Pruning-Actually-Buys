# Autonomous zero-shot compression report -- FashionMNIST/mlp3_bn

*generated 2026-07-25 09:09*

## Aim

Compress **mlp3_bn** (243,658 params, dense test accuracy **90.12%**) to within **+/-2.5** points -- i.e. into **[87.62%, 92.62%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **90.05%** via `N05_omp_transport` at multiplier x12.0, widths [193, 128, 64] (185,243 params, -24.0%), drop +0.07 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 256 | 16.07 | 11.56 |
| h2 | 128 | 16.36 | 10.45 |
| h3 | 64 | 11.47 | 10.02 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]`

## Ranked results (top 25 of 282 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **90.05%** | +0.07 | `N05_omp_transport` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 2 | **90.04%** | +0.08 | `N04_bias_calibrated_omp` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 3 | **90.02%** | +0.10 | `N02_omp_ls` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 4 | **90.02%** | +0.10 | `N02_omp_ls_lowridge` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 5 | **90.02%** | +0.10 | `N03_gradual_omp` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 6 | **89.91%** | +0.21 | `07_mixed_importance` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 7 | **89.88%** | +0.24 | `N07_fwdbwd_omp` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 8 | **89.88%** | +0.24 | `N08_augdict_fwdbwd` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 9 | **89.85%** | +0.27 | `01_direct_ls_svd` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 10 | **89.85%** | +0.27 | `N06_augdict_omp` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 11 | **89.84%** | +0.28 | `N05_omp_transport` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 12 | **89.83%** | +0.29 | `N05_omp_transport` | x12.0 | [193, 128, 57] | 184,270 | 24.4% |
| 13 | **89.82%** | +0.30 | `10_iterative_refinement` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 14 | **89.82%** | +0.30 | `N01_span_ls` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 15 | **89.82%** | +0.30 | `N10_fisher_span_ls` | x12.0 | [193, 128, 64] | 185,243 | 24.0% |
| 16 | **89.78%** | +0.34 | `N09_fisher_omp_e2` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 17 | **89.78%** | +0.34 | `N09_fisher_omp` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 18 | **89.78%** | +0.34 | `N09_fisher_omp_e4` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 19 | **89.76%** | +0.36 | `N02_omp_ls` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 20 | **89.76%** | +0.36 | `N02_omp_ls_lowridge` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 21 | **89.76%** | +0.36 | `N03_gradual_omp` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 22 | **89.76%** | +0.36 | `N04_bias_calibrated_omp` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 23 | **89.76%** | +0.36 | `N05_omp_transport` | x12.0 | [193, 98, 64] | 177,503 | 27.2% |
| 24 | **89.73%** | +0.39 | `N07_fwdbwd_omp` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |
| 25 | **89.73%** | +0.39 | `N08_augdict_fwdbwd` | x10.0 | [161, 128, 64] | 156,027 | 36.0% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N05_omp_transport` | 90.05% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 90.04% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N02_omp_ls` | 90.02% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 90.02% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N03_gradual_omp` | 90.02% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `07_mixed_importance` | 89.91% | x12.0 | [193, 128, 64] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `N07_fwdbwd_omp` | 89.88% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 89.88% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `01_direct_ls_svd` | 89.85% | x12.0 | [193, 128, 64] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `N06_augdict_omp` | 89.85% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `10_iterative_refinement` | 89.82% | x12.0 | [193, 128, 64] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 89.82% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N10_fisher_span_ls` | 89.82% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `N09_fisher_omp_e2` | 89.78% | x10.0 | [161, 128, 64] | novel (phase 2) |
| `N09_fisher_omp` | 89.78% | x10.0 | [161, 128, 64] | novel (phase 2) |
| `N09_fisher_omp_e4` | 89.78% | x10.0 | [161, 128, 64] | novel (phase 2) |
| `08_ot_then_ls` | 89.72% | x12.0 | [193, 128, 64] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `N06_augdict_omp_wide` | 89.68% | x12.0 | [193, 128, 64] | novel (phase 2) |
| `03_activation_pattern_kmeans` | 88.69% | x12.0 | [193, 128, 64] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `05_balanced_signature` | 88.37% | x12.0 | [193, 128, 64] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 88.37% | x12.0 | [193, 128, 64] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 86.44% | x12.0 | [193, 128, 64] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `02_topm_l2_norm` | 78.45% | x12.0 | [193, 128, 64] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |
| `04_cosine_ot_intrafusion` | 71.19% | x12.0 | [193, 128, 64] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [24, 25, 17] | 76.64% | +13.48 |
| x2.0 | [32, 33, 23] | 79.84% | +10.28 |
| x2.5 | [40, 41, 29] | 81.17% | +8.95 |
| x3.0 | [48, 49, 34] | 83.72% | +6.40 |
| x4.0 | [64, 65, 46] | 85.67% | +4.45 |
| x5.0 | [80, 82, 57] | 87.26% | +2.86 |
| x5.25 | [84, 86, 60] | 87.61% | +2.51 |
| x5.31 | [85, 87, 61] | 87.53% | +2.59 |
| x5.34 | [86, 87, 61] | 87.84% | +2.28 |
| x5.38 | [86, 88, 62] | 87.91% | +2.21 |
| x5.5 | [88, 90, 63] | 87.90% | +2.22 |
| x6.0 | [96, 98, 64] | 88.31% | +1.81 |
| x8.0 | [129, 128, 64] | 89.44% | +0.68 |
| x10.0 | [161, 128, 64] | 89.84% | +0.28 |
| x12.0 | [193, 128, 64] | 90.05% | +0.07 |

## Why the winner works

The winning method is **`N05_omp_transport`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint fashionmnist_mlp3_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (282 runs). Best model: `Explore/results/fashionmnist_mlp3_bn_compressed.safetensors`.
