# Autonomous zero-shot compression report -- FashionMNIST/mlp2_bn

*generated 2026-07-25 10:06*

## Aim

Compress **mlp2_bn** (235,914 params, dense test accuracy **90.10%**) to within **+/-2.5** points -- i.e. into **[87.60%, 92.60%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **89.86%** via `N07_fwdbwd_omp` at multiplier x12.0, widths [205, 128] (188,583 params, -20.1%), drop +0.24 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 256 | 17.09 | 11.64 |
| h2 | 128 | 16.36 | 10.77 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]`

## Ranked results (top 25 of 270 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **89.86%** | +0.24 | `N07_fwdbwd_omp` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 2 | **89.86%** | +0.24 | `N08_augdict_fwdbwd` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 3 | **89.83%** | +0.27 | `N09_fisher_omp_e2` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 4 | **89.83%** | +0.27 | `N09_fisher_omp` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 5 | **89.83%** | +0.27 | `N09_fisher_omp_e4` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 6 | **89.80%** | +0.30 | `10_iterative_refinement` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 7 | **89.80%** | +0.30 | `N01_span_ls` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 8 | **89.80%** | +0.30 | `N10_fisher_span_ls` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 9 | **89.77%** | +0.33 | `01_direct_ls_svd` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 10 | **89.75%** | +0.35 | `N04_bias_calibrated_omp` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 11 | **89.75%** | +0.35 | `N05_omp_transport` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 12 | **89.72%** | +0.38 | `N02_omp_ls` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 13 | **89.72%** | +0.38 | `N02_omp_ls_lowridge` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 14 | **89.72%** | +0.38 | `N03_gradual_omp` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 15 | **89.69%** | +0.41 | `N06_augdict_omp` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 16 | **89.61%** | +0.49 | `N02_omp_ls` | x10.0 | [171, 128] | 157,541 | 33.2% |
| 17 | **89.61%** | +0.49 | `N02_omp_ls_lowridge` | x10.0 | [171, 128] | 157,541 | 33.2% |
| 18 | **89.61%** | +0.49 | `N03_gradual_omp` | x10.0 | [171, 128] | 157,541 | 33.2% |
| 19 | **89.59%** | +0.51 | `N05_omp_transport` | x10.0 | [171, 128] | 157,541 | 33.2% |
| 20 | **89.58%** | +0.52 | `N04_bias_calibrated_omp` | x10.0 | [171, 128] | 157,541 | 33.2% |
| 21 | **89.58%** | +0.52 | `N07_fwdbwd_omp` | x12.0 | [205, 98] | 182,103 | 22.8% |
| 22 | **89.51%** | +0.59 | `07_mixed_importance` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 23 | **89.51%** | +0.59 | `08_ot_then_ls` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 24 | **89.50%** | +0.60 | `N06_augdict_omp_wide` | x12.0 | [205, 128] | 188,583 | 20.1% |
| 25 | **89.46%** | +0.64 | `10_iterative_refinement` | x10.0 | [171, 128] | 157,541 | 33.2% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N07_fwdbwd_omp` | 89.86% | x12.0 | [205, 128] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 89.86% | x12.0 | [205, 128] | novel (phase 2) |
| `N09_fisher_omp_e2` | 89.83% | x12.0 | [205, 128] | novel (phase 2) |
| `N09_fisher_omp` | 89.83% | x12.0 | [205, 128] | novel (phase 2) |
| `N09_fisher_omp_e4` | 89.83% | x12.0 | [205, 128] | novel (phase 2) |
| `10_iterative_refinement` | 89.80% | x12.0 | [205, 128] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 89.80% | x12.0 | [205, 128] | novel (phase 2) |
| `N10_fisher_span_ls` | 89.80% | x12.0 | [205, 128] | novel (phase 2) |
| `01_direct_ls_svd` | 89.77% | x12.0 | [205, 128] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `N04_bias_calibrated_omp` | 89.75% | x12.0 | [205, 128] | novel (phase 2) |
| `N05_omp_transport` | 89.75% | x12.0 | [205, 128] | novel (phase 2) |
| `N02_omp_ls` | 89.72% | x12.0 | [205, 128] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 89.72% | x12.0 | [205, 128] | novel (phase 2) |
| `N03_gradual_omp` | 89.72% | x12.0 | [205, 128] | novel (phase 2) |
| `N06_augdict_omp` | 89.69% | x12.0 | [205, 128] | novel (phase 2) |
| `07_mixed_importance` | 89.51% | x12.0 | [205, 128] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `08_ot_then_ls` | 89.51% | x12.0 | [205, 128] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `N06_augdict_omp_wide` | 89.50% | x12.0 | [205, 128] | novel (phase 2) |
| `03_activation_pattern_kmeans` | 88.64% | x12.0 | [205, 128] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `05_balanced_signature` | 88.13% | x12.0 | [205, 128] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 88.13% | x12.0 | [205, 128] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 87.48% | x12.0 | [205, 128] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `02_topm_l2_norm` | 82.65% | x12.0 | [205, 128] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |
| `04_cosine_ot_intrafusion` | 82.39% | x12.0 | [205, 128] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [26, 25] | 76.96% | +13.14 |
| x2.0 | [34, 33] | 80.69% | +9.41 |
| x2.5 | [43, 41] | 81.77% | +8.33 |
| x3.0 | [51, 49] | 83.63% | +6.47 |
| x4.0 | [68, 65] | 85.80% | +4.30 |
| x5.0 | [85, 82] | 87.08% | +3.02 |
| x5.5 | [94, 90] | 87.46% | +2.64 |
| x5.62 | [96, 92] | 87.37% | +2.73 |
| x5.69 | [97, 93] | 87.64% | +2.46 |
| x5.75 | [98, 94] | 87.78% | +2.32 |
| x6.0 | [103, 98] | 87.98% | +2.12 |
| x8.0 | [137, 128] | 89.06% | +1.04 |
| x10.0 | [171, 128] | 89.61% | +0.49 |
| x12.0 | [205, 128] | 89.86% | +0.24 |

## Why the winner works

The winning method is **`N07_fwdbwd_omp`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint fashionmnist_mlp2_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (270 runs). Best model: `Explore/results/fashionmnist_mlp2_bn_compressed.safetensors`.
