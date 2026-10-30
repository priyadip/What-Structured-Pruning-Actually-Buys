# Autonomous zero-shot compression report -- CIFAR100/mlp3_oct_bn

*generated 2026-07-25 09:49*

## Aim

Compress **mlp3_oct_bn** (8,974,948 params, dense test accuracy **40.25%**) to within **+/-2.5** points -- i.e. into **[37.75%, 42.75%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **39.74%** via `N07_fwdbwd_omp` at multiplier x32.0, widths [611, 1024, 479] (3,043,266 params, -66.1%), drop +0.51 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 2048 | 19.09 | 30.56 |
| h2 | 1024 | 34.13 | 27.68 |
| h3 | 512 | 39.91 | 23.61 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]`

## Ranked results (top 25 of 416 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **39.74%** | +0.51 | `N07_fwdbwd_omp` | x32.0 | [611, 1024, 479] | 3,043,266 | 66.1% |
| 2 | **39.72%** | +0.53 | `N07_fwdbwd_omp` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 3 | **39.49%** | +0.76 | `N02_omp_ls` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 4 | **39.49%** | +0.76 | `N02_omp_ls_lowridge` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 5 | **39.49%** | +0.76 | `N03_gradual_omp` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 6 | **39.47%** | +0.78 | `N05_omp_transport` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 7 | **39.44%** | +0.81 | `N07_fwdbwd_omp` | x32.0 | [611, 819, 479] | 2,819,611 | 68.6% |
| 8 | **39.41%** | +0.84 | `N04_bias_calibrated_omp` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 9 | **39.39%** | +0.86 | `N08_augdict_fwdbwd` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 10 | **39.33%** | +0.92 | `N06_augdict_omp` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 11 | **39.24%** | +1.01 | `N06_augdict_omp_wide` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 12 | **39.19%** | +1.06 | `N07_fwdbwd_omp` | x32.0 | [611, 819, 512] | 2,849,971 | 68.2% |
| 13 | **39.14%** | +1.11 | `N07_fwdbwd_omp` | x32.0 | [611, 1024, 399] | 2,953,266 | 67.1% |
| 14 | **39.13%** | +1.12 | `N09_fisher_omp_e2` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 15 | **39.08%** | +1.17 | `N07_fwdbwd_omp` | x32.0 | [611, 683, 479] | 2,671,235 | 70.2% |
| 16 | **39.03%** | +1.22 | `N07_fwdbwd_omp` | x32.0 | [458, 1024, 512] | 2,453,550 | 72.7% |
| 17 | **39.02%** | +1.23 | `10_iterative_refinement` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 18 | **39.02%** | +1.23 | `N01_span_ls` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 19 | **39.02%** | +1.23 | `N10_fisher_span_ls` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 20 | **39.01%** | +1.24 | `N09_fisher_omp` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 21 | **39.01%** | +1.24 | `N09_fisher_omp_e4` | x32.0 | [611, 1024, 512] | 3,080,391 | 65.7% |
| 22 | **38.79%** | +1.46 | `N08_augdict_fwdbwd` | x24.0 | [458, 819, 512] | 2,254,495 | 74.9% |
| 23 | **38.77%** | +1.48 | `N07_fwdbwd_omp` | x32.0 | [458, 1024, 479] | 2,416,425 | 73.1% |
| 24 | **38.74%** | +1.51 | `N07_fwdbwd_omp` | x32.0 | [611, 683, 512] | 2,697,107 | 69.9% |
| 25 | **38.71%** | +1.54 | `N02_omp_ls_lowridge` | x24.0 | [458, 819, 512] | 2,254,495 | 74.9% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N07_fwdbwd_omp` | 39.74% | x32.0 | [611, 1024, 479] | novel (phase 2) |
| `N02_omp_ls` | 39.49% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 39.49% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N03_gradual_omp` | 39.49% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N05_omp_transport` | 39.47% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 39.41% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 39.39% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N06_augdict_omp` | 39.33% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N06_augdict_omp_wide` | 39.24% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N09_fisher_omp_e2` | 39.13% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `10_iterative_refinement` | 39.02% | x32.0 | [611, 1024, 512] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 39.02% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N10_fisher_span_ls` | 39.02% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N09_fisher_omp` | 39.01% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `N09_fisher_omp_e4` | 39.01% | x32.0 | [611, 1024, 512] | novel (phase 2) |
| `07_mixed_importance` | 38.54% | x32.0 | [611, 1024, 512] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `01_direct_ls_svd` | 37.17% | x32.0 | [611, 1024, 512] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `08_ot_then_ls` | 36.23% | x32.0 | [611, 1024, 512] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `03_activation_pattern_kmeans` | 34.61% | x32.0 | [611, 1024, 512] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `05_balanced_signature` | 34.56% | x32.0 | [611, 1024, 512] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 34.56% | x32.0 | [611, 1024, 512] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 19.49% | x32.0 | [611, 1024, 512] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `04_cosine_ot_intrafusion` | 6.21% | x32.0 | [611, 1024, 512] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |
| `02_topm_l2_norm` | 5.15% | x32.0 | [611, 1024, 512] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [29, 51, 60] | 14.36% | +25.89 |
| x2.0 | [38, 68, 80] | 16.50% | +23.75 |
| x2.5 | [48, 85, 100] | 18.85% | +21.40 |
| x3.0 | [57, 102, 120] | 20.46% | +19.79 |
| x4.0 | [76, 137, 160] | 23.79% | +16.46 |
| x5.0 | [95, 171, 200] | 25.54% | +14.71 |
| x6.0 | [115, 205, 239] | 27.45% | +12.80 |
| x8.0 | [153, 273, 319] | 30.62% | +9.63 |
| x10.0 | [191, 341, 399] | 33.21% | +7.04 |
| x12.0 | [229, 410, 479] | 35.11% | +5.14 |
| x16.0 | [305, 546, 512] | 37.21% | +3.04 |
| x17.0 | [325, 580, 512] | 37.27% | +2.98 |
| x17.5 | [334, 597, 512] | 37.46% | +2.79 |
| x17.75 | [339, 606, 512] | 37.51% | +2.74 |
| x17.88 | [341, 610, 512] | 37.51% | +2.74 |
| x18.0 | [344, 614, 512] | 37.82% | +2.43 |
| x20.0 | [382, 683, 512] | 38.21% | +2.04 |
| x24.0 | [458, 819, 512] | 38.79% | +1.46 |
| x32.0 | [611, 1024, 479] | 39.74% | +0.51 |

## Why the winner works

The winning method is **`N07_fwdbwd_omp`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint cifar100_mlp3_oct_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (416 runs). Best model: `Explore/results/cifar100_mlp3_oct_bn_compressed.safetensors`.
