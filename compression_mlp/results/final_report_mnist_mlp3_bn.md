# Autonomous zero-shot compression report -- MNIST/mlp3_bn

*generated 2026-07-25 09:06*

## Aim

Compress **mlp3_bn** (243,658 params, dense test accuracy **99.27%**) to within **+/-2.5** points -- i.e. into **[96.77%, 101.77%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **99.23%** via `N07_fwdbwd_omp` at multiplier x12.0, widths [208, 106, 58] (192,230 params, -21.1%), drop +0.04 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 256 | 17.31 | 11.96 |
| h2 | 128 | 8.83 | 10.65 |
| h3 | 64 | 7.25 | 8.11 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]`

## Ranked results (top 25 of 300 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **99.23%** | +0.04 | `N07_fwdbwd_omp` | x12.0 | [208, 106, 58] | 192,230 | 21.1% |
| 2 | **99.22%** | +0.05 | `N07_fwdbwd_omp` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 3 | **99.22%** | +0.05 | `N08_augdict_fwdbwd` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 4 | **99.22%** | +0.05 | `N07_fwdbwd_omp` | x12.0 | [208, 106, 44] | 190,592 | 21.8% |
| 5 | **99.20%** | +0.07 | `N07_fwdbwd_omp` | x12.0 | [208, 106, 36] | 189,656 | 22.2% |
| 6 | **99.20%** | +0.07 | `N07_fwdbwd_omp` | x12.0 | [208, 88, 58] | 187,424 | 23.1% |
| 7 | **99.19%** | +0.08 | `N07_fwdbwd_omp` | x12.0 | [208, 106, 29] | 188,837 | 22.5% |
| 8 | **99.18%** | +0.09 | `10_iterative_refinement` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 9 | **99.18%** | +0.09 | `N01_span_ls` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 10 | **99.18%** | +0.09 | `N10_fisher_span_ls` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 11 | **99.18%** | +0.09 | `N07_fwdbwd_omp` | x12.0 | [208, 71, 64] | 183,377 | 24.7% |
| 12 | **99.18%** | +0.09 | `N07_fwdbwd_omp` | x12.0 | [208, 88, 64] | 188,018 | 22.8% |
| 13 | **99.17%** | +0.10 | `N02_omp_ls_lowridge` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 14 | **99.16%** | +0.11 | `N02_omp_ls` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 15 | **99.16%** | +0.11 | `N03_gradual_omp` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 16 | **99.16%** | +0.11 | `N09_fisher_omp_e2` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 17 | **99.16%** | +0.11 | `N09_fisher_omp` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 18 | **99.16%** | +0.11 | `N09_fisher_omp_e4` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 19 | **99.15%** | +0.12 | `N06_augdict_omp` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 20 | **99.13%** | +0.14 | `01_direct_ls_svd` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 21 | **99.12%** | +0.15 | `N07_fwdbwd_omp` | x12.0 | [208, 71, 58] | 182,885 | 24.9% |
| 22 | **99.11%** | +0.16 | `N04_bias_calibrated_omp` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 23 | **99.11%** | +0.16 | `N05_omp_transport` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 24 | **99.11%** | +0.16 | `N06_augdict_omp_wide` | x12.0 | [208, 106, 64] | 192,932 | 20.8% |
| 25 | **99.10%** | +0.17 | `N02_omp_ls` | x10.0 | [173, 88, 64] | 157,463 | 35.4% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N07_fwdbwd_omp` | 99.23% | x12.0 | [208, 106, 58] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 99.22% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `10_iterative_refinement` | 99.18% | x12.0 | [208, 106, 64] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 99.18% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N10_fisher_span_ls` | 99.18% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 99.17% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N02_omp_ls` | 99.16% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N03_gradual_omp` | 99.16% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N09_fisher_omp_e2` | 99.16% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N09_fisher_omp` | 99.16% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N09_fisher_omp_e4` | 99.16% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N06_augdict_omp` | 99.15% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `01_direct_ls_svd` | 99.13% | x12.0 | [208, 106, 64] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `N04_bias_calibrated_omp` | 99.11% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N05_omp_transport` | 99.11% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `N06_augdict_omp_wide` | 99.11% | x12.0 | [208, 106, 64] | novel (phase 2) |
| `07_mixed_importance` | 99.05% | x12.0 | [208, 106, 64] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `03_activation_pattern_kmeans` | 99.03% | x12.0 | [208, 106, 64] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `08_ot_then_ls` | 98.97% | x12.0 | [208, 106, 64] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `05_balanced_signature` | 98.95% | x12.0 | [208, 106, 64] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 98.95% | x12.0 | [208, 106, 64] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 98.37% | x12.0 | [208, 106, 64] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `02_topm_l2_norm` | 98.34% | x12.0 | [208, 106, 64] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |
| `04_cosine_ot_intrafusion` | 93.03% | x12.0 | [208, 106, 64] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [26, 13, 11] | 90.85% | +8.42 |
| x2.0 | [35, 18, 15] | 93.59% | +5.68 |
| x2.5 | [43, 22, 18] | 95.29% | +3.98 |
| x3.0 | [52, 26, 22] | 96.34% | +2.93 |
| x3.25 | [56, 29, 24] | 96.45% | +2.82 |
| x3.38 | [59, 30, 25] | 96.72% | +2.55 |
| x3.41 | [59, 30, 25] | 96.72% | +2.55 |
| x3.44 | [60, 30, 25] | 96.95% | +2.32 |
| x3.5 | [61, 31, 25] | 97.04% | +2.23 |
| x4.0 | [69, 35, 29] | 97.33% | +1.94 |
| x5.0 | [87, 44, 36] | 98.06% | +1.21 |
| x6.0 | [104, 53, 44] | 98.49% | +0.78 |
| x8.0 | [138, 71, 58] | 98.89% | +0.38 |
| x10.0 | [173, 88, 64] | 99.10% | +0.17 |
| x12.0 | [208, 106, 58] | 99.23% | +0.04 |

## Why the winner works

The winning method is **`N07_fwdbwd_omp`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint mnist_mlp3_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (300 runs). Best model: `Explore/results/mnist_mlp3_bn_compressed.safetensors`.
