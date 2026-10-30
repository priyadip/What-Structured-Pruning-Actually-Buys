# Autonomous zero-shot compression report -- CIFAR100/mlp3_quad_bn

*generated 2026-07-25 10:01*

## Aim

Compress **mlp3_quad_bn** (3,832,164 params, dense test accuracy **37.27%**) to within **+/-2.5** points -- i.e. into **[34.77%, 39.77%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **37.10%** via `N07_fwdbwd_omp` at multiplier x24.0, widths [449, 512, 256] (1,767,205 params, -53.9%), drop +0.17 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 1024 | 18.70 | 28.00 |
| h2 | 512 | 27.36 | 23.78 |
| h3 | 256 | 27.00 | 18.18 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]`

## Ranked results (top 25 of 389 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **37.10%** | +0.17 | `N07_fwdbwd_omp` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 2 | **37.05%** | +0.22 | `N06_augdict_omp` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 3 | **37.01%** | +0.26 | `N04_bias_calibrated_omp` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 4 | **37.00%** | +0.27 | `N06_augdict_omp_wide` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 5 | **37.00%** | +0.27 | `N07_fwdbwd_omp` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 6 | **37.00%** | +0.27 | `N08_augdict_fwdbwd` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 7 | **36.98%** | +0.29 | `N02_omp_ls_lowridge` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 8 | **36.97%** | +0.30 | `N02_omp_ls` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 9 | **36.97%** | +0.30 | `N03_gradual_omp` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 10 | **36.96%** | +0.31 | `N09_fisher_omp_e2` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 11 | **36.96%** | +0.31 | `N09_fisher_omp` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 12 | **36.96%** | +0.31 | `N09_fisher_omp_e4` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 13 | **36.95%** | +0.32 | `N05_omp_transport` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 14 | **36.91%** | +0.36 | `10_iterative_refinement` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 15 | **36.91%** | +0.36 | `N01_span_ls` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 16 | **36.91%** | +0.36 | `N10_fisher_span_ls` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 17 | **36.85%** | +0.42 | `N09_fisher_omp_e2` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 18 | **36.85%** | +0.42 | `N09_fisher_omp` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 19 | **36.85%** | +0.42 | `N09_fisher_omp_e4` | x32.0 | [598, 512, 256] | 2,301,370 | 39.9% |
| 20 | **36.76%** | +0.51 | `N08_augdict_fwdbwd` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 21 | **36.72%** | +0.55 | `N06_augdict_omp` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 22 | **36.70%** | +0.57 | `N05_omp_transport` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 23 | **36.69%** | +0.58 | `N07_fwdbwd_omp` | x24.0 | [449, 438, 256] | 1,714,961 | 55.2% |
| 24 | **36.66%** | +0.61 | `N04_bias_calibrated_omp` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |
| 25 | **36.65%** | +0.62 | `N02_omp_ls` | x24.0 | [449, 512, 256] | 1,767,205 | 53.9% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N07_fwdbwd_omp` | 37.10% | x24.0 | [449, 512, 256] | novel (phase 2) |
| `N06_augdict_omp` | 37.05% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 37.01% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N06_augdict_omp_wide` | 37.00% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 37.00% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 36.98% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N02_omp_ls` | 36.97% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N03_gradual_omp` | 36.97% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N09_fisher_omp_e2` | 36.96% | x24.0 | [449, 512, 256] | novel (phase 2) |
| `N09_fisher_omp` | 36.96% | x24.0 | [449, 512, 256] | novel (phase 2) |
| `N09_fisher_omp_e4` | 36.96% | x24.0 | [449, 512, 256] | novel (phase 2) |
| `N05_omp_transport` | 36.95% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `10_iterative_refinement` | 36.91% | x32.0 | [598, 512, 256] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 36.91% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `N10_fisher_span_ls` | 36.91% | x32.0 | [598, 512, 256] | novel (phase 2) |
| `01_direct_ls_svd` | 36.41% | x32.0 | [598, 512, 256] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `07_mixed_importance` | 36.29% | x32.0 | [598, 512, 256] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `03_activation_pattern_kmeans` | 35.89% | x32.0 | [598, 512, 256] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `08_ot_then_ls` | 35.80% | x32.0 | [598, 512, 256] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `05_balanced_signature` | 35.50% | x32.0 | [598, 512, 256] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 35.50% | x32.0 | [598, 512, 256] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 30.92% | x32.0 | [598, 512, 256] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `04_cosine_ot_intrafusion` | 22.37% | x32.0 | [598, 512, 256] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |
| `02_topm_l2_norm` | 19.46% | x32.0 | [598, 512, 256] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [28, 41, 40] | 13.15% | +24.12 |
| x2.0 | [37, 55, 54] | 15.22% | +22.05 |
| x2.5 | [47, 68, 67] | 18.01% | +19.26 |
| x3.0 | [56, 82, 81] | 19.83% | +17.44 |
| x4.0 | [75, 109, 108] | 23.10% | +14.17 |
| x5.0 | [93, 137, 135] | 25.49% | +11.78 |
| x6.0 | [112, 164, 162] | 27.77% | +9.50 |
| x8.0 | [150, 219, 216] | 31.52% | +5.75 |
| x10.0 | [187, 274, 256] | 32.94% | +4.33 |
| x12.0 | [224, 328, 256] | 34.41% | +2.86 |
| x12.5 | [234, 342, 256] | 34.56% | +2.71 |
| x12.75 | [238, 349, 256] | 34.74% | +2.53 |
| x12.88 | [241, 352, 256] | 34.79% | +2.48 |
| x13.0 | [243, 356, 256] | 34.89% | +2.38 |
| x14.0 | [262, 383, 256] | 35.15% | +2.12 |
| x16.0 | [299, 438, 256] | 35.85% | +1.42 |
| x20.0 | [374, 512, 256] | 36.63% | +0.64 |
| x24.0 | [449, 512, 256] | 37.10% | +0.17 |
| x32.0 | [598, 512, 256] | 37.05% | +0.22 |

## Why the winner works

The winning method is **`N07_fwdbwd_omp`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint cifar100_mlp3_quad_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (389 runs). Best model: `Explore/results/cifar100_mlp3_quad_bn_compressed.safetensors`.
