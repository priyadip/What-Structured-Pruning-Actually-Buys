# Autonomous zero-shot compression report -- CIFAR10/mlp3_quad_bn

*generated 2026-07-25 09:03*

## Aim

Compress **mlp3_quad_bn** (3,809,034 params, dense test accuracy **65.45%**) to within **+/-2.5** points -- i.e. into **[62.95%, 67.95%]** -- with **no fine-tuning** (zero backward passes).

**Outcome: AIM ACHIEVED.** Best: **65.18%** via `N09_fisher_omp` at multiplier x32.0, widths [709, 512, 256] (2,676,175 params, -29.7%), drop +0.27 pts.

## The GRIDE foundation (all widths derive from this)

Every width in this report is `M_l = round(multiplier * d_l)` with `d_l` the GRIDE intrinsic dimension of layer *l*'s **neuron cloud**, measured at runtime on 32768 calibration samples.

| layer | original width | GRIDE d (neuron cloud) | GRIDE d (activation manifold) |
|---|---|---|---|
| h1 | 1024 | 22.15 | 31.76 |
| h2 | 512 | 22.92 | 23.48 |
| h3 | 256 | 14.38 | 16.26 |

Multiplier grid swept (derived from the discovered widths): `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]`

## Ranked results (top 25 of 390 runs)

| rank | accuracy | drop | strategy | multiplier | widths | params | reduction |
|---|---|---|---|---|---|---|---|
| 1 | **65.18%** | +0.27 | `N09_fisher_omp` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 2 | **65.18%** | +0.27 | `N09_fisher_omp_e4` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 3 | **65.16%** | +0.29 | `N08_augdict_fwdbwd` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 4 | **65.15%** | +0.30 | `N09_fisher_omp_e2` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 5 | **65.11%** | +0.34 | `10_iterative_refinement` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 6 | **65.11%** | +0.34 | `N01_span_ls` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 7 | **65.11%** | +0.34 | `N10_fisher_span_ls` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 8 | **65.11%** | +0.34 | `N09_fisher_omp` | x32.0 | [709, 458, 256] | 2,624,011 | 31.1% |
| 9 | **65.08%** | +0.37 | `N07_fwdbwd_omp` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 10 | **65.03%** | +0.42 | `N02_omp_ls_lowridge` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 11 | **65.02%** | +0.43 | `N02_omp_ls` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 12 | **65.02%** | +0.43 | `N03_gradual_omp` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 13 | **65.02%** | +0.43 | `N04_bias_calibrated_omp` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 14 | **65.01%** | +0.44 | `N05_omp_transport` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 15 | **65.00%** | +0.45 | `N09_fisher_omp` | x32.0 | [709, 512, 230] | 2,662,577 | 30.1% |
| 16 | **64.91%** | +0.54 | `N06_augdict_omp` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 17 | **64.90%** | +0.55 | `N06_augdict_omp_wide` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 18 | **64.81%** | +0.64 | `01_direct_ls_svd` | x32.0 | [709, 512, 256] | 2,676,175 | 29.7% |
| 19 | **64.81%** | +0.64 | `N09_fisher_omp` | x32.0 | [709, 367, 256] | 2,536,105 | 33.4% |
| 20 | **64.80%** | +0.65 | `N08_augdict_fwdbwd` | x24.0 | [532, 512, 256] | 2,041,630 | 46.4% |
| 21 | **64.77%** | +0.68 | `N07_fwdbwd_omp` | x24.0 | [532, 512, 256] | 2,041,630 | 46.4% |
| 22 | **64.76%** | +0.69 | `N04_bias_calibrated_omp` | x20.0 | [443, 458, 256] | 1,684,765 | 55.8% |
| 23 | **64.75%** | +0.70 | `N09_fisher_omp` | x32.0 | [709, 512, 173] | 2,632,766 | 30.9% |
| 24 | **64.73%** | +0.72 | `N09_fisher_omp` | x24.0 | [532, 512, 256] | 2,041,630 | 46.4% |
| 25 | **64.73%** | +0.72 | `N09_fisher_omp_e4` | x24.0 | [532, 512, 256] | 2,041,630 | 46.4% |

## Best per strategy

| strategy | best accuracy | multiplier | widths | what it does |
|---|---|---|---|---|
| `N09_fisher_omp` | 65.18% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N09_fisher_omp_e4` | 65.18% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N08_augdict_fwdbwd` | 65.16% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N09_fisher_omp_e2` | 65.15% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `10_iterative_refinement` | 65.11% | x32.0 | [709, 512, 256] | two-stage geometric schedule N -> sqrt(N*M) -> M, LS-corrected at each stage |
| `N01_span_ls` | 65.11% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N10_fisher_span_ls` | 65.11% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N07_fwdbwd_omp` | 65.08% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N02_omp_ls_lowridge` | 65.03% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N02_omp_ls` | 65.02% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N03_gradual_omp` | 65.02% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N04_bias_calibrated_omp` | 65.02% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N05_omp_transport` | 65.01% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N06_augdict_omp` | 64.91% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `N06_augdict_omp_wide` | 64.90% | x32.0 | [709, 512, 256] | novel (phase 2) |
| `01_direct_ls_svd` | 64.81% | x32.0 | [709, 512, 256] | SVD/leverage neuron selection + closed-form LS on the consuming layer |
| `03_activation_pattern_kmeans` | 64.69% | x32.0 | [709, 512, 256] | binarise ReLU masks, k-means the neurons, merge each cluster |
| `07_mixed_importance` | 64.46% | x32.0 | [709, 512, 256] | protect the top-20% highest-variance neurons verbatim, merge the rest, then LS |
| `08_ot_then_ls` | 64.33% | x32.0 | [709, 512, 256] | Intra-Fusion OT merge for the producing layer, then re-solve the consumer by LS |
| `05_balanced_signature` | 64.03% | x32.0 | [709, 512, 256] | k-means on [incoming | bias | outgoing] with both blocks scale-normalised |
| `09_kernel_kmeans_rbf` | 64.03% | x32.0 | [709, 512, 256] | RBF kernel k-means on balanced signatures (non-linear cluster shapes) |
| `06_greedy_wasserstein1` | 61.83% | x32.0 | [709, 512, 256] | exact 1-D W1 between neuron activation laws + average-linkage agglomeration |
| `04_cosine_ot_intrafusion` | 59.50% | x32.0 | [709, 512, 256] | Intra-Fusion baseline: keep top-|w|_1, EMD-transport the rest, column-normalised |
| `02_topm_l2_norm` | 57.35% | x32.0 | [709, 512, 256] | classic magnitude pruning baseline: keep the M largest ||(w,b)||_2 rows |

## Accuracy vs. GRIDE budget

Is the bottleneck the *budget* (M too small) or the *method*?

| multiplier | widths | best accuracy | drop |
|---|---|---|---|
| x1.5 | [33, 34, 22] | 42.94% | +22.51 |
| x2.0 | [44, 46, 29] | 46.57% | +18.88 |
| x2.5 | [55, 57, 36] | 48.89% | +16.56 |
| x3.0 | [66, 69, 43] | 49.99% | +15.46 |
| x4.0 | [89, 92, 58] | 52.74% | +12.71 |
| x5.0 | [111, 115, 72] | 54.70% | +10.75 |
| x6.0 | [133, 138, 86] | 56.92% | +8.53 |
| x8.0 | [177, 183, 115] | 59.35% | +6.10 |
| x10.0 | [221, 229, 144] | 60.99% | +4.46 |
| x12.0 | [266, 275, 173] | 62.23% | +3.22 |
| x13.0 | [288, 298, 187] | 62.75% | +2.70 |
| x13.25 | [293, 304, 191] | 62.85% | +2.60 |
| x13.38 | [296, 307, 192] | 62.77% | +2.68 |
| x13.5 | [299, 309, 194] | 63.11% | +2.34 |
| x14.0 | [310, 321, 201] | 63.51% | +1.94 |
| x16.0 | [354, 367, 230] | 63.83% | +1.62 |
| x20.0 | [443, 458, 256] | 64.76% | +0.69 |
| x24.0 | [532, 512, 256] | 64.80% | +0.65 |
| x32.0 | [709, 512, 256] | 65.18% | +0.27 |

## Why the winner works

The winning method is **`N09_fisher_omp`**.

It combines two things that pure OT/k-means merging does not have:

1. **Selection instead of averaging.** The kept neurons retain their *exact* weights, so their ReLU activations are exact. Merging two neurons is only output-preserving when they co-activate on every input (Section 6.5's exactness condition); when they do not, the merge injects error that no downstream layer can see or undo.
2. **A closed-form correction of the consuming layer against the DENSE network's own target,** solved with an intercept. Removing a ReLU neuron removes a non-negative contribution with a non-zero *mean*; only a free bias can absorb that mean. Because the target is the dense net's pre-activation at layer l+1 -- not the layer's own activations -- the fit also absorbs the error inherited from already-compressed upstream layers instead of compounding it.

In the OT language of the paper, the least-squares map plays the role of the transport plan, but it is chosen to minimise the *functional* error the next layer actually sees, rather than a geometric cost in parameter space. That is exactly the gap the counterexample in Section 4.5 predicts: distance between neuron signatures is not the quantity that governs output error.

## Reproduce

```bash
cd <PROJECT_ROOT>
python3 Explore/run_autonomous_compression.py --checkpoint cifar10_mlp3_quad_bn --id-space neuron --calib 32768
```

Raw log: `Explore/results/compression_results.csv` (390 runs). Best model: `Explore/results/cifar10_mlp3_quad_bn_compressed.safetensors`.
