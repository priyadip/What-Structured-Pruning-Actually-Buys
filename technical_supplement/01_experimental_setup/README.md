# Experimental Setup

## The object being compressed

A **unit** is a producing layer whose outputs are read by exactly one consumer.
The definition is uniform across the three studies, which is what makes their
results comparable at all.

| Study | Producer | Consumer |
| --- | --- | --- |
| Fully connected | a hidden layer | its successor |
| Convolutional | the internal convolutions of a residual block | the block's output projection |
| Transformer | `ffn.lin1` inside a feed-forward block | `ffn.lin2` |
| Transformer, Section 6 | the twelve attention heads | `attention.out_lin` |

Channels on a residual skip stream are excluded, because a skip channel is read
by an entire stage rather than by one consumer, so the decomposition of Theorem 1
does not apply to it.

BatchNorm is folded into the preceding linear map before anything is compressed,
so a unit is the pair `(w, b)`. The Transformer has nothing to fold, since no
normalisation sits between producer and consumer inside the feed-forward block.

## The operator under study

Algorithm 1 of the paper, `select-and-correct`:

1. `U` is the producer's post-activation on unlabelled calibration inputs
2. `Y` is the dense consumer's pre-activation on the same inputs
3. `M_l = round(alpha * d_l)`, clipped to `[4, N_l]`, where `d_l` is the GRIDE intrinsic dimension
4. `S = Select(U^T U, U^T Y, M)`
5. `[V; c]` solves the ridged least-squares problem against `Y` with a free intercept
6. the producer is replaced by rows `S`, the consumer by `(V, c)`

No gradients, no labels, no optimiser steps. This is enforced by construction, not
by convention: none of the compression drivers ever builds an optimiser or reads a
label.

## The three studies

### Study 1, fully connected

Driver `compression_mlp/run_autonomous_compression.py`. Eight checkpoints across four
datasets, 14 operators, budget swept on a geometric ladder derived at runtime.
Output `compression_mlp/results/compression_results.csv`, 2760 runs.

### Study 2, convolutional

Driver `compression_cnn/hidden_layer_compression/run_cnn_compression.py` with
shared machinery in `compression_cnn/common/`. Sixteen checkpoints, VGG11/16/19-BN
and ResNet18/34/50, on FashionMNIST, CIFAR-10/100 and ImageNet32. Convolution is
treated as a linear map on extracted patches, which is what lets the same
select-and-correct operator apply unchanged. Output
`.../results/compression_results_cnn.csv`, 677 runs.

### Study 3, Transformer

Drivers `compression_transformer/scripts/phase3_sweep.py` through `phase9_alloc.py` over
`typeform/distilbert-base-uncased-mnli`. Seventeen operators, budget ladder
alpha in 8 to 20, seeds 0/1/2, targets `dense` and `residual`, sequential mode on
and off. Outputs `compression_transformer/results/*.csv`, 485 runs.

Study 3 is the only one that supplies the flat-spectrum regime of Section 6,
because a Transformer offers two unit types read by a single consumer, 3072
scalar neurons and 12 attention heads, that differ in how many alternatives a
selection rule can choose between and in almost nothing else.

## The analysis layer

`analysis/scripts/` reads all of the above and produces every number in the
paper. `load_all.py` normalises the three CSV schemas into one record type, which
is the step that makes cross-architecture contrasts possible.

One detail in `load_all.py` is worth flagging because it silently changed results
when it was wrong. The fully connected driver wrote `model_name` without a
dataset prefix, so `mlp2_bn` names two different checkpoints. The loader
qualifies it as `f"{dataset.lower()}_{model_name}"`. Without that, a paired
contrast can match an MNIST run against a FashionMNIST one at the same nominal
model and budget. Fixing it moved the checkpoint count from 21 to 25 and the
headline ratio from 132 to 85.

## What each analysis script establishes

| Script | Question it answers |
| --- | --- |
| `a1_selection_vs_correction.py` | the pooled contrast, before any regime stratification |
| `a2_regime.py` | how contrasts move with distance from dense |
| `a3_component_regime.py` | the same, per component, giving Table 4 |
| `a4_who_wins_at_the_frontier.py` | which operator wins where, and the spread within the family |
| `a5_collapse_law.py` | how accuracy collapses as the budget tightens |
| `a6_at_the_frontier.py` | the central result, every contrast at each checkpoint's own frontier |
| `a7_mechanism.py` | exact residuals on 154 cells, recomputed from checkpoints |
| `a8_theory.py` | numerical verification of Theorem 1 and Corollary 2 |
| `a9_spectra.py` | spectral tails of the feed-forward and attention design matrices |
| `a10_dim_vs_rank.py` | GRIDE dimension against spectral effective rank |

`a7` and `a8` are the only ones that touch checkpoints. Everything else reads
CSV and JSON, runs on CPU, and completes in under a minute.

## Order of execution

```
a1 a2 a3 a4 a5 a6      independent, any order, CSV only
a7                     needs the 8 fully connected checkpoints
a8                     imports build_bench and load_ref from a7
a9                     needs DistilBERT and compression_transformer/scripts/core.py
a10                    needs a7's output
make_tables.py         needs a2 a3 a4 a6 a7 a9, plus attention.csv and e3_thinet.csv
make_figures.py        needs a4 a6
make_fig_mechanism.py  needs a8
make_fig_extra.py      needs a6 a7 a8
```
