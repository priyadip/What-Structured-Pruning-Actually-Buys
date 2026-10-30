# Select-and-correct on a Transformer

**Model.** `typeform/distilbert-base-uncased-mnli`, 6 blocks, `dim` 768,
`hidden_dim` 3072, GELU, 12 heads, 66,955,779 parameters.
**Task.** MNLI, evaluated on `validation_matched` (9815 pairs).
**Dense accuracy, measured.** 82.21%. The acceptance window is `+/- 2.5`, so
79.71% or better.
**Compression operator.** No gradient, no label, no optimiser step, no
fine-tuning. Calibration is 2048 unlabelled MNLI *training* inputs. Labels enter
only in the evaluation that locates `alpha*`, which is a model-selection cost
and is declared as one.

Everything below is regenerated from `results/*.csv` by `scripts/analyze.py`,
which re-reads the raw files with a different code path from the one that wrote
them. `scripts/selftest.py` checks that the selection kernel at `gs = 1` returns
the same indices as the MLP study's `omp_select` and `pivoted_cholesky_select`,
that the refit equals a dense `lstsq` with intercept to 8e-8, that a
keep-everything budget is an exact no-op on all six blocks, and that no padding
token enters the design matrix.

---

## 1. Adapting the method

### The unit

The producer is `ffn.lin1`, whose 3072 output neurons are the units, and the
consumer is `ffn.lin2`. Reading `TransformerBlock.forward` out of the installed
`transformers` rather than trusting a description of it settles three questions
the task brief left ambiguous, and two of its premises turn out to be wrong.

```
attention_output = sa_layer_norm(attention(h) + h)
ffn_output       = lin2(gelu(lin1(attention_output)))
output           = output_layer_norm(ffn_output + attention_output)
```

**The consumer is `lin2`, not the residual add.** Exactly one module reads the
3072-dimensional intermediate; `selftest.py` asserts it by scanning every
`Linear` in the block for `in_features == 3072`. The residual add consumes
`lin2`'s *output*, which is 768-dimensional and is not what is being cut. So the
unit is the identical producer/consumer pair the MLP study compressed, and the
CNN study's grouped machinery specialises to it at `gs = 1`.

**There is no LayerNorm to fold.** `sa_layer_norm` sits upstream of `lin1` and
`output_layer_norm` downstream of the residual add. Nothing lies between
producer and consumer except GELU. The BatchNorm folding of the CNN study has no
counterpart here and inventing one would be wrong.

**Sequence length.** Rows of the design matrix are (example, token) pairs with
padding excluded, since a pad position's activations are an artefact of the batch
shape and would enter the Gram with the same weight as a real token. 2048
calibration examples give 80,539 valid token rows, of which 65,536 are sampled,
against 3073 unknowns: 21.3-fold over-determination. Fits below twofold are
refused, and `selftest.py` checks the guard fires.

### The budget

`M_l = round(alpha d_l)` with `d_l` the GRIDE intrinsic dimension, median over
neighbour-order scales. No width in this study comes from a PCA rank, a sparsity
ratio or a hand-set number. Three clouds are defensible for an FFN layer and all
three are measured; `neuron` is the default because an FFN layer *is* an MLP
layer and that is what the MLP study used.

| block | `neuron` | `filter_proj` | `weight` | `N/d` (neuron) | sd over seeds |
|---|---|---|---|---|---|
| 0 | 84.27 | 84.39 | 84.17 | 36.5 | 2.14 |
| 1 | 79.81 | 70.62 | 70.45 | 38.5 | 3.74 |
| 2 | 85.78 | 64.14 | 62.36 | 35.8 | 2.41 |
| 3 | 97.78 | 69.14 | 69.14 | 31.4 | 3.49 |
| 4 | 95.03 | 53.80 | 53.80 | 32.3 | 3.70 |
| 5 | 52.65 | 55.03 | 55.03 | 58.3 | 0.62 |

Projecting the weight rows onto the row space of the layer input barely moves
the estimate, because with 65,536 rows in 768 dimensions the input Gram is
essentially full rank: `filter_proj` and `weight` agree to within 2 units on
five of six blocks. The `neuron` cloud is the one that differs, and it differs
most in the middle blocks, where what the units *emit* is spread over more
directions than what they *are*.

### The correction

`[V; c] = argmin ||[U_S, 1][V;c]' - Y||_F^2 + lambda ||V||_F^2` with `lambda`
relative to the mean Gram diagonal. Two targets are available on a residual
architecture and both are tested:

- `dense`: `Y = ffn_dense(a_dense)`, the paper's rule transplanted.
- `residual`: `Y = ffn_dense(a_dense) + (a_dense - a_comp)`, which asks the
  refit to also correct drift that arrived on the residual stream. This is new;
  a ResNet's internal refit was never given the option, because a ResNet block's
  internal error and its stream error are not separable the same way.

---

## 2. Result

### Frontiers

| cloud | target | `alpha*` | accuracy | delta | whole model | encoder | FFN |
|---|---|---|---|---|---|---|---|
| `filter_proj` | residual | 13.53 | 79.90 | -2.31 | **29.97%** | 46.54% | 70.83% |
| `filter_proj` | dense | 13.88 | 79.76 | -2.46 | 29.66% | 46.06% | 70.09% |
| `neuron` | residual | 11.16 | 79.96 | -2.25 | 29.63% | 46.01% | 70.01% |
| `neuron` | dense | 12.12 | 79.86 | -2.35 | 28.52% | 44.29% | 67.40% |

The best configuration keeps 1142, 956, 868, 936, 728, 745 units out of 3072 per
block. The two improvements over the transplanted rule are close to additive:
measuring the budget on the weight cloud rather than the activation cloud is
worth 1.14 points of whole-model reduction, correcting the target for residual
drift is worth 1.11, and taking both is worth 1.45. The `alpha*` values are not
comparable across clouds, since `d_l` differs.

At the configuration that is the direct transplant of the paper's rule, `neuron`
and the dense target, the widths are 1022, 968, 1040, 1186, 1152, 638. That is
the configuration every ablation and baseline comparison below uses, so nothing
in those tables is credited to the two Transformer-specific improvements.

Three denominators are quoted because a Transformer has no single honest one.
The word-embedding table is 23,835,648 parameters, 35.6% of the model, and no
method here touches it, so FFN compression alone is capped at 42.3% of the whole
model. The 20% criterion is stated against the whole model, which is the strict
reading, and is met with margin.

**Primary criterion: met.** 79.90% at 29.97% whole-model parameter reduction,
against a requirement of 79.5% at 20%.

### Stability and the robust frontier

Three calibration seeds, with `d_l` re-measured per seed so the widths move with
the seed too, on the `neuron` cloud with both targets:

| `alpha` | dense target | sd | residual target | sd | mean whole-model reduction |
|---|---|---|---|---|---|
| 10 | 78.84 | 0.23 | | | 31.04% |
| 11.16 | | | 79.72 | 0.23 | 29.74% |
| 12 | 79.60 | 0.10 | 80.07 | 0.05 | 28.79% |
| 12.125 | 79.71 | 0.13 | | | 28.64% |
| 13 | 80.19 | 0.18 | 80.37 | 0.28 | 27.66% |
| 14 | 80.49 | 0.15 | 80.47 | 0.12 | 26.53% |
| 16 | 80.87 | 0.39 | | | 24.28% |

Mean sd is 0.189 with the dense target and 0.170 with the residual one, so the
robust window at `|delta| <= 2.5 - 2 sigma` is `|delta| <= 2.12` and `<= 2.16`
respectively. **The robust frontier is `alpha = 12` with the residual target:
80.07% at 28.79% whole-model reduction**, against `alpha = 13` and 27.66% with
the dense target. The tight frontier is a boundary point and is optimistic by
roughly one percentage point of parameters; the robust number is the one to plan
against. Both robust frontiers are measured on the `neuron` cloud, so the
`filter_proj` configuration that gives the best tight frontier has no robust
counterpart here and is quoted only as a tight frontier.

Neither the ridge nor the calibration size is load-bearing. Sweeping the ridge
from `1e-6` to `1e-2` moves accuracy across a span of 0.08 points with no trend.
Varying calibration eightfold, 512 to 4096 examples, moves it 0.33 points at
`alpha = 12.125` and 0.64 at `alpha = 16`, again without a trend, matching the
0.15 to 0.33 the paper reports on convolutions.

---

## 3. Comparison at matched width

Every operator receives identical per-layer widths, so what separates them is
how units are removed and never how many survive. Mean accuracy over
`alpha` in {24, 16, 12, 8, 6, 4, 2}, and the paired difference from `omp_ls`
computed budget by budget:

| operator | mean accuracy | `omp_ls` minus this | wins |
|---|---|---|---|
| `fwdbwd_ls` | 76.75 | -0.02 | 4/7 |
| **`omp_ls`** | **76.73** | | |
| `span_ls` | 76.36 | +0.37 | 5/7 |
| `thinet_lsfull` | 76.17 | +0.56 | 6/7 |
| `thinet` (as published) | 74.47 | +2.26 | 7/7 |
| `thinet_densetgt` | 71.19 | +5.32 | 6/6 |
| `omp` (no refit) | 69.20 | +7.53 | 7/7 |
| `ot_then_ls` (merge, then refit) | 66.64 | +10.09 | 7/7 |
| `magnitude_prod_ls` | 66.40 | +10.33 | 7/7 |
| `magnitude_ls` | 63.99 | +12.74 | 7/7 |
| `random_ls` | 53.55 | +23.18 | 7/7 |
| `random` | 48.55 | +28.19 | 7/7 |
| `kmeans_merge` | 40.82 | +35.91 | 7/7 |
| `ot_intrafusion` (Intra-Fusion) | 38.96 | +37.77 | 7/7 |
| `magnitude` (no refit) | 38.35 | +38.39 | 7/7 |

**Secondary criterion: met.** The method beats magnitude pruning by 38.39 points
without a refit and 12.74 with one, and beats optimal-transport merging by 37.77
without a refit and 10.09 with one, winning every one of seven matched budgets
in all four comparisons.

Data-free merging collapses to near chance and stays there, which is what the
paper reports for convolutions. The theorem behind that claim is stated for
ReLU and uses positive homogeneity, so it does not transfer to GELU. This is
empirical agreement with an argument whose proof does not carry over, and it is
reported as such rather than as a confirmation of the theory.

`thinet` implemented as published is a far stronger baseline here than on
convolutions, losing 2.26 points rather than the 41.3 the paper reports.
Decomposing the gap: replacing its per-channel scalar rescale by the full refit
is worth +1.70, and replacing its greedy subtractive selection by group OMP is
worth a further +0.56. Giving it the dense target while it still has only a
scalar rescale *costs* 3.28 points, the opposite sign to the paper's +14.5,
because `|S|` scale factors cannot reach a target that differs from the layer's
own output. The three choices are strongly sub-additive and only the refit
carries real weight.

---

## 4. Ablation

Paired contrasts over `alpha` in {16, 12, 8, 6}, both members of every pair at
identical width.

| contrast | this work | paper (CNN) | t |
|---|---|---|---|
| refit, given merging | **+31.88** | +23.9 | 32.3 |
| refit, given magnitude selection | **+28.84** | +21.6 | 24.7 |
| OMP over random, given a refit | +24.96 | not reported | 8.6 |
| OMP over magnitude, given a refit | **+12.63** | +3.9 | 12.5 |
| selection over merging, given a refit | **+9.90** | +7.2 | 24.2 |
| refit, given OMP selection | +5.77 | not reported | 4.9 |
| residual target over dense target | +0.58 | not applicable | 1.8 |
| OMP over spanning, given a refit | **+0.08** | +2.1 | 0.5 |
| free intercept | **-0.03** | load-bearing | -0.6 |
| sequential absorption, on over off | **-0.44** | load-bearing | -1.9 |

Three of these disagree with the paper and each disagreement is informative.

**The output-targeted selection criterion buys nothing.** Against target-blind
spanning of the activation space it is worth 0.08 points across four budgets
with the sign changing twice. On convolutions it was worth 2.1. What carries the
result on a Transformer is the budget and the correction, not the selection
rule, and the effects that do survive are all larger here than there.

**The free intercept stops mattering.** The paper's argument is that dropping a
ReLU unit removes a non-negative contribution with non-zero mean, and only a
bias can represent a constant. GELU is not non-negative and its output is
near-centred for near-centred input, so the argument loses its force. Measured,
the intercept is worth -0.03 points. It costs nothing and is kept, but it is not
doing work here.

**Sequential absorption does not transfer, and the residual stream is why.**
Compressing each block against the *dense* input, which switches absorption off
entirely, is worth +0.44 points rather than losing anything. Measuring the drift
during the walk explains it:

| block | FFN write ratio | inherited drift | fraction of drift in the span of the survivors |
|---|---|---|---|
| 1 | 1.496 | 0.133 | 8.2% |
| 2 | 1.694 | 0.170 | 9.8% |
| 3 | 0.436 | 0.241 | 12.3% |
| 4 | 3.257 | 0.338 | 11.9% |
| 5 | 0.725 | 0.405 | 13.5% |

Between 86 and 92 percent of the error a block inherits lies outside the column
space of the units its budget kept, so no choice of consumer weights can remove
it. On a convolution the inherited error arrives through the very channels being
refitted; here it arrives on a 768-dimensional residual stream the FFN writes
onto but does not control. Absorption is not a portable component. It requires
the producer to own the path the error took.

The residual-corrected target recovers +0.58 over the naive absorbing target and
grows with compression, reaching +1.46 at `alpha = 6`. Against no absorption at
all it is a wash, mean +0.14 with the sign changing three times. It is the right
target to use, since it never costs anything and is what the next block actually
reads, but the honest statement is that on this architecture absorption is worth
nothing either way.

---

## 5. Analysis

### Does `log(mean N/d)` predict compressibility?

**Not by extrapolation, and the within-model test has no resolution.**

The paper's regression, recomputed in this session from
`original_research/scripts/load_data.py`, is
`cut = 61.01 log(mean N/d) - 91.07` with `r = +0.898`, fitted on 16 CNNs over
`log(N/d)` in `[1.803, 3.289]`. DistilBERT sits at `log(mean N/d) = 3.659`,
0.37 beyond the top of that range, and the line predicts a cut of 132%, which
is not a possible value. Observed is 70.1% of the FFN and 29.7% of the model.
A linear fit within one architecture family does not survive the move to
another, which is the expected failure of an unbounded predictor for a bounded
quantity but is worth stating plainly rather than quietly rescaling.

The six blocks vary in `N/d` by a factor of 1.9, so the predictor can also be
tested without leaving the model, using each block's solo accuracy at fixed
`alpha` as its compressibility. The correlation is `+0.745` at `alpha = 1`,
`+0.586` at 2 and `+0.630` at 4, the same sign as the paper. But with six points
the leave-one-out range spans `[-0.381, +0.885]` at `alpha = 1` and straddles
zero at all three budgets: block 5 alone carries the entire correlation, and
dropping it flips the sign. **This is not evidence.** It is reported so that a
positive `r` computed on six correlated within-model layers is not mistaken for
a replication of a result established across 24 independent checkpoints.

**Tertiary criterion: not met.**

### Is there a knee at `alpha = 1`?

**No.** A fine scan of 17 evenly log-spaced budgets from 0.35 to 2.8 at three
seeds settles it. Averaged over the whole span on each side of `alpha = 1`,
where the noise averages down, the slope of accuracy in `log alpha` is:

| seed | below `alpha = 1` | above | ratio | free breakpoint |
|---|---|---|---|---|
| 0 | 8.32 | 7.31 | 0.88 | 0.75 |
| 1 | 5.78 | 8.86 | 1.53 | 0.85 |
| 2 | 7.42 | 7.34 | 0.99 | 0.65 |

The ratios straddle one; a knee at the intrinsic dimension would put all three
well below it. A two-segment fit with a free breakpoint places the break at
0.65, 0.75 and 0.85, never at 1, and the fitted slope *rises* through it, which
is the accuracy floor at very small budgets rather than a diminishing return at
`d_l`. The ladder is monotone throughout.

This required care. On the coarse geometric ladder the second difference at
`alpha = 1` is `-11.62`, the largest in absolute value of all thirteen interior
points, which looks like a knee and is not one. Adjacent fine-grid points are
about 0.05 apart in `log alpha` while each accuracy carries 0.19 points of seed
noise, so a one-interval slope inherits roughly four points per unit `log alpha`
of pure noise; the measured local slopes swing between 0.63 and 25.72 on
neighbouring intervals. Finite-difference curvature on an uneven grid reports
the sampling pattern.

### Where does the accuracy go, and does allocation matter?

Compressing one block and leaving the other five dense:

| block | `d_l` | solo `alpha*` | solo `M_l` | accuracy | delta |
|---|---|---|---|---|---|
| 0 | 84.3 | 2.0 | 169 | 79.88 | -2.33 |
| 1 | 79.8 | 1.0 | 80 | 80.62 | -1.59 |
| 2 | 85.8 | 1.0 | 86 | 80.14 | -2.07 |
| 3 | 97.8 | 1.0 | 98 | 80.40 | -1.81 |
| 4 | 95.0 | 1.0 | 95 | 81.07 | -1.14 |
| 5 | 52.7 | 1.0 | 53 | **82.31** | **+0.10** |

Five of six blocks reach `alpha = 1`, meaning each is given exactly as many
units as its measured intrinsic dimension, and the last improves on dense
accuracy while losing 98.3% of its FFN. Yet the joint frontier sits at
`alpha = 12`. The whole gap is error accumulation across the stack, which the
drift measurement above shows cannot be corrected.

This is the clearest evidence in the study that `d_l` is a meaningful budget:
`M_l = d_l` suffices in isolation on five of six blocks, and those five
dimensions range from 53 to 98 units out of 3072. A sparsity ratio would not
have predicted a per-block target that varies by a factor of two.

Freeing the multipliers does not help. All three allocations below hold the
total unit count at 6006 and stay inside the rule, since every width is still
`round(alpha_l d_l)` and only `alpha_l` is allowed to differ.

| allocation | widths | accuracy |
|---|---|---|
| uniform, `alpha = 12.125` | 1022, 968, 1040, 1186, 1152, 638 | 80.24 |
| Zador, `K` bisected to the same count | 954, 1001, 938, 833, 855, 1426 | 79.56 |
| oracle coordinate descent, stopped at 108 evaluations | 1022, 834, 1239, 1159, 1272, 480 | 80.51 |

Zador's allocation, derived from the same bound that sets the widths, loses 0.68
points. Its multipliers are 11.3, 12.5, 10.9, 8.5, 9.0, 27.1, so it gives block 5
more than twice everyone else's, and the table above shows block 5 is the least
sensitive block in the model. The bound allocates by intrinsic dimension alone,
and here dimension and sensitivity run opposite ways.

Coordinate descent over per-layer multipliers, using held-out accuracy directly
as its objective and therefore an oracle rather than a method, gains 0.26 points
over 108 evaluations. Its six accepted moves are worth 80.35, 80.37, 80.39,
80.43, 80.47, 80.51, so the last three are worth 0.04 each and the search was
stopped there rather than run to convergence. Two standard deviations of the
seed noise is 0.38, so the whole gain sits inside the noise: even an oracle with
a hundred labelled evaluations does not reliably beat spending the budget
uniformly. The paper's finding is reproduced.

---

## 6. Limitations

**Attention.** Selection and correction transfer to heads as a `gs = 64` grouped
problem, and the selection rule matters far more there than on FFN units. The
budget half does not apply cleanly: GRIDE needs a cloud with enough points to
have neighbours, and twelve heads are below that. Measuring the dimension on the
768 coordinates the heads produce and quantising to whole heads keeps the rule
intact but at a granularity of 64, coarse enough that the budget carries little
information. So the method has a minimum unit count, and DistilBERT's attention
is under it. See section 7.

**Embeddings.** 35.6% of the model is a word-embedding table this method cannot
touch, since an embedding row has no consumer to refit against. Any whole-model
reduction reported here is bounded by that.

**Residual streams.** Confirmed rather than solved. Internal units are
compressible and stream channels are not, and beyond that the stream also blocks
error absorption between blocks, which the paper's ResNet experiments never had
to confront because a ResNet's internal refit was never asked to correct
something that arrived from outside its own producer.

**Longer sequences.** All experiments truncate at 128 tokens. The design matrix
grows linearly in sequence length and the method has no dependence on position,
so nothing should change, but this is untested.

**GELU.** The method transfers. The explanation for why merging fails does not,
since it rests on ReLU's positive homogeneity. The empirical collapse of merging
is reproduced, but this study has no theorem for it.

**One checkpoint.** Every number here is one model on one task. The paper's
claims rest on 24 checkpoints across five datasets; this is one more
architecture family, not a second study.

---

## 7. Attention heads

### The adaptation

`out_lin` reads the concatenation of twelve heads and nothing else reads them,
so `(heads -> out_lin)` is the same producer/consumer pair as an FFN unit with
`gs = 64` instead of `gs = 1`, and the convolutional group-selection kernel
handles it unchanged. Removing head `h` deletes rows `[64h, 64h+64)` of `q_lin`,
`k_lin` and `v_lin` and the matching columns of `out_lin`, 196,800 parameters,
and leaves `attention_head_size` and therefore the softmax scaling alone.

The budget is where it strains. GRIDE estimates a dimension from
nearest-neighbour distance ratios and twelve points do not make a cloud with
neighbours. The dimension is therefore measured on the 768 coordinates the heads
produce, which is well populated and is the same `neuron` cloud used for the
FFN, and `round(alpha d)` is then quantised to whole heads. That keeps the rule
intact but at a granularity of 64. The measured dimensions are 53.7, 53.9, 72.2,
75.3, 61.8, 35.2, between 0.55 and 1.18 heads' worth, so on its own the budget
resolves almost nothing and `alpha` is doing most of the work.

### Result

| `alpha` | heads kept | accuracy | delta | whole model | encoder |
|---|---|---|---|---|---|
| 1 | 1,1,1,1,1,1 | 46.90 | -35.31 | 19.40% | 30.12% |
| 3 | 3,3,3,4,3,2 | 70.28 | -11.93 | 15.87% | 24.65% |
| 6 | 5,5,7,7,6,3 | 79.44 | -2.77 | 11.46% | 17.80% |
| **7** | **6,6,8,8,7,4** | **79.95** | **-2.26** | **9.70%** | **15.06%** |
| 8 | 7,7,9,9,8,4 | 80.59 | -1.62 | 8.23% | 12.78% |

Heads are far less redundant than FFN neurons per parameter removed. Cutting the
FFN by 70.8% costs 2.31 points; cutting attention by 46% costs 2.26 and yields
only 9.70% of the model against 70.83% of the FFN yielding 29.97%.

### The finding that matters, and it is not the frontier

`omp_ls` against `span_ls` at matched head counts, alongside the same contrast on
FFN units:

| `alpha` | heads kept | `omp_ls` | `span_ls` | difference |
|---|---|---|---|---|
| 1 | 1,1,1,1,1,1 | 46.90 | 39.61 | **+7.28** |
| 2 | 2,2,2,2,2,1 | 56.19 | 43.19 | **+13.00** |
| 3 | 3,3,3,4,3,2 | 70.28 | 55.88 | **+14.40** |
| 4 | 3,3,5,5,4,2 | 71.53 | 57.47 | **+14.06** |
| 6 | 5,5,7,7,6,3 | 79.44 | 69.80 | **+9.64** |
| 7 | 6,6,8,8,7,4 | 79.95 | 79.75 | +0.20 |
| 8 | 7,7,9,9,8,4 | 80.59 | 80.37 | +0.22 |
| FFN units, all budgets | 1000 or so of 3072 | | | +0.08 |

Section 4 reported that the output-targeted criterion buys nothing over spanning
on FFN units. On attention heads at a tight budget it is worth up to 14.40
points, and then collapses to 0.20 once the budget is loose enough that most head
subsets work.

So the value of a target-aware selection rule is not a property of the
architecture. It is a property of how consequential an individual choice is.
Choosing 3 of 12 heads, each carrying 64 coordinates, admits few alternatives and
scoring against what the consumer reads is worth fourteen points. Choosing a
thousand of 3072 scalar neurons is a decision where almost any spanning set will
do. The paper's convolutional units sit between the two, at `N/d` of 6 to 27
against 31 to 58 here, which accounts for its intermediate +2.1 without any
appeal to convolution as such. This is the sharpest prediction the study makes
and it is testable on the existing CNN results.

### Combining the two unit types does not pay

Compressing attention and FFN in the same pass, attention first inside each block
because that is the order the forward pass runs, over a grid of five FFN
multipliers crossed with four attention ones. Twelve of the twenty
configurations stay inside the window; the three that give the most reduction,
against the FFN-only frontier:

| FFN `alpha` | attention `alpha` | accuracy | whole model |
|---|---|---|---|
| 16 | 10 | 79.82 | 29.12% |
| 14 | 12 | 79.97 | 29.02% |
| 16 | 12 | 80.57 | 26.81% |
| FFN alone, `filter_proj` and residual target | none | 79.90 | **29.97%** |

No split in the grid beats spending the whole budget on the FFN. This follows from
the redundancy asymmetry above rather than from any interaction: a parameter
removed from an FFN buys more accuracy-per-parameter than one removed from
attention, so at a fixed whole-model target the FFN should be exhausted first.
The joint sweep is worth running anyway, because the opposite result would have
been the interesting one.

---

## Provenance

| claim | file |
|---|---|
| dense accuracy, parameter counts, no-op check | `results/phase0_ground_truth.json` |
| GRIDE dimensions, three clouds, seed spread | `results/phase2_gride.json` |
| ladder, frontiers, cloud and target variants | `results/sweep.csv` |
| matched-width comparison | `results/baselines.csv` |
| ablation contrasts | `results/ablation.csv` |
| drift and reachability | `results/phase5b_drift.json` |
| seeds, robust frontier, calibration curve | `results/stability.csv` |
| ridge and row-count sweeps | `results/hyper.csv` |
| per-block solo ladders | `results/perlayer.csv` |
| allocation: uniform, Zador, oracle search | `results/alloc.csv` |
| fine knee scan and two-segment fit | `results/knee_test.json` |
| attention heads | `results/attention.csv` |
| the saved model, reloaded and re-evaluated | `results/artifact.json` |
| everything above, recomputed | `results/summary.json` |

The compressed model is `artifacts/distilbert_mnli_ffn.safetensors`, 46,887,170
parameters against 66,955,779 dense, 187.6 MB. `scripts/export_best.py` writes
it, then builds a fresh model with the compressed shapes, loads the tensors into
it with `strict=True`, and re-evaluates: 79.8981% both in-process and after the
round trip.

`results/baselines_thinet_densetgt_buggy.csv` holds seven rows produced by an
incorrect `thinet_densetgt` before the greedy score was corrected to use the
residual `Z_full - Y`; they are kept out of the main file and out of every
number above.
