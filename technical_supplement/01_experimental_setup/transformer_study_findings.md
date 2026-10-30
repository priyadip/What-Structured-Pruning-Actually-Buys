# FINDINGS

Running record. Newest first inside each section. Every number here names the
file it came from. Nothing is written here that has not been produced by a
script in `scripts/` and stored in `results/`.

---

## F1. The method transfers. The primary criterion is met.

**2026-07-27.** Source: `results/sweep.csv`, `results/frontier_omp_ls_neuron_s0.json`.

Dense DistilBERT-MNLI scores **82.21%** on `validation_matched` (9815 pairs),
close to the 82.0 the task brief quotes; the measured value is what the window
is built on, so the acceptance threshold is 79.71%.

The frontier, located by sweeping a single global `alpha` and bisecting, is

| | |
|---|---|
| `alpha*` | 12.125 |
| widths `M_l` | 1022, 968, 1040, 1186, 1152, 638 |
| accuracy | **79.86%** (delta **-2.35**) |
| parameter reduction, whole model | **28.52%** |
| parameter reduction, encoder only | 44.29% |
| parameter reduction, FFN only | 67.40% |

The success criterion asked for accuracy at or above 79.5 with at least 20%
parameter reduction. Both hold, with margin, and the reduction is quoted against
the whole 66.96M-parameter model including the 23.8M-parameter embedding table
that no method here touches.

No fine-tuning, no gradient, no label and no optimiser step enters the
compression operator. Labels appear only in the evaluation that locates
`alpha*`, which is a model-selection cost and is declared as one.

---

## F2. Merging collapses on a Transformer exactly as it does on a convolution.

**2026-07-27.** Source: `results/ablation.csv`, four budgets, matched widths.

At `alpha = 12`, where every operator receives identical per-layer widths:

| operator | accuracy |
|---|---|
| `omp_ls` (the method) | 79.68 |
| `span_ls` | 79.70 |
| `ot_then_ls` (merge, then refit) | 69.58 |
| `magnitude_ls` | 68.48 |
| `random_ls` | 55.02 |
| `random` | 45.47 |
| `magnitude` | 37.98 |
| `ot_intrafusion` (Intra-Fusion, no refit) | 35.52 |

Data-free merging falls to near chance and stays there, which is what the paper
reports for convolutions. The theorem behind that claim is stated for ReLU and
does not carry to GELU (RESEARCH R1), so this is empirical agreement with an
argument whose proof does not transfer, and the report says so.

---

## F3. The decomposition inverts twice relative to the paper.

**2026-07-27.** Source: `results/ablation.csv`, paired contrasts over
`alpha` in {16, 12, 8, 6}, both members of every pair at identical width.

| contrast | this work | paper (CNN) |
|---|---|---|
| refit, given magnitude selection | **+28.84** | +21.6 |
| refit, given merging | **+31.88** | +23.9 |
| selection over merging, given a refit | **+9.90** | +7.2 |
| OMP over magnitude, given a refit | **+12.63** | +3.9 |
| OMP over spanning, given a refit | **+0.08** | +2.1 |
| refit, given OMP selection | +5.77 | not reported |

Two things change. First, every effect that survives is larger here, so the
claim that rebuilding the consumer is what matters is if anything stronger on a
Transformer than on a convolution. Second, and against the paper, the
output-targeted criterion buys nothing over target-blind spanning: `+0.08`
points across four budgets, with the sign changing twice. On FFN units it is
enough to span the activation space; on convolutions it was not. The operative
content of the method here is the budget and the correction, not the selection
rule.

---

## F4. Sequential absorption does not transfer, and the residual stream is why.

**2026-07-27.** Source: `results/ablation.csv`, tags `C_noabsorb`, `E_residual`.

Compressing each block against the *dense* input rather than the compressed one,
which switches the absorption off entirely, is worth **+0.44 points** on
average, not a loss:

| alpha | absorb (paper's rule) | no absorption | residual-corrected target |
|---|---|---|---|
| 16 | 81.22 | 81.05 | 81.11 |
| 12 | 79.68 | 80.22 | 80.12 |
| 8  | 77.72 | 78.68 | 78.25 |
| 6  | 76.32 | 76.74 | 77.78 |

Correcting the target for the drift that arrived on the residual stream,
`Y <- Y_dense + (a_dense - a_comp)`, recovers **+0.58** over the naive absorbing
target and grows with compression, reaching `+1.46` at `alpha = 6`. Against no
absorption at all it is a wash, mean `+0.14` with the sign changing three times.

So the honest statement is not that absorption helps once corrected. It is that
on this architecture absorption is worth nothing either way, and the mechanism
the paper relies on for chained compression has no counterpart here. The
residual-corrected target is nevertheless the right one to use, since it never
costs anything and is what the next block actually reads.

---

## F5. The free intercept stops mattering under GELU.

**2026-07-27.** Source: `results/ablation.csv`, tag `D_nointercept`.

Re-solving the consumer's bias is worth `-0.03` points across four budgets,
with values `+0.03, -0.17, -0.03, +0.05`. That is a null.

This is the prediction of RESEARCH R3. The paper's argument is that dropping a
ReLU unit removes a non-negative contribution with non-zero mean, which only a
bias can represent. GELU is not non-negative, its output is near-centred for
near-centred input, and the argument loses its force. The intercept costs
nothing and is kept, but it is not carrying the result here.

---

## F6. No knee at `alpha = 1`. Getting there took two wrong turns.

**Final answer: no knee.** The entry below keeps the route, because both wrong
turns are traps that would recur in any curvature analysis of a budget ladder.

**2026-07-27, first revision 2026-07-28.** Source: `results/sweep.csv`,
`results/summary.json` field `knee_scan`.

Accuracy is monotone in the budget across the whole ladder from `alpha = 0.25`
to 48, which matches the paper. At this stage the claim of *no feature at*
`alpha = 1` appeared not to survive a look at the curvature.

An earlier version of this entry said the point at `alpha = 1` sat on the same
curve as its neighbours, which was read off the raw sequence rather than
measured. On the geometric ladder alone, differentiating accuracy with respect
to `log alpha`:

| interval | slope, points per unit `log alpha` |
|---|---|
| 0.25 to 0.5 | 1.83 |
| 0.5 to 0.75 | 8.47 |
| 0.75 to 1.0 | **17.78** |
| 1.0 to 1.5 | 6.16 |
| 1.5 to 2.0 | 4.79 |

The gain per unit budget peaks on the interval ending at `alpha = 1` and falls
by a factor of about three immediately after. The second difference there,
`-11.62`, is the largest in absolute value of all thirteen interior ladder
points, against a median of 1.44.

Two caveats kept this from being a result. The ladder's log spacing is uneven
around that region, and a second difference over uneven spacing partly reports
the sampling pattern. Separately, the first version of this scan included the
bisection points at 12.125, 12.25 and 12.5, whose spacing is a hundredth of the
ladder's; on that contaminated set the three largest curvatures were all
bisection artefacts and `alpha = 1` ranked third. The scan now runs on the
geometric ladder only.

**Resolved, later on 2026-07-28.** Source: `results/knee_test.json`, a fine scan
of 17 evenly log-spaced budgets from 0.35 to 2.8 at three seeds.

There is no knee. The feature was a sampling artefact and the original reading
was right, though not for the reason first given.

On the fine grid, adjacent points are about 0.05 apart in `log alpha` while each
accuracy carries roughly 0.19 points of seed noise, so a one-interval slope
inherits about four points per unit `log alpha` of noise on its own. The
measured local slopes swing between 0.63 and 25.72 on neighbouring intervals,
which is that noise and nothing else. The coarse ladder's apparent spike of
17.78 on `[0.75, 1.0]` was the same thing: two points, one short interval.

Averaged over the whole span on each side of `alpha = 1`, where the noise
averages down, the slopes are 8.32 against 7.31 for seed 0, 5.78 against 8.86
for seed 1, and 7.42 against 7.34 for seed 2: ratios of 0.88, 1.53 and 0.99,
straddling one. A knee at the intrinsic dimension would put all three well below
one. Fitting two segments with a free breakpoint puts it at 0.75, 0.85 and 0.65,
never at 1, and the fitted slope *rises* through it, which is the accuracy floor
at very small budgets rather than a diminishing return at `d_l`.

So the paper's conclusion holds on a seventh architecture family: `d_l` sets a
budget, not a threshold. The detour is recorded because it is a real trap.
Curvature estimated by finite differences on an uneven or bisection-contaminated
grid will manufacture a knee wherever the sampling happens to be densest.

---

## F7. Why absorption fails: the drift is not in the span of the survivors.

**2026-07-28.** Source: `results/phase5b_drift.json`, `alpha = 12`, seed 0.

Measured block by block during the walk, with the drift defined as
`||a_comp - a_dense|| / ||a_dense||` at the FFN's input and reachability as the
fraction of that drift's energy lying in the column space of the units the
budget kept:

| block | FFN write ratio | inherited drift | drift reachable by the refit |
|---|---|---|---|
| 0 | 0.543 | 0 (nothing upstream) | not applicable |
| 1 | 1.496 | 0.133 | 8.2% |
| 2 | 1.694 | 0.170 | 9.8% |
| 3 | 0.436 | 0.241 | 12.3% |
| 4 | 3.257 | 0.338 | 11.9% |
| 5 | 0.725 | 0.405 | 13.5% |

Between 86 and 92 percent of the error a block inherits lies outside the span of
its own surviving units, so no choice of consumer weights can remove it. This is
the mechanism behind F4 and it is a property of the architecture rather than of
the budget: the drift arrives on a 768-dimensional residual stream that the FFN
writes onto but does not control, whereas a convolutional block's inherited
error arrives through the very channels being refitted. Absorption is not a
transferable component; it depends on the producer owning the path the error
took.

---

## F8. Any one block can be cut to its intrinsic dimension. All six cannot.

**2026-07-28.** Source: `results/perlayer.csv`, `results/perlayer_frontier.json`.
Each row compresses one block and leaves the other five dense.

| block | `d_l` | solo `alpha*` | solo `M_l` | accuracy | delta |
|---|---|---|---|---|---|
| 0 | 84.3 | 2.0 | 169 | 79.88 | -2.33 |
| 1 | 79.8 | 1.0 | 80 | 80.62 | -1.59 |
| 2 | 85.8 | 1.0 | 86 | 80.14 | -2.07 |
| 3 | 97.8 | 1.0 | 98 | 80.40 | -1.81 |
| 4 | 95.0 | 1.0 | 95 | 81.07 | -1.14 |
| 5 | 52.7 | 1.0 | 53 | **82.31** | **+0.10** |

Five of the six blocks reach `alpha = 1`, meaning each is given exactly as many
units as its measured intrinsic dimension, and the last block improves on dense
accuracy while losing 98.3% of its FFN. Yet the joint frontier sits at
`alpha = 12`. The gap between "one block at `alpha = 1`" and "six blocks at
`alpha = 12`" is entirely error accumulation across the stack, which F7 shows
cannot be corrected because it travels on the residual stream.

The single-block result is the cleanest confirmation in this study that
`d_l` is a meaningful budget: `M_l = d_l` is sufficient in isolation on five of
six blocks, which is not something a sparsity ratio would have predicted, since
those five dimensions range from 53 to 98 units out of 3072.

---

## F9. The frontier is stable, and the robust frontier is one rung up.

**2026-07-28.** Source: `results/stability.csv`, seeds 0, 1, 2. The GRIDE
dimensions are re-measured per seed, so the widths themselves move with the
seed and that variation is inside the numbers below rather than held out of them.

| alpha | mean accuracy | sd | mean total reduction |
|---|---|---|---|
| 8 | 77.52 | 0.21 | 33.29% |
| 10 | 78.84 | 0.23 | 31.04% |
| 12 | 79.60 | 0.10 | 28.79% |
| 12.125 | 79.71 | 0.13 | 28.64% |
| 13 | 80.19 | 0.18 | 27.66% |
| 14 | 80.49 | 0.15 | 26.53% |
| 16 | 80.87 | 0.39 | 24.28% |
| 20 | 81.22 | 0.12 | 19.77% |

Mean sd across budgets is 0.19, so the robust window at `|delta| <= 2.5 - 2 sigma`
is `|delta| <= 2.12`, requiring at least 80.09%. The **robust frontier is
`alpha = 13`, 27.66% whole-model reduction at 80.19% mean accuracy**, one rung
above the tight frontier and still well past the 20% criterion. The tight
frontier is a boundary point and is optimistic by roughly one percentage point
of parameters; the robust one is what should be quoted for planning.

---

## F10. Neither the ridge nor the calibration size is load-bearing.

**2026-07-28.** Source: `results/hyper.csv`, `results/stability.csv`.

Sweeping the ridge four orders of magnitude, from `1e-6` to `1e-2`, at
`alpha = 12` moves accuracy between 80.08 and 80.16, a span of 0.08 points with
no trend. Varying calibration eightfold, 512 to 4096 examples, moves it by 0.33
points at `alpha = 12.125` and 0.64 at `alpha = 16`, again with no monotone
trend. The paper reports 0.15 to 0.33 on convolutions, so this is the same
behaviour. Neither choice is doing work that should be credited to the method,
and no failure reported here can be blamed on data starvation.

---

## F11. Attention heads: the selection rule matters here, and it is the budget
## that does not transfer.

**2026-07-28.** Source: `results/attention.csv`, `results/phase8_attention_gride.json`.

`out_lin` reads the concatenation of twelve heads and nothing else reads them,
so `(heads -> out_lin)` is the same producer/consumer pair as an FFN unit with
`gs = 64` instead of `gs = 1`, and the convolutional group-selection kernel
handles it unchanged. Removing head `h` deletes rows `[64h, 64h+64)` of `q_lin`,
`k_lin` and `v_lin` and the matching columns of `out_lin`, 196,800 parameters,
and leaves `attention_head_size` and therefore the softmax scaling alone.

The budget is where it breaks. GRIDE estimates a dimension from
nearest-neighbour distance ratios, and twelve points do not make a cloud with
neighbours. The dimension is therefore measured on the 768 coordinates the heads
produce, which is well populated, and `round(alpha d)` is quantised to whole
heads. That keeps the rule intact but at a granularity of 64: the measured
dimensions are 53.7, 53.9, 72.2, 75.3, 61.8, 35.2, which is between 0.55 and
1.18 heads' worth, so the budget resolves almost nothing on its own.

**Frontier, attention only:** `alpha = 7`, heads 6, 6, 8, 8, 7, 4, accuracy
79.95, **9.7%** whole-model reduction. `alpha = 6.5` misses at 79.65. Heads are
far less redundant than FFN neurons: cutting the FFN by 70.8% costs 2.31 points,
while cutting attention by 42% costs 2.26.

**The result that matters is the reversal.** `omp_ls` against `span_ls` at
matched head counts:

| `alpha` | heads kept | `omp_ls` | `span_ls` | difference |
|---|---|---|---|---|
| 1 | 1,1,1,1,1,1 | 46.90 | 39.61 | **+7.28** |
| 2 | 2,2,2,2,2,1 | 56.19 | 43.19 | **+13.00** |
| 3 | 3,3,3,4,3,2 | 70.28 | 55.88 | **+14.40** |
| 4 | 3,3,5,5,4,2 | 71.53 | 57.47 | **+14.06** |
| 6 | 5,5,7,7,6,3 | 79.44 | 69.80 | **+9.64** |
| 7 | 6,6,8,8,7,4 | 79.95 | 79.75 | +0.20 |
| 8 | 7,7,9,9,8,4 | 80.59 | 80.37 | +0.22 |

On FFN units the same contrast is `+0.08` at every budget (F3). On attention
heads it reaches `+14.40`, and then collapses to `+0.20` once the budget is
loose enough that most head subsets work.

This says what the value of a target-aware selection criterion depends on, and
it is not the architecture. It is how consequential an individual choice is.
Choosing 3 of 12 heads, each carrying 64 coordinates, is a decision with few
alternatives and large consequences, and scoring against what the consumer reads
is worth fourteen points. Choosing 1000 of 3072 scalar neurons is a decision
where almost any spanning set will do, and the criterion is worth nothing. The
paper's convolutional units sit between the two, at `N/d` of 6 to 27 against 31
to 58 here, which predicts its intermediate `+2.1` without any appeal to
convolution as such.

---

## F12. Uniform allocation survives, and Zador's points the wrong way.

**2026-07-28.** Source: `results/alloc.csv`, `results/alloc_search.json`.
All three hold the total at 6006 units and stay inside the rule, since every
width is `round(alpha_l d_l)` with only `alpha_l` free.

| allocation | widths | accuracy |
|---|---|---|
| uniform, `alpha = 12.125` | 1022, 968, 1040, 1186, 1152, 638 | 80.24 |
| Zador, `K` bisected to the same count | 954, 1001, 938, 833, 855, 1426 | 79.56 |
| oracle coordinate descent, stopped at 108 evaluations | 1022, 834, 1239, 1159, 1272, 480 | 80.51 |

Zador loses 0.68 points. Its multipliers are 11.3, 12.5, 10.9, 8.5, 9.0, 27.1,
so it gives block 5 more than twice everyone else's, and F8 shows block 5 is the
least sensitive block in the model. The bound allocates by intrinsic dimension
alone, and here dimension and sensitivity run opposite ways.

The oracle gains 0.26 points using held-out accuracy directly as its objective
over 108 evaluations. Its six accepted moves reach 80.35, 80.37, 80.39, 80.43,
80.47, 80.51, so the last three are worth 0.04 each; the search was stopped
there rather than run to convergence, which is recorded because it means the
0.26 is a lower bound on what an oracle could eventually find. Two standard
deviations of the seed noise is 0.38, so the whole gain sits inside the noise
and even that oracle does not reliably beat uniform. The paper's result is
reproduced.

---

## F13. Compressing both unit types together does not pay.

**2026-07-28.** Source: `results/attention.csv`, tag `joint`.

Attention and FFN in the same pass, attention first inside each block because
that is the order the forward runs, over a grid of five FFN multipliers crossed
with four attention ones. Twelve of the twenty stay inside the window; the three
with the most reduction:

| FFN `alpha` | attention `alpha` | accuracy | whole model |
|---|---|---|---|
| 16 | 10 | 79.82 | 29.12% |
| 14 | 12 | 79.97 | 29.02% |
| 16 | 12 | 80.57 | 26.81% |
| FFN alone, best configuration | none | 79.90 | **29.97%** |

No split beats spending the whole budget on the FFN. This follows from the
redundancy asymmetry rather than from any interaction between the two: a
parameter removed from an FFN buys more accuracy per parameter than one removed
from attention, so at a fixed whole-model target the FFN should be exhausted
first. The sweep was worth running because the opposite result would have been
the interesting one.

---

## F14. The two Transformer-specific improvements are close to additive.

**2026-07-28.** Source: `results/sweep.csv`, all four (cloud, target) frontiers.

| cloud | target | `alpha*` | accuracy | whole model |
|---|---|---|---|---|
| `filter_proj` | residual | 13.53 | 79.90 | **29.97%** |
| `filter_proj` | dense | 13.88 | 79.76 | 29.66% |
| `neuron` | residual | 11.16 | 79.96 | 29.63% |
| `neuron` | dense | 12.12 | 79.86 | 28.52% |

Against the direct transplant of the paper's rule, `neuron` and the dense
target, measuring the budget on the weight cloud instead of the activation cloud
is worth 1.14 points of whole-model reduction, correcting the target for
residual drift is worth 1.11, and taking both is worth 1.45. Not quite additive
but close, which suggests they act on different parts of the error.

Note that `alpha*` is not comparable across clouds, because `d_l` differs: the
`filter_proj` dimensions are smaller in the middle blocks, so the same
multiplier buys fewer units. Every ablation and baseline comparison in this
study uses `neuron` with the dense target, so none of those numbers is credited
to either improvement.

---

## F0. Architecture facts established by inspection

**2026-07-27.** Source: `transformers` 5.14.1,
`models/distilbert/modeling_distilbert.py`, read directly rather than assumed.

`TransformerBlock.forward` is

```
attention_output = self.attention(hidden_states)[0]
attention_output = self.sa_layer_norm(attention_output + hidden_states)
ffn_output       = self.ffn(attention_output)          # lin2(gelu(lin1(x)))
ffn_output       = self.output_layer_norm(ffn_output + attention_output)
```

Three consequences, all of which contradict or sharpen the framing in the task
brief:

1. **The FFN unit's consumer is `lin2`, not "the residual add plus the next
   layer".** The 3072-dimensional intermediate is read by exactly one module.
   The residual add consumes `lin2`'s *output*, which is 768-dimensional and is
   not what we are cutting. So the unit is `(lin1, lin2)` and it is the
   identical pair to an MLP's `(layer l, layer l+1)`.

2. **No LayerNorm sits between `lin1` and `lin2`.** There is nothing to fold for
   an FFN unit. `sa_layer_norm` is upstream of `lin1` and `output_layer_norm` is
   downstream of the residual add. The BatchNorm-folding step of the CNN study
   has no counterpart here, and inventing one would be wrong.

3. **`attention_output` is simultaneously `lin1`'s input and the tensor added to
   `lin2`'s output.** That is what makes the residual-absorbing target of R2
   expressible: the correction the block needs is `Y_dense + (x_dense - x_comp)`
   with the same `x` on both sides.

Parameter accounting, from `config`: 6 layers, dim 768, hidden_dim 3072, GELU,
12 heads, vocab 30522, total 66,955,779 parameters.
