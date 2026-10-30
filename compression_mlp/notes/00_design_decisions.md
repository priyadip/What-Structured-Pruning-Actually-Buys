# Design decisions — `compression_mlp/`

Why the code is shaped the way it is. Written as the work happened; negative
results are kept, not deleted.

---

## 1. The golden constraint is enforced in exactly one place

`widths_from_gride()` is the only function that produces a layer width, and it
computes nothing but `round(multiplier * d_l)` clipped to `[m_min, N_l]`. Every
strategy receives `target_m` as an argument and an `assert` in `compress_stack()`
rejects any strategy that returns a different count. There is no code path in
which a width can come from a sparsity ratio, a PCA rank, or a constant.

The multiplier grid is *derived*, not written down: `multiplier_grid()` walks a
geometric ladder up to `max(N_l / d_l)` — the multiplier at which the **last**
layer saturates — so a bigger model automatically gets a longer sweep. (Using
`min` here was a bug; see "Bugs found" below.) Phase 4 then bisects between the
last miss and the first hit, because the geometric ladder is too coarse to locate
the frontier: it steps 12 -> 16, and the true frontier for CIFAR-10 `mlp3_quad_bn`
is x13.5.

## 2. GRIDE is measured on the neuron cloud (with the activation manifold as a cross-check)

Two different point clouds can be called "the intrinsic dimension of layer *l*":

| cloud | points | coordinates | what it measures |
|---|---|---|---|
| **neuron cloud** (default) | the `N` neurons | activations over `T` calibration samples | the support dimension of the measure `mu_l` that Zador's bound is stated for |
| activation manifold | the `T` samples | the `N` neuron responses | the paper's `r_l` |

Zador's theorem bounds the *m*-point quantization error of the **neuron measure**,
so the neuron cloud is the dimension the width rule should key off. Both are
computed and both are printed and reported on every run; `--id-space` switches
which one drives the widths.

GRIDE returns one estimate per neighbour-order scale. The scalar handed to the
width rule is the **median over scales** — the robust plateau value.

For tractability the estimator sees at most 2200 points and 2048 coordinates;
coordinates are subsampled at random, which preserves distances up to the
Johnson–Lindenstrauss factor and therefore leaves the ID estimate intact.

## 3. BatchNorm is folded first

Every checkpoint is `*_bn`. A neuron is only the pair `(w, b)` once BN is folded
into the preceding `Linear` (`w <- w*gamma/sqrt(var+eps)`,
`b <- (b-mu)*gamma/sqrt(var+eps)+beta`). Folding is verified on every run: the
BN-folded dense model is evaluated and its accuracy printed next to the original.
It has matched to 0.00 points on every model tested.

Compressed models are rebuilt with `batch_norm=False`, so the running statistics
of a layer that no longer exists can never be read.

## 4. The least-squares target is the **dense network's** value, not the layer's own

This is the single most consequential choice in the file.

The obvious formulation reconstructs the layer's own activations,
`A ≈ A_keep · R`, and then sets `V' = V·R^T`. That makes each layer's fit
locally optimal but lets error *compound*: layer 2 is fitted to reproduce the
activations of an already-degraded layer 1.

Instead, `Bench.Y_orig[l]` stores the **dense, uncompressed** network's
pre-activation at layer `l+1`, and every LS-based strategy fits against that,
using activations recomputed through the layers already compressed. Each layer's
fit therefore absorbs the upstream error instead of inheriting it. This is the
GPTQ / SparseGPT sequential-correction idea transplanted from weight
quantization to structured neuron removal.

## 5. The intercept is not optional

`ls_consuming()` solves for `[V', c']` jointly against `[A_keep, 1]`. Deleting a
ReLU neuron removes a **non-negative** contribution whose *mean* is non-zero.
No reweighting of the surviving neurons can represent a constant — only a bias
can. Leaving the intercept out biases every downstream pre-activation by the
mean of what was removed, and the bias grows with the compression ratio. The
ridge never shrinks the intercept column.

## 6. The ridge is relative, not absolute

`lam = ridge * mean(diag(G))`. A fixed absolute ridge over-regularises when `M`
is large and under-regularises when it is small, which shows up as a
*non-monotone* accuracy-vs-budget curve — a tell-tale artefact, not a real
effect. Scaling by the mean Gram diagonal makes the regularisation strength
comparable across the whole multiplier sweep. (Empirically the ridge is barely
binding here: `1e-4` and `1e-6` agree to 0.01 points.)

## 7. Pivoted Cholesky instead of pivoted QR

Column-pivoted QR of `A [T,N]` and pivoted Cholesky of `A^T A [N,N]` visit
columns in the *same order*, but the QR costs `O(T·N^2)` and the Cholesky
`O(N·m)`. At `T=32768, N=2048` that is the difference between minutes and
milliseconds per layer — and the sweep runs this thousands of times.

## 8. Strategies are objects with one method, and novel ones are hot-loaded

`Strategy.apply(ctx, target_m)` is the entire contract. `LayerContext` carries
everything a method could legitimately want (both weight matrices, the layer
input, the activations, the dense target, the output metric) so a new idea is a
~10-line class. Phase 2 imports `novel_strategies/*.py` at runtime and sweeps
whatever `STRATEGIES` list it finds, which is how rounds 1–3 below were added
without touching the driver.

---

## What the full sweep settled

All claims below are measured over 96 (checkpoint x GRIDE-budget) cells across 8
MLP checkpoints on 4 datasets. Numbers are regenerated by `summarize.py`; nothing
here is transcribed by hand.

**Merging loses to rebuilding, everywhere, by a lot.** Every barycentric method
(k-means on activation patterns, balanced signatures, kernel k-means, greedy W1)
and every pure pruning/transport method (magnitude, Intra-Fusion OT) is beaten by
the corrected-least-squares family in **all 96 cells**, mean margin **+16.1
points** (range +0.1 to +47.9). The reason is the ReLU exactness condition:
replacing a group of neurons by its barycentre preserves the layer output *only*
when the group's members co-activate on every input. When they do not, the error
is injected upstream of every downstream layer and nothing can observe it, let
alone undo it. Selection keeps survivors exact and the closed-form correction
repairs what was removed. **Intra-Fusion in particular collapses to near-chance
at the budgets the GRIDE rule actually asks for**, only recovering at multipliers
so large that little compression remains.

**No seed strategy ever wins a cell — 0 of 96.**

**A correction to an earlier conclusion in this file.** After the first CIFAR-10
sweep I recorded the augmented dictionary (`N06`, `N08`), exact forward-backward
selection (`N07`), and bias calibration (`N04`) as *negative* results, and
concluded the selection rule was saturated. **That was wrong** — it was an
artefact of testing one checkpoint. Across all eight, `N08_augdict_fwdbwd`
(dictionary + forward-backward together) is the single best method, winning 31 of
96 cells, and the augmented-dictionary variants take another 26. Its margin over
plain OMP+LS is +1.27 at heavy compression and +0.25 at light — small next to the
+16 from rebuilding, but consistent and real.

The lesson is methodological and worth keeping: **do not generalise a negative
result from one checkpoint.** Every claim here is now qualified by the sweep it
came from, and `02_results.md` is generated from the CSV.

**Selection target matters more than selection algorithm.** Choosing neurons to
span the *activation* space (pivoted Cholesky) instead of the space the next layer
*reads* (OMP against `A^T Y`) costs **5.09 points** at heavy compression and 1.22
at light. That is the second-largest effect in the whole study, and it is the
operational form of the paper's Section 4.5 counterexample: distance between
neuron signatures is not the quantity that governs output error.

**The Fisher / output-Jacobian metric hurts** (`N09`, `N10`) — still a negative
result, and the most interesting one, because it is the theoretically-motivated
fix. Weighting the layer-wise residual by `M_l = E[J_l^T J_l]` (closed form for a
ReLU MLP, no autograd; see `Bench._jacobian_metrics`) costs about a point at
every budget on CIFAR-10. The explanation is that the metric assumes the rest of
the network is **fixed**, and it is not: the downstream layers are themselves
re-fitted against the dense target immediately afterwards. Directions the dense
Jacobian calls unimportant are still the raw material those later fits work with,
so down-weighting them destroys information the sequential correction would have
used. The Gauss-Newton weighting and the sequential correction solve the same
problem twice and interfere. (It is not uniformly bad — the `e2` variant wins the
MNIST 3-layer frontier — but it never leads.)

**Gradual multi-stage removal is a provable no-op** (`N03`). OMP's greedy order is
**nested**, so the first `m` atoms chosen from the full pool are the same `m`
chosen from any superset stage containing them. Staging cannot change the
outcome. Worth stating explicitly because progressive pruning *is* a real win for
gradient-based methods, and the intuition does not carry over.

**Per-layer budget allocation does not beat the uniform Zador rule** (phase 3).
Freeing each layer's multiplier independently — still GRIDE-legal, since every
width remains `round(mult_l * d_l)` with `mult_l` from the same grid — and running
coordinate descent at a *fixed parameter budget* returns the uniform allocation on
both CIFAR-10 (62.12 vs 62.10) and CIFAR-100 (`[12,12,12]` wins). This is a
**positive** result: the Zador rule's *shape* (`M_l` proportional to `d_l`) is
already the right way to spread a budget across layers, and only its *scale* is
left to choose. A fixed-sparsity rule has no such property.

**Calibration size stops mattering quickly.** 8 192 -> 16 384 -> 32 768 samples
moves the result by a few tenths with no clear trend, so the least-squares
estimation error is not what binds. 32 768 is used throughout for safety.

## Bugs found and fixed during the sweep

Recorded because each one silently produced plausible-looking wrong numbers.

1. **`--exhaustive` skipped the entire adaptive phase.** `phase1()` returned
   "achieved" whenever any configuration hit the window, and the driver only ran
   phase 2 when phase 1 had *failed*. A full sweep therefore omitted every novel
   strategy, and the frontier search picked its bisection strategies from seed
   methods only. Fixed so `--exhaustive` always runs phase 2; the affected runs
   were discarded and redone.
2. **The multiplier ladder stopped at `min(N_l/d_l)`.** That truncates the sweep
   as soon as the single tightest layer saturates. On CIFAR-100, whose last layer
   has `d=39` in 512 units, it capped the grid at x12 and made the target
   unreachable; with the ceiling at `max(N_l/d_l)` the same model clears the
   window at x18. Widths stay legal past a layer's saturation point because each
   is clipped at its own `N_l`.
3. **Absolute rather than relative ridge** made accuracy non-monotone in the
   budget (a fixed lambda over-regularises at large M). Fixed by scaling with the
   mean Gram diagonal, which restored monotonicity.
4. **The saved artefact was the *most accurate* model**, i.e. the least
   compressed one — the least interesting of the two. Now both are saved:
   `*_compressed` (highest accuracy) and `*_frontier` (smallest that still clears
   the window).
