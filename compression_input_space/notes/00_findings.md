# Findings — input reduction at a GRIDE budget

2 185 runs, 8 hidden-compressed MLPs, 4 datasets. Hidden layers frozen; only the
embedding and the linear map `A` are fitted; no labels, no gradients on any
network weight.

---

## 1. srGW-MDS loses, and the manifold term is exactly why

The method specified for this study — regularised srGW-MDS on landmarks plus a
distance-based out-of-sample extension — was beaten on every model by a
closed-form linear encoder. Three independent lines of evidence say the *manifold
preservation* is the problem, not the implementation:

**The λ ranking is monotone in switching the stress term off.** Best accuracy per
λ, pooled over all models and `r`:

| λ (weight on reconstruction) | 100 | 10 | 0.5 | 1.0 | 5.0 | 0.1 | 0.01 |
|---|---|---|---|---|---|---|---|
| best acc | **98.36** | 97.29 | 94.97 | 94.89 | 94.68 | 94.43 | 93.99 |

The more the MDS stress is down-weighted, the better the model. λ→∞ (pure
reconstruction) is the limit, and that limit is what `opt_linear` solves exactly.

**The out-of-sample ranking runs the same way.** `relaxed` optimises the true
stress most faithfully and is *worst*; `gower`, which only matches squared
distances in one closed-form shot, is best of the three:

| MNIST, r=45 | relaxed | nystrom | gower | opt_linear |
|---|---|---|---|---|
| acc | 93.33 | 96.25 | 97.29 | **98.16** |

**Landmarks are not the bottleneck.** Sweeping K ∈ {55, 100, 200, 500} across all
five selection strategies (including the specified `centroid_boundary`) never
closes the gap; the best landmark rule overall is `greedy_cover`, and the spread
between strategies (98.36 … 97.87) is far smaller than the gap to `opt_linear`.

**Why.** The first layer is *linear* before its ReLU. Preserving `W₁x` therefore
depends only on the rank-`r` subspace of the input that `W₁` reads — and the
optimal such subspace is computable in closed form (reduced-rank regression,
Eckart–Young in the data metric). Input-manifold geometry is irrelevant to a
linear readout, so spending embedding capacity on it is pure loss. This is the
same lesson as the hidden-layer study one level up: **what matters is what the
next layer reads, not the geometry of the thing being compressed.**

## 2. What did beat the ceiling — and it rehabilitates a failed idea

`opt_linear` is provably optimal among linear encoders *for the objective
`E‖W₁x − APx‖²`*. But that objective is a surrogate for accuracy. Weighting the
residual by the metric the frozen network pulls back from its logits,
`Ω = E[JᵀJ]`, gives `N1_jacobian_linear`:

| MNIST | r=15 | r=23 | r=30 |
|---|---|---|---|
| `opt_linear` | 91.90 | 96.41 ✗ | 97.59 |
| `N1_jacobian_linear` | **93.74** | **96.94 ✅** | **97.80** |

At r=23 it turns a miss into a hit. Across all `(model, r)` cells it has a mean
gap of **0.05** to the best encoder and wins **45/70**; `opt_linear` is second at
0.49.

**This is the construction that *failed* in the hidden-layer study** (`N09`,
−1 pt). The reason it works here is structural, not incidental: there, each
layer's least-squares refit already corrected downstream error, so the metric
double-counted and destroyed information the sequential correction would have
used. Here the hidden layers are **frozen** — nothing downstream can compensate —
so the metric is the only mechanism available for saying "this direction does not
matter." Same formula, opposite sign, for a reason that is now explicit.

`N3_actfreq_linear` (diagonal metric: weight units by firing rate) sits at 0.50,
i.e. no better than `opt_linear`. So the gain is the **cross-unit** structure of
Ω, not merely down-weighting dead units.

## 3. Non-linear capacity was never the constraint

`N2_rff1024` (random Fourier features → reduced-rank readout) has strictly more
capacity than any rank-`r` linear map at the same output width, and it loses
(mean gap 1.71, 0/70 wins). Combined with §1, the binding constraint is the
*metric*, not the *rank* and not the *function class*.

## 4. Four bugs that produced plausible-looking wrong numbers

Recorded because each was silent, and three of them only surfaced because
something downstream checked.

**(a) Frontier selection on a subsample is optimistic.** Two configurations chosen
on the 20 000-sample search subsample sat at −2.41 and −2.50 — just inside the
window — and fell to −2.65 and −2.54 when refitted on all data. Frontier
selection picks the point with the *least* margin by construction, so this is
systematic, not bad luck. Fixed properly: `validate_frontier()` re-fits the
candidate on the full training set and walks up the GRIDE ladder until one
genuinely clears. It fires on real data (cifar100_mlp3_oct: r=97 misses at −2.51
→ r=145 clears at −2.47).

**(b) `--models` was a substring match.** `mnist` matched
`fashionmnist_mlp2_bn_compressed`, so two models were run under another's
configuration and their artefacts overwritten. `cifar10`/`cifar100` had the same
latent collision and simply never bit. Fixed with anchored matching on `_`
boundaries, plus an error when a pattern matches nothing.

**(c) The training data order was not deterministic.** The project's train loader
is built with `shuffle=True`, so draining it returned rows in a different order
every run. That changed the GRIDE subsample, hence `d_input`, hence the entire
`r` grid: the same configuration came out at r=79 on one run and r=81 on the
next, with all accuracies still correct. Seeding immediately before the drain
pins it — two independent runs now agree to the last digit. This was the worst of
the four, because nothing looked wrong.

**(d) Saved weights could belong to a different run than their metadata.** `_save`
read `self._best_*` (the globally most accurate run) while writing the *frontier*
run's spec, so an r=193 weight matrix was written into an r=145 spec. Caught only
because `verify_fully_compressed.py` rebuilds from disk and the load raised a
shape mismatch. Fixed by pairing each run with its own artefacts and asserting
`A.shape[1] == run.r` before writing.

The common thread: **every number that gets quoted must be regenerated from disk
by a separate code path.** Three of these four would have shipped otherwise.

## 5. The honest cost accounting

A linear encoder folds: `A(Px) = (AP)x`, kept factored as `r·D + M₁·r`
parameters instead of `M₁·D`. That is a real saving and it is what the headline
62.7–91.4% reductions measure. A distance-based encoder cannot fold — it needs
`K` distances to stored landmarks, `K·D` flops per sample, which for K=500,
D=3072 is comparable to the first layer it replaces. It would have *reshaped*
the cost, not removed it. Since the distance-based encoders also lost on
accuracy, nothing is given up by preferring the folding ones.

## Open

- Everything here is MLPs on flattened images. For a conv stem the first layer is
  still linear, so the reduced-rank argument should carry, but `W₁` is shared
  across spatial positions and the reduction would apply to channels, not pixels.
- `N1` needs the frozen network to compute Ω. For a first stage applied *before*
  any hidden compression, Ω would come from the dense network — untested.
- The two-stage budget (hidden ±X then input ±Y) is allocated by hand. Both
  stages consume the same ±2.5, and nothing here optimises the split.
