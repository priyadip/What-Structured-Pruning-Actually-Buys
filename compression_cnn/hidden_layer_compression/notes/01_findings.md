# Findings — channel compression on VGG / ResNet

Numbers here are prose summaries of `../cnn_compression_summary.md`, which is
generated from the CSV by `summarize.py`. Where the two disagree, the generated
file is right and this one is stale.

## 1. The intrinsic dimension predicts compressibility

This is the paper's claim stated as a falsifiable prediction and then measured.
Nothing in the compression procedure sees accuracy, dataset, or model size — only
`d_l`. Across all 16 saved frontier configurations, the correlation between
`log(mean N/d)` and the achievable parameter cut is **r = +0.898**.

**All 16 models land inside ±2.5 points with zero fine-tuning**, and the cut they
allow ranges over an order of magnitude:

| | mean N/d | params cut |
|---|---|---|
| FashionMNIST VGG16 | 17.9 | **−96.9%** |
| CIFAR-10 VGG16 | 17.6 | −90.6% |
| CIFAR-100 ResNet18 | 7.6 | −36.3% |
| ImageNet32 VGG16 | 6.4 | −17.1% |
| ImageNet32 ResNet50 | — | −5.6% |

ImageNet32's VGG16 is not resistant because ImageNet32 is hard; it is resistant
because its filters genuinely occupy the space they are given. Its deepest unit
measures `d = 136.6` in 512 channels, so the GRIDE rule *refuses* to cut it — and
the refusal is correct: the ladder shows that forcing the cut to −62.9% of
parameters costs 20 points. A method budgeting by a fixed sparsity ratio would
have compressed it exactly as hard as FashionMNIST and destroyed it.

The saved widths make the same point directly. On ImageNet32 ResNet50 at the
frontier, 30 of 32 units come back at their **full** width and only 2 are
touched; on FashionMNIST ResNet34, all 16 are cut to between 4 and 45 channels
from 64–512. The rule is the same rule; the checkpoints differ.

## 2. The strategy ranking transfers from MLPs — and sharpens

Paired differences, same model and same budget, 50 cells:

| contrast | isolates | Δ |
|---|---|---|
| `S7_ot_then_ls` − `S6_ot_intrafusion` | adding a refit to OT merging | **+23.91** |
| `S2_magnitude_ls` − `S1_magnitude` | the least-squares consumer refit | **+21.62** |
| `S4_omp_ls` − `S7_ot_then_ls` | select-and-correct vs. merge-and-correct | **+7.21** |
| `S4_omp_ls` − `S2_magnitude_ls` | output-targeted selection vs. magnitude | **+3.88** |
| `S4_omp_ls` − `S3_span_ls` | targeting `Uᵀ Y` vs. spanning the patch space | **+2.14** |

Same hierarchy as the MLP study, with the same winner (`S10_augdict_fwdbwd`
narrowly, `S4_omp_ls` within 0.03 at a fraction of the cost). Two differences
worth recording:

**Merging fails harder on CNNs than on MLPs.** `S6_ot_intrafusion` and
`S8_kmeans_merge` do not merely degrade — they collapse to chance (10.0% on
CIFAR-10) at budgets where `S4_omp_ls` still reaches 84%. The reason is
structural. Merging two units into their barycentre is output-preserving only
under the ReLU exactness condition: the members must co-activate wherever the
layer is evaluated. For an MLP that means "on every sample". For a conv it means
"on every sample **and** at every spatial position" — `H·W` times more
conditions, each of which must hold. The condition is not slightly harder to
satisfy on a CNN; it is harder by the size of the feature map.

**The refit does proportionally more of the work.** On MLPs the LS-corrected
rebuild was worth +16.1 points; here the same contrast is +21.6, and adding a
refit to OT merging is worth +23.9. Removing a channel from a conv layer perturbs
every spatial position of the next layer's input, so there is more for the
consumer's closed-form solve to absorb.

## 3. Parameters and compute do not move together

Channel compression removes parameters much faster than it removes
multiply-accumulates — on CIFAR-10 VGG11, −81.8% of parameters against −46.8% of
MACs. The cause is not a defect in the method but the shape of the architecture:
a CNN's parameters sit in the deep, low-resolution layers, its MACs in the early,
high-resolution ones, and the GRIDE rule *correctly* leaves the early layers
alone because their intrinsic dimension is already close to their width
(`d = 8.5` in 64 channels for VGG11's first conv). There is nothing to cut there.

The two ratios track the GRIDE profile, not the dataset: on FashionMNIST, where
the early-layer dimensions are also small (`d = 3.5` in 64), the same model gives
−85.1% parameters and −86.3% MACs. So the asymmetry is a property of the
*checkpoint*, and it is exactly what the intrinsic dimension says it should be.

This is what motivated the second study: patch-space factorisation cuts
parameters and MACs by the same factor, and so attacks precisely the layers this
one cannot.

## 4. Two things that were expected to help and did not

**Zador allocation with a flat layer weight — negative.** Solving
`min Σ_l (m_l/d_l)^(−2/d_l)` under a MAC budget produces a per-layer allocation
that is *worse* than a single uniform multiplier at matched cost: on CIFAR-10
VGG11, 54.5% at −91.9% MACs where the uniform ladder reaches 74.3% at −91.4%.
The allocation starves exactly the early, high-MAC layers, and error introduced
there is amplified by everything downstream with nothing able to correct it. The
distortion model treats layers as independent and equally important, and they are
neither. The weighted variant (`w_l = tr(Ω_l Σ_l)`, the logit-pullback metric
contracted with the layer's own output covariance) is what the term was missing;
results are in the generated summary under *Zador-derived per-layer allocation*.

**Metric-weighted selection — negative, but by less than the MLP study.**
Scoring the OMP selection in the logit-pullback metric instead of the plain
Frobenius norm costs **−0.27 points over 4 paired cells** across two models,
winning 1 of 4 (`N1_metric_omp`); weighting the refit as well costs −0.26
(`N2_metric_omp_ls`). On VGG11 the gap is clear (−0.45, 0/2 cells); on ResNet18 it
is inside the noise (−0.11, 1/2). See `../novel_strategies/metric_omp.py`.

This bears on an open question from the MLP study, which recorded the same
failure and offered an explanation without demonstrating it: the metric
double-counts what the sequential refit already does. That explanation predicts
the metric should pay wherever nothing downstream can compensate.

**The evidence is directionally consistent but weaker than it first looked, and
the honest version is worth stating.** In study 2, `F2_rrr_jacobian` versus
`F1_rrr` — the provable optimum for the unweighted objective — comes out at
**−0.02 points over 169 paired cells**, i.e. a statistical dead heat, while
winning more cells outright (97/187 against 43/195). So the sign flips in the
predicted direction (negative where a refit follows, non-negative where less of
one does) but neither magnitude is large on CNNs, and study 2's factoriser still
refits its `1×1` mixing matrix, so it is not the fully-frozen condition the MLP
study tested. The clean experiment — metric with the downstream refit *disabled*
— has still not been run, on either architecture family. Calling this question
closed would be overstating it.

## 5. What was deliberately not built

Stage-level residual-stream compression. A ResNet18's shortcut convs and stem
total ~172k of 11.2M parameters, so restricting to internal channels leaves
**98% reachable**; the added machinery would buy at most two percent, and the
second study attacks the same weights from the other side. The arithmetic is in
`../../common/surgery.py`.
