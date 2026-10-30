# Evaluation Details

## The two levels of measurement

The paper measures its central effect twice, at different levels, and it is worth
being explicit about which evidence is which.

**Residual level, Section 4.** Frobenius norms computed in double precision on
154 layer-budget cells. No network is evaluated, no accuracy is involved, no
budget is searched. This is the stronger evidence because it cannot inherit an
artefact of an accuracy pipeline, and because the theorem it tests is an identity
rather than a trend.

**Accuracy level, Sections 5 and 6.** Top-1 accuracy on held-out test splits over
3922 recorded runs. This is where the claim becomes practically meaningful, and
also where every confound has to be controlled by construction.

## Metric definitions

### Accuracy

Top-1 accuracy on the held-out test split, in percent. For MNLI it is accuracy on
the matched validation set, which is the standard reported figure for that task.

### Paired contrast

Every comparison the paper reports is a difference between two operators run at
**identical per-layer widths** on the **same checkpoint** at the **same budget**:

```
contrast = acc(operator A) - acc(operator B)      at matched (model, alpha, widths)
```

This is the single most important methodological point in the evaluation. Because
the widths are identical, no contrast can be explained by one operator having
been given a larger network. It also cancels any per-cell noise the two operators
share, including the non-determinism described in the hardware note.

The paper reports, for each contrast, the mean, the standard error, the paired
t statistic, and the number of cells in which the sign is positive. The win count
is reported because it is distribution-free and does not assume the differences
are normal.

### The two residuals

For a unit with post-activation matrix `U`, consumer weights `V`, bias `b`, and
retained index set `S`:

```
R_keep(S) = || Y - U_S V_S^T - 1 b^T ||_F        keep the consumer's own weights
R_fit(S)  = min_Theta || Y - [U_S, 1] Theta ||_F  re-solve the consumer
```

with `Y = U V^T + 1 b^T`. The **correction** term is `R_keep - R_fit`. The
**selection** term is the difference in `R_fit` between a target-aware and a
target-blind rule at the same budget. Both are normalised by `||Y||_F` when
tabulated, so cells of different scale can be pooled.

### Spectral tail

```
tau_M^2 = sum_{i>M} sigma_i(U)^2
```

reported normalised as `tau_M / ||U||_F`. This is the quantity Corollary 2 bounds
the selection term by, and it is computed directly from the singular values of
the design matrix with no estimator involved.

### Cost measures

Two are reported and they diverge. **Parameter reduction** is the percentage of
whole-model weights removed. **MAC reduction** is the percentage of
multiply-accumulate operations removed. On convolutional networks these differ by
up to a factor of three, because such networks hold their parameters in the deep
low-resolution layers and their arithmetic in the early high-resolution ones. No
contrast in the paper depends on which is read, since every comparison is at
identical widths, but the two numbers are not interchangeable when quoting a
compression rate.

## The operating point

This is the definitional choice the paper's central claim rests on, so it is
stated precisely.

> The **frontier** of a checkpoint is the smallest budget `alpha` at which the
> best available operator is still within 2.5 accuracy points of the dense model.

Three properties matter.

It is defined **per checkpoint**, so it controls for regime exactly rather than
comparing a hard model at a tight budget against an easy model at a loose one.

It is **symmetric in the operators being contrasted**, since it depends on the
best operator rather than on either member of a pair, so choosing it cannot
select on the contrast being measured.

It is a **boundary point**, and therefore optimistic by roughly one percentage
point of parameters. That bias shifts every model in the same direction and
leaves the paired contrasts unchanged.

The 2.5-point threshold is a stipulation, and the paper says so. Figure 3 gives
the entire budget curve without any window, so a reader who prefers a different
threshold can read the answer off directly.

## Why reading at the operating point matters

A budget sweep is not a uniform sample of interesting configurations. At its
tightest rung a VGG19 on ImageNet32 scores 0.57 percent against a dense 49.06
percent. Averaging a component ablation over such a ladder measures the component
mostly on networks that have already collapsed, where both terms of the
decomposition are large. The paper's methodological claim is that this changes a
component's apparent value by two orders of magnitude, from a factor of 85 to
near parity.

## Statistical treatment

| Method | Where used |
| --- | --- |
| Paired t statistic | every contrast, reported as `t` |
| Standard error, shown as 2 SE | all error bars in figures |
| Win counts, for example 24 of 24 | every headline contrast |
| Pearson and Spearman correlation, log-log | Corollary 2 validation, Figure 2 |
| Median and range | operator spread within the repaired family |

No signed-rank or other nonparametric significance test is used. This is why
question 4.12 of the AAAI reproducibility checklist is answered `partial` rather
than `yes`.

## Seed variance

The stability study reruns the Transformer at seeds 0, 1, 2 with the GRIDE
dimensions re-measured per seed, so the widths themselves move with the seed
rather than being held fixed. The resulting spread is drawn as the shaded noise
band in Figure 3, and the selection contrast enters that band once roughly three
quarters of the network is retained. That is the sense in which the paper claims
the criteria become indistinguishable at deployable budgets: not that the
difference is exactly zero, but that it is inside the run-to-run noise of the
pipeline itself.
