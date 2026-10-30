# Supplementary material: two places the repair cannot reach

**Status: not part of the submitted PDF.** AAAI 2027 forbids an appendix, and the
main paper is capped at seven pages of content. This file holds the material cut
from what was Section 7.1, together with the longer form of the implications
discussion that was compressed into a single paragraph. Everything here is
already supported by experiments in the repository; nothing was written for this
file. The paper retains a two-sentence pointer at the end of
`\section{What this leaves standing}`.

The reason to keep it: both instances are *predictions* of Theorem 1 rather than
incidental negative results, and if a reviewer asks why sequential absorption is
not used, or why the Gauss-Newton metric is absent from the algorithm, the answer
is here rather than in the runs.

---

## The common statement

Theorem 1 is stated for a single unit whose target is the dense pre-activation.
Applied to a whole network it also predicts where the recipe should stop working:
when the error a unit must correct does not lie in `span[U_S, 1]`, since the
Pythagorean identity then bounds what the repair can remove strictly below the
error present.

Both instances below, and the selection result that occupies most of the paper,
are the same claim at different scales:

> an upstream design choice is worth exactly the part of its error that the
> downstream repair cannot already remove. Where the repair reaches, upstream
> choices are nearly free; where it does not, they are worth everything.

This is the sentence the paper keeps.

## 1. Sequential correction on a residual architecture

**The claim.** On DistilBERT, compressing each block against the already
compressed path *costs* 0.44 points against compressing every block against the
dense input, which is the opposite of what the GPTQ/SparseGPT family reports.

**Why, measured.** The chained form of the theorem is

```
Y = U V^T + 1 b^T + Xi            (Xi = error inherited from upstream)
R_fit(S) = ||(I - P_S)(D + Xi)||_F
```

so the repair removes `P_S (D + Xi)`: it corrects inherited error exactly to the
extent that the error lies in the span of the retained units, and not at all
otherwise. Instrumenting the walk:

| quantity | value |
| --- | --- |
| inherited drift at block 0, `||a_comp - a_dense|| / ||a_dense||` | 0.13 |
| inherited drift at block 5 | 0.41 |
| fraction of drift energy **outside** `span(U_S)` | 86% to 92% |

The projector cannot touch that component whatever the consumer weights are,
while fitting on a design matrix that has itself drifted degrades the part that
*was* reachable. So the mechanism is not that absorption is a bad idea; it is
that a residual stream routes most of the inherited error along a path the
producer does not own.

**The structural distinction from quantisation.** In GPTQ and SparseGPT the
inherited error travels through the very weights being refitted, so it is in
range by construction. A residual stream carries it along a path the producer
does not own, so it is out of range by construction. This is why the sign flips
rather than merely the magnitude.

**The fix, and why it is still not worth it.** Correcting the target for the
drift, that is fitting against `Y + (a_dense - a_comp)`, recovers 0.58 points
over the naive sequential version and is what one should use if compressing
sequentially at all. Against no absorption at all it is a wash.

**Evidence.** `../06_results_csv/03_transformer/ablation.csv` (`target` and `sequential`
columns), `../01_experimental_setup/transformer_study_report.md`, section on
sequential absorption, drift
instrumentation in `compression_transformer/scripts/core.py`.

## 2. A metric that changes sign with the repair

**The claim.** Weighting the layer-wise residual by the Gauss-Newton metric
`Omega = E[J^T J]` pulled back from the logits helps or hurts depending on
whether a repair follows it.

| setting | outcome |
| --- | --- |
| downstream layers refitted against the dense target | costs about 1 point at every budget |
| downstream network frozen, nothing can compensate | wins 45 of 70 cells against the unweighted optimum |

**The reading Theorem 1 suggests.** Down-weighting an output direction is only
useful if no later projection would have removed it anyway. Where a repair
follows, the metric and the repair address the same error twice and interfere.
This is the reason the algorithm in the paper carries no metric: it is not that
the metric is wrong, it is that it is redundant with the operator that follows
it.

**Evidence.** `ablations/` metric ablation, the row set where the
downstream network is frozen versus refitted.

## 3. Longer form of the implications discussion

Compressed in the paper to one paragraph; the full version made four points.

1. **Diminishing returns on the criterion.** Once a repair is in place, effort
   spent on the selection criterion has sharply diminishing returns. This is
   consistent with Wanda matching much more elaborate criteria using only a
   product of weight magnitude and input norm. Corollary 2 says this should be
   expected rather than surprising.

2. **When it stops holding.** A unit whose design matrix has no spectral tail,
   of which attention heads are the practical example. This is the one part
   that survives into the paper's compressed paragraph, because it is the
   falsifiable half.

3. **How deep a chain can be corrected across.** The residual stream limits how
   far sequential correction propagates, which bears on the depth over which
   one-shot methods should attempt to correct. See instance 1 above.

4. **What the result does not say.** It does not say selection is worthless.
   The effect is real, sign-consistent and positive at t = 3.62, and at budgets
   well past the operating point it is worth several points. It says the effect
   is small *where these methods are used*, and that a paper claiming a
   selection criterion as its contribution should demonstrate the claim there.

The protocol recommendation is kept in the paper because it is the
methodological contribution: report component ablations at the operating point
or at a named distance from dense, not averaged over a budget sweep; the two
differ here by a factor of 85. Where several operators are compared, the spread
within the family is more informative than the identity of the winner, which
changes with the model on fifteen of twenty-five. And where a component's
contribution is expressible as a residual, that measurement is stronger evidence
than the accuracy contrast it explains, since it needs no window, no ladder and
no held-out search.

## Other material cut in the same seven-page pass

Reaching seven pages of content took more than Section 7.1 and 7.2. Recorded
here so nothing is lost track of:

- **The ranking figure** (`figures/fig2_ranking.pdf`, formerly Figure 5): panel
  (a) the spread between best and worst of five repaired operators at identical
  width, panel (b) which operator takes the cell inside and outside the
  deployable window. Both of its numbers survive in the Section 5.2 prose
  (median spread 0.78 inside against 15.15 outside; ladder winner differs from
  frontier winner on 15 of 25). The script still generates it, it is simply not
  included.
- **The spectral-tail table** (`tables/t8_spectra.tex`, formerly Table 7): folded
  into the Section 6 prose, which now quotes the tail at three budgets (0.698
  against 0.544 at `M/N=0.05`, 0.443 against 0.339 at 0.25, 0.144 against 0.103
  at 0.75) and the ratio range 1.28 to 1.40. No number was dropped, only the
  two intermediate columns at 0.10 and 0.50.
- **One citation**, `sanh2020movement` (movement pruning), which appeared once
  as a list member in the introduction's enumeration of criteria. The
  bibliography is now 26 entries. Worth restoring first if space is ever
  recovered.
- **Figure heights** were reduced from 2.25in to 1.93in for the column-width
  figures and 2.32in to 2.08in for the full-width mechanism figure. This is in
  `make_fig_extra.py` and `make_fig_mechanism.py`, not in the LaTeX, so a
  regeneration keeps the fit.

## One-sidedness, kept in the paper's Limitations

The failure analysis is one-sided: we show sequential correction fails when the
error path is not owned by the producer and measure why, but have no case where
it demonstrably helps. That sentence stays in Limitations even though the
section it refers to is now a single paragraph, because the honesty cost of
dropping it is higher than the space it takes.
