# Ablations

Two families of ablation sit behind the paper. The first is built into the
studies themselves, since every operator comparison at matched width is already a
component ablation. The second is a set of four targeted experiments, `e1`
through `e4`, run specifically to attack the paper's claims after they had been
formed.

The second family is the more interesting one, because each of the four changed
something in the paper. This page says what.

## The four targeted experiments

| Id | Question | Outcome | Effect on the paper |
| --- | --- | --- | --- |
| `e1` | How much does the seed move the frontier? | Non-trivially | Qualified the window claim; the noise band in Figure 3 exists because of this |
| `e2` | Does calibration size change the answer? | Stops mattering past about 8k samples | Justified the protocol sentence on calibration sizing |
| `e3` | What does ThiNet actually cost, reimplemented properly? | 41.3 points, of which a full re-solve recovers 36.3 | Became Table 5 and the paper's clearest single decomposition |
| `e4` | Does the Gauss-Newton metric help when no refit follows? | **Indistinguishable from zero on convolutions** | Contradicted the project's standing explanation; see below |

## The one that did not go the paper's way

`e4` deserves its own note, because it is the experiment that was built to confirm
a mechanism and did not.

The standing explanation in this project for two months was that a Gauss-Newton
weighting of the layer-wise residual hurts when a refit follows because the metric
and the refit correct the same error twice. The prediction is that with the
downstream network frozen, so nothing can compensate, the metric should help.

`e4` tested exactly that on convolutions. The result, recorded verbatim in
`project_log.md` line 787, is that the frozen arm is "indistinguishable from zero
rather than positive, so the double-counting explanation the project has carried
for two months looks wrong on convolutions".

Measured here: with a refit, the metric costs 0.089 points over 20 paired cells,
and is actually ahead at two of five multipliers. Without a refit, it wins 8 of
20 cells, which is a coin flip.

The paper's Section 7 sentence nonetheless states the mechanism, drawing its
supporting "wins 45 of 70" figure from the input-representation study rather than
from `e4`. That is a different experimental setting. The full accounting is in
`CLAIM_TO_EVIDENCE.md` under the Gauss-Newton subsection, and it is the one
place this supplement finds the paper claiming more than its artefacts support.

Two caveats recorded alongside `e4` at the time are worth carrying forward,
because they are reasons the experiment may be the wrong test rather than the
mechanism being wrong. Without a refit the models sit far outside the deployable
window at every budget, which is a regime the method never operates in. And the
metric is a poor estimate on early layers regardless.

## Component ablations inside the main studies

These are the ablations the paper's central claim is built from. Each is a paired
contrast at identical per-layer widths.

| Contrast | What varies | Result at the operating point |
| --- | --- | --- |
| `ot_then_ls` against `ot_intrafusion` | repair present or absent, selection fixed | +44.1, wins 24 of 24 |
| `magnitude_ls` against `magnitude` | repair present or absent, selection fixed | +32.7, wins 16 of 16 |
| `omp_ls` against `span_ls` | selection target-aware or blind, repair fixed | +0.52, t = 3.62 |
| `omp_ls` against `magnitude_ls` | selection data-driven or weight-norm, repair fixed | small |
| refinements against plain `omp_ls` | backward elimination, augmented dictionaries | -0.00, t = -0.05 |

The last row is the one with practical consequence: two refinements that cost
real implementation complexity are worth nothing at the budget anyone would ship,
and the paper says they can be removed.

## Transformer-specific ablations

| Contrast | Result |
| --- | --- |
| Sequential absorption against per-block dense targets | costs 0.44 points |
| Residual-corrected target against naive sequential | recovers 0.58 points |
| Attention heads, target-aware against blind, 18 of 72 kept | +14.40 |
| Attention heads, same contrast, 39 of 72 kept | +0.20 |
| Feed-forward neurons, same contrast | +0.08 |

The last three lines are the paper's falsifiable prediction being confirmed. The
theory says selection must recover where the design matrix has no spectral tail;
attention heads are that case, and the effect returns by a factor of roughly 180
against the feed-forward unit of the very same model on the very same data.
