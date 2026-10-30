# Claim to Evidence Map

Every quantitative claim in the paper, with the file it comes from and the script
that produced it. Paths are relative to the repository root.

Read the columns as: what the paper says, which recorded or recomputed artefact
carries it, and which script turns the second into the first.

---

## Abstract and Introduction

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Compression runs re-analysed | 3922 | the three study CSVs, summed | `analysis/scripts/load_all.py` |
| Checkpoints | 25 | 24 vision + DistilBERT | `trainer/checkpoint/CHECKPOINT_MANIFEST.csv` |
| Repair at the operating point | +44.1 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Repair wins | 24 of 24 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Selection at the operating point | +0.52 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Ratio between the two | 85 | `results/paper_numbers.json` (`ratio`) | `make_tables.py` |
| Spread among five criteria, inside window | median 0.78 | `results/a4_frontier_winners.json` | `a4_who_wins_at_the_frontier.py` |
| Ladder winner is not frontier winner | 15 of 25 | `results/paper_numbers.json` (`omp_not_winner`) | `a4_who_wins_at_the_frontier.py` |
| Attention reversal | +14.4 | `compression_transformer/results/attention.csv` | `phase8_attention.py` |
| Residual ratio across budgets | 7.6 to 28.8 | `results/a7_mechanism.json` | `a7_mechanism.py` |
| Identity verified to | 6e-16 | `results/a8_theory.json` | `a8_theory.py` |

## Section 3, Decomposing the layer-wise error

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Theorem 1, exact decomposition | proof in paper | verified numerically below | `a8_theory.py` |
| Corollary 2, tail bound | proof in paper | verified numerically below | `a8_theory.py` |
| Proposition 3, merging | proof in paper | consistent with Table 5 OT rows | `make_tables.py`, `t9_comparison` |
| GRIDE median dimension | 18 | `results/a10_dim_vs_rank.json` | `a10_dim_vs_rank.py` |
| Spectral effective rank median | 216 | `results/a10_dim_vs_rank.json` | `a10_dim_vs_rank.py` |
| Ratio between them | 11.9, up to 54 | `results/a10_dim_vs_rank.json` | `a10_dim_vs_rank.py` |

This last row is the paper answering a question a reviewer is entitled to ask,
namely whether the intrinsic dimension that sets the budgets is being quietly
conflated with the spectral tail that governs the theory. It is not, and the two
differ by more than an order of magnitude.

## Section 4, The two terms measured directly

All rows here are recomputed from checkpoints in double precision with no ridge.
No network is evaluated, so nothing in this section can inherit an artefact of an
accuracy pipeline.

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Layer-budget cells | 154 (22 layers, 8 checkpoints, 7 budgets) | `results/a7_mechanism.json` | `a7_mechanism.py` |
| Maximum relative violation of Eq. 6 | 5.7e-16 | `results/a8_theory.json` | `a8_theory.py` |
| Two routes to `R_fit` agree to | 2.4e-8 | `results/a8_theory.json` | `a8_theory.py` |
| Tail bound violations | 0 of 154 | `results/a8_theory.json` | `a8_theory.py` |
| Bound slack, median and max | 0.30, 0.58 | `results/a8_theory.json` | `a8_theory.py` |
| Predicted vs measured separation | r = +0.78 log-log | `results/fig4_tail.json` | `make_fig_extra.py` |
| Correction / selection ratio | 7.63 rising to 28.82 | `results/paper_numbers.json` | `a7_mechanism.py` |
| Same ratio in effective-rank units | 8.9 to 85.1 | `results/a7_mechanism.json` | `a7_mechanism.py` |

Table 1 in the paper is `tables/t3_mechanism.tex`, built from `a7_mechanism.json`.
Figure 2 is `figures/fig4_tail.pdf`, built from `a7_mechanism.json` and
`a8_theory.json` jointly.

## Section 5, The repair dominates the criterion

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Fully connected runs | 2760 | `compression_mlp/results/compression_results.csv` | `run_autonomous_compression.py` |
| Convolutional runs | 677 | `compression_cnn/hidden_layer_compression/results/compression_results_cnn.csv` | `run_cnn_compression.py` |
| Transformer runs | 485 | `compression_transformer/results/*.csv` | `phase3` through `phase9` |
| Repair vs merging | +44.1, 24/24 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Repair vs magnitude | +32.7, 16/16 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Selection, target-aware vs blind | +0.52, t = 3.62 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Selection by band | +7.48 down to +0.59 | `results/a3_component_regime.json` | `a3_component_regime.py` |
| Repair against merging by band | +25.54 down to +24.14 | `results/a3_component_regime.json` | `a3_component_regime.py` |
| Crossover by retained fraction | +7.02, +2.26, +0.34 | binned from the three CSVs | `make_fig_extra.py`, `fig3` |
| Per-family selection means | +0.51, +0.62, +0.35 | `results/paper_numbers.json` (`window_by_arch`) | `a6_at_the_frontier.py` |
| Within-band standard deviation | 1.05 | `results/paper_numbers.json` | `a6_at_the_frontier.py` |
| Spread inside vs outside window | 0.78 vs 15.15 | `results/a4_frontier_winners.json` | `a4_who_wins_at_the_frontier.py` |
| What the ladder winner gives away | 0.12 mean, 0.44 max | `results/paper_numbers.json` | `a4_who_wins_at_the_frontier.py` |

### Table 2, the master per-checkpoint table

Built by `make_tables.py`, function `t7_master`, from `load_all.py` over all
three study CSVs. The MAC column comes from `mac_reduction_pct` as recorded for
convolutional models, and is computed exactly from the retained widths for fully
connected ones by `load_all.mlp_macs`. The Transformer MAC entry is dashed
because the runs did not record it and it depends on sequence length.

### Table 5, comparison with published methods

Built by `make_tables.py`, function `t9_comparison`. The ThiNet rows come from
`ablations/results/e3_thinet.csv`, written by `e3_thinet.py`, which
implements ThiNet Algorithm 1 verbatim. Everything else comes from the three
study CSVs at each checkpoint's own frontier multiplier.

| Claim | Value | Source |
| --- | --- | --- |
| ThiNet loses | 41.3 points | `e3_thinet.csv` |
| Full re-solve recovers | 36.3 of those | `e3_thinet.csv` |
| Remainder attributable to selection and target | 5.0 | difference of the two above |
| OT with repair gives up to selection with repair | 2.7 | `results/thinet_comparison.json` |

## Section 6, When the criterion matters again

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Attention, target-aware at 18 of 72 heads | +14.40 | `compression_transformer/results/attention.csv` | `phase8_attention.py` |
| Attention, at 39 of 72 heads | +0.20 | `compression_transformer/results/attention.csv` | `phase8_attention.py` |
| Feed-forward, same contrast | +0.08 | `compression_transformer/results/sweep.csv` | `phase3_sweep.py` |
| Tail at `M/N` = 0.05 | 0.698 vs 0.544 | `results/a9_spectra.json` | `a9_spectra.py` |
| Tail at `M/N` = 0.25 | 0.443 vs 0.339 | `results/a9_spectra.json` | `a9_spectra.py` |
| Tail at `M/N` = 0.75 | 0.144 vs 0.103 | `results/a9_spectra.json` | `a9_spectra.py` |
| Ratio range | 1.28 to 1.40 | `results/a9_spectra.json` | `a9_spectra.py` |

The paper reports the spectral gap as modest and says explicitly that the
spectrum alone does not explain a fourteen-point reversal, attributing the rest
to granularity. That is a deliberate weakening of the claim to what the evidence
supports.

## Section 7, What this leaves standing

| Claim | Value | Source artefact | Producing script |
| --- | --- | --- | --- |
| Refinements worth at the frontier | -0.00, t = -0.05 | `results/a6_at_the_frontier.json` | `a6_at_the_frontier.py` |
| Sequential absorption costs | 0.44 points | `compression_transformer/results/ablation.csv` | `phase5_ablation.py` |
| Drift energy outside the retained span | 86 to 92 percent | `compression_transformer/logs/`, drift instrumentation | `phase5b_drift.py` |
| Residual-corrected target recovers | 0.58 points | `compression_transformer/results/ablation.csv` | `phase5_ablation.py` |

### The Gauss-Newton metric claim needs care

The paper states that a Gauss-Newton weighting of the residual "helps or hurts by
about a point depending only on whether a repair follows". That sentence
compresses evidence drawn from three different experimental settings, and the
supporting numbers are weaker and less consistent than the sentence implies. The
full picture is set out here so a reader can judge it directly.

| Setting | Measured effect | n | Source | Script |
| --- | --- | --- | --- | --- |
| Hidden layer, refit follows, convolutional | -0.27 | 4 paired cells | `results_metric_experiment/compression_results_cnn.csv` | `metric_omp.py` |
| Hidden layer, refit follows, targeted ablation | -0.089 overall, and positive at 2 of 5 multipliers (+0.052, +0.058) | 20 paired cells | `ablations/results/e4_metric_norefit.csv` | `e4_metric_norefit.py` |
| Hidden layer, no refit, targeted ablation | metric wins 8 of 20 | 20 paired cells | `ablations/results/e4_metric_norefit.csv` | `e4_metric_norefit.py` |
| Input representation, whole network frozen | `N1_jacobian_linear` wins 45 of 70 against `opt_linear`, mean gap 0.05 | 70 cells | `compression_input_space/results/input_compression_results.csv` | `novel_strategies/innovations.py` |

Three things follow that the paper's single sentence does not convey.

The "about a point" figure originates in the fully connected study, recorded at
`01_experimental_setup/project_progress_log.md` line 369. The convolutional measurements are an order of magnitude
smaller, at -0.09 to -0.27, and the effect is not present at every budget.

The "wins 45 of 70" figure comes from the **input-representation** study, where
the entire network downstream of the compressed representation is frozen. That is
a different experimental setting from the hidden-layer pruning the rest of the
paper concerns, even though the mechanism being argued about is the same.

The experiment built specifically to test this on hidden layers, `e4`, did not
confirm it. `01_experimental_setup/project_progress_log.md` line 787 records the outcome verbatim: the frozen arm
is "indistinguishable from zero rather than positive, so the double-counting
explanation the project has carried for two months looks wrong on convolutions".
Two caveats are recorded alongside it, namely that without a refit the models sit
far outside the deployable window at every budget, and that the metric is a poor
estimate on early layers.

The direction of the effect is consistent with the paper's argument in every
setting measured. The magnitude, the claim that it holds at every budget, and the
transfer from the input-representation setting to the hidden-layer setting are
not established by the artefacts in this supplement. A reader should treat this
as the weakest empirical claim in the paper. It is also the one claim whose
supporting evidence sits outside the three studies the main results are built
from.

## Claims the paper explicitly reports as weak

These are recorded here because a supplement that only lists confirmations is not
doing its job.

| Claim | Status as stated in the paper |
| --- | --- |
| Spectral gap between attention and feed-forward | 1.31x only, called modest, granularity credited as the second factor |
| Failure analysis of sequential correction | one-sided, no case where it demonstrably helps |
| Corollary 2 tightness | about threefold loose, used for ordering not magnitude |
| Transformer sample size | one checkpoint, n = 4 in the per-family window statistic |
| Checkpoint balance | 16 of 25 convolutional, so pooled statistics are weighted accordingly |
| GRIDE versus spectral rank | differ by 11.9x, the theory uses only the latter |

## Claims this supplement finds weaker than the paper states

One entry, documented in full in the Gauss-Newton subsection above: the
metric-versus-repair sentence in Section 7 draws its two numbers from different
experimental settings, its magnitude is an order of magnitude larger than the
convolutional measurements support, and the purpose-built ablation `e4` did not
reproduce the frozen-network result on hidden layers. This is recorded here
rather than smoothed over, because a supplement that only confirms is not
performing its function.

Nothing else in the paper was found to overstate its evidence. Every other number
checked in this document reconciles with the artefact it is drawn from.

## Anything referenced in the paper but absent from the supplement

None. Every artefact cited above is present in either `06_results_csv/`,
`10_supporting_analysis/analysis_json/`, or the code supplement. Trained weights are
the one deliberate omission and are covered by
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv`.
