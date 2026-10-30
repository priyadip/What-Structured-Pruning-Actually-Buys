# CSV Index

Thirty-one result files, 10403 data rows. Every one was written by an experiment
driver at the time the experiment ran. None was produced by the analysis layer,
and none has been edited for this supplement.

## How the paper's headline count is assembled

The paper says it re-analyses 3922 compression runs. That number is exactly:

| Source | Rows | In the 3922 |
| --- | --- | --- |
| `01 MLP study/compression_results.csv` | 2760 | yes |
| `02 CNN study/compression_results_cnn.csv` | 677 | yes |
| `03 Transformer study/sweep.csv` | 108 | yes |
| `03 Transformer study/baselines.csv` | 112 | yes |
| `03 Transformer study/alloc.csv` | 116 | yes |
| `03 Transformer study/ablation.csv` | 48 | yes |
| `03 Transformer study/perlayer.csv` | 48 | yes |
| `03 Transformer study/stability.csv` | 44 | yes |
| `03 Transformer study/hyper.csv` | 9 | yes |
| **Total** | **3922** | |

The loader that performs this union is
`analysis/scripts/load_all.py`. Files not in the
table above are used for specific claims but are deliberately outside the pooled
count, so that no run is counted twice.

## 01 MLP study

| File | Rows | Role |
| --- | --- | --- |
| `compression_results.csv` | 2760 | the fully connected sweep; 8 checkpoints, 14 operators |

Written by `compression_mlp/run_autonomous_compression.py`. Carries `gride_dims`,
`widths`, `orig_widths` per run, which is what allows the analysis to pair
contrasts at identical widths and to recompute MAC reduction exactly.

Note the schema quirk documented in `../01_experimental_setup/README.md`:
`model_name` here has no dataset prefix, so it must be qualified before use.

## 02 CNN study

| File | Rows | Role |
| --- | --- | --- |
| `compression_results_cnn.csv` | 677 | the convolutional sweep; 16 checkpoints |
| `compression_results_cnn_metric_experiment.csv` | 34 | Gauss-Newton metric with a refit following |

The second file supports one sentence in Section 7 and is small. Its paired
comparison is `N1_metric_omp` and `N2_metric_omp_ls` against `S4_omp_ls`, four
paired cells each, mean -0.27 and -0.26. See the Gauss-Newton subsection of
`../CLAIM_TO_EVIDENCE.md`, which explains why this number does not support the
paper's stated magnitude.

## 03 Transformer study

| File | Rows | Role |
| --- | --- | --- |
| `sweep.csv` | 108 | the main budget ladder |
| `baselines.csv` | 112 | published operators at matched budget |
| `alloc.csv` | 116 | per-layer budget allocation |
| `attention.csv` | 100 | **Section 6**, the predicted reversal on attention heads |
| `ablation.csv` | 48 | sequential absorption, residual-corrected target |
| `perlayer.csv` | 48 | each block compressed alone |
| `stability.csv` | 44 | seeds 0, 1, 2 |
| `hyper.csv` | 9 | calibration size and ridge sensitivity |
| `artifact.csv` | 2 | the exported best model |
| `baselines_thinet_densetgt_buggy.csv` | 7 | **quarantined, do not use** |

`attention.csv` carries the +14.40 figure that is the paper's falsifiable
prediction confirmed, and it is deliberately excluded from the 3922 so that the
pooled statistics and the Section 6 result stay independent.

`baselines_thinet_densetgt_buggy.csv` is retained rather than deleted because
honesty about a corrected error is worth more than a tidy folder. The ThiNet
dense-target variant scored its greedy step with `-2b` instead of
`-2(tot - b_dense)`, which produced a plausible but wrong mean of 35.97. The rows
were quarantined when the bug was found. Nothing in the paper reads this file.

## 04 Ablations

| File | Rows | Role |
| --- | --- | --- |
| `e1_seed_variance.csv` | 48 | seed noise, qualifying the window claim |
| `e1b_robust_frontier.csv` | 16 | frontier at the 2 sigma robust threshold |
| `e2_calibration.csv` | 8 | calibration size sensitivity |
| `e3_thinet.csv` | 96 | **Table 5**, ThiNet reimplemented and decomposed |
| `e4_metric_norefit.csv` | 80 | metric with and without a following refit |
| `e5_joint_budget.csv` and variants | 131, 25, 131 | joint search over per-layer budgets |
| `e6_id_vs_epoch.csv` | 6 | intrinsic dimension against training epoch |
| `e7_walltime.csv` | 128 | wall-clock against MAC reduction |
| `e*_driver_log.csv` | various | raw driver logs behind the above |

`e3_thinet.csv` is the source of the paper's cleanest single decomposition:
ThiNet loses 41.3 points, a full re-solve on its own greedy selection recovers
36.3, and the residual 5.0 is everything its criterion and target contribute
together.

`e7_walltime.csv` supports a claim the paper makes only implicitly, that MAC
reduction and wall-clock speedup are not the same thing.

## 05 Superseded

| File | Rows | Role |
| --- | --- | --- |
| `exploratory_runs.csv` | 351 | early exploratory sweep, superseded |
| `partial_runs_missing_phase2.csv` | 1872 | incomplete runs, missing a pipeline phase |

**Nothing in the paper reads these.** They are included so the record is complete
and so a reviewer can see what was discarded and why. `partial_runs_missing_phase2.csv`
in particular is large enough that including it in the pooled count would have
inflated the headline number by nearly half, which is exactly the sort of thing a
supplement should make checkable.

## 06 Input-space study

| File | Rows | Role |
| --- | --- | --- |
| `input_compression_results.csv` | 3174 | the frozen-network half of the Section 7 metric claim |

This is a **different experimental setting** from the rest of the paper: it
compresses the input representation with the whole downstream network frozen,
rather than pruning a hidden layer. It is included because the paper's "wins 45
of 70 cells" figure comes from here and from nowhere else, a fact that is easy to
miss and that materially affects how the sentence should be read. See
`../CLAIM_TO_EVIDENCE.md`.
