# Analysis

A re-analysis of every compression experiment in this repository, asking what
scientific claim the accumulated evidence actually supports.

## The finding

Every study here ranks compression operators by averaging accuracy over a budget
ladder. Those ladders spend most of their rungs on destroyed networks. Read at
each checkpoint's own frontier instead, across 21 checkpoints spanning MLP, CNN
and Transformer:

| contrast | mean | wins |
|---|---|---|
| closed-form consumer refit, given merging | **+41.37** | 19/19 |
| closed-form consumer refit, given magnitude | **+32.74** | 16/16 |
| target-aware rather than target-blind selection | **+0.31** | 15/20 |
| backward elimination on top of OMP | +0.08 | 4/6 |

A ratio of 132. Five refitted operators the literature ranks confidently have a
median spread of 0.49 points there, and the ladder-average winner is not the
frontier winner on 11 of 21 checkpoints.

The mechanism is measured directly as two residuals, with no accuracy, no
acceptance window and no budget search: their ratio rises monotonically from 7.6
to 28.8 across 154 layer-budget cells. The prediction that follows, that
selection recovers where there is no spectral tail, is confirmed on 12 attention
heads at +14.40.

## Layout

| path | |
|---|---|
| `../reproduce.sh` | reruns every analysis and regenerates tables and figures |
| `scripts/load_all.py` | one loader for three CSV schemas, 3922 records |
| `scripts/a1`-`a7` | the analyses, each writing a JSON to `results/` |
| `results/`, `tables/`, `figures/` | generated, never hand-edited |

## Reproducing

    ./reproduce.sh tables

Runs on CPU in about a minute, reading only the recorded CSVs.
`scripts/a7_mechanism.py` additionally needs the MLP checkpoints in
`../trainer/checkpoint/` and a GPU is helpful but not required.

## What was rejected

Recorded with the evidence: the `log(mean N/d)`
compressibility law (does not extrapolate, confounded with dense accuracy, LOO
straddles zero), a knee at the intrinsic dimension (no knee on any family), and
the Zador-derived per-layer allocation (loses to a uniform multiplier on all
three).

## What is not here

The manuscript source is not included in this artifact. The tables in `tables/`
and the figures in `figures/` are the exact fragments the paper inputs, and
`./reproduce.sh tables` regenerates them from the recorded runs.
