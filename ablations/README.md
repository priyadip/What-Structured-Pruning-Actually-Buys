# `to_do_experiment/` — the follow-up runs from `TO_DO_EXPERIMENTS.md`

The experiments the paper named as missing, executed. Each script re-uses the
**production** compression code from `compression_cnn/` rather than
reimplementing it, so an experiment cannot accidentally measure a different
method than the paper reports; what varies is only the thing under test.

```
python3 e1_seed_variance.py   --seeds 0 1 2        # error bars on Table 1
python3 e2_calibration.py                          # calibration-size sensitivity
python3 e3_thinet.py                               # faithful ThiNet control
python3 e3_thinet.py --selftest                    # batched kernel == production
python3 e4_metric_norefit.py                       # the decisive metric experiment
python3 e5_joint_budget.py                         # joint (channel, rank) search
python3 e7_walltime.py --iters 100                 # do MAC cuts become speedups?
```

Results land in `results/*.csv`; findings are written up in
`../technical_supplement/07_ablations/RESULTS.md`.

| file | role |
|---|---|
| `harness.py` | shared: builds a production `Session`, reads the saved frontier specs, redirects the driver's CSV so a side experiment can never contaminate the paper's sweep |
| `e*_*.py` | one experiment each, self-describing in its module docstring |
| `../technical_supplement/07_ablations/RESULTS.md` | what was found, and what changed in the paper because of it |

## Design rules these scripts follow

**Never write into the paper's CSVs.** `run_cnn_compression` appends to a
module-level `CSV_PATH`; `harness.redirect_csv` points it at
`to_do_experiment/results/` before any Session is constructed.

**Read the configuration under test from disk.** The multiplier and strategy
each experiment re-runs come from the saved `*_frontier.arch.json`, so the
configuration being measured is provably the one in Table 1 rather than one
retyped from a summary.

**Any new kernel is checked against the one it replaces.** `e3_thinet.py`
batches the greedy-elimination scan for a ~5× speedup; `--selftest` asserts it
returns index-for-index what `common/linalg.forward_backward_select` returns, on
three shapes including `gs=1`. A faster kernel that computes something slightly
different would silently invalidate the whole experiment.
