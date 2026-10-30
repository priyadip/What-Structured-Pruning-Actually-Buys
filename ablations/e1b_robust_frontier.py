#!/usr/bin/env python3
"""E1b -- what a seed-robust frontier would cost.

E1 shows the compressed accuracy at a *fixed* configuration is stable to about
0.1 points, but that roughly one run in six falls outside the +/-2.5 window
anyway. That is not instability in the method; it is what a frontier *is*. The
frontier is defined as the tightest budget still inside the window, so it is
selected to sit exactly at the boundary, and any perturbation has a coin-flip
chance of pushing it out.

The constructive response is to define the frontier with a margin. This script
asks what that costs, using the production sweep's own ladder:

    frontier        smallest budget with |drop| <= 2.5
    robust frontier smallest budget with |drop| <= 2.5 - k * sd(seed)

with `sd(seed)` measured per model by E1. Reporting the second alongside the
first turns "16/16 inside the window" from a statement that happens to be true
of one draw into one that survives re-running.

    python3 e1b_robust_frontier.py [--k 2.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import harness as H

LADDER = H.HLC / "results" / "compression_results_cnn.csv"
E1 = H.RESULTS / "e1_seed_variance.csv"
OUT = H.RESULTS / "e1b_robust_frontier.csv"
FIELDS = ["model", "sd_seed", "frontier_mult", "frontier_drop", "frontier_param_cut",
          "robust_mult", "robust_drop", "robust_param_cut", "cost_pts"]


def ladder_rows():
    """(model -> [(multiplier, best accuracy, param cut, dense)]) from the
    production sweep, best strategy at each budget."""
    best = defaultdict(dict)
    dense = {}
    with LADDER.open() as fh:
        for r in csv.DictReader(fh):
            try:
                m = float(r["multiplier"])
                acc = float(r["compressed_acc"])
            except ValueError:
                continue                       # per-layer allocation rows
            if acc != acc:
                continue
            k = r["model_name"]
            dense[k] = float(r["dense_acc"])
            cur = best[k].get(m)
            if cur is None or acc > cur[0]:
                best[k][m] = (acc, float(r["param_reduction_pct"]))
    return {k: sorted((m, a, p) for m, (a, p) in v.items()) for k, v in best.items()}, dense


def seed_sd():
    if not E1.exists():
        return {}
    by = defaultdict(list)
    with E1.open() as fh:
        for r in csv.DictReader(fh):
            by[r["model"]].append(float(r["acc"]))
    return {m: (st.stdev(v) if len(v) > 1 else 0.0) for m, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=2.0,
                    help="margin in units of the measured per-model seed s.d.")
    a = ap.parse_args()

    lad, dense = ladder_rows()
    sds = seed_sd()
    if not sds:
        raise SystemExit("run e1_seed_variance.py first -- no seed s.d. available")

    rows = []
    print(f"margin = {a.k} x the per-model seed s.d.\n")
    print(f"{'model':24s} {'sd':>5s} {'frontier':>22s} {'robust frontier':>22s} "
          f"{'cost':>7s}")
    for m in sorted(sds):
        if m not in lad:
            continue
        s = sds[m]
        d = dense[m]
        tight = 2.5 - a.k * s
        fr = rb = None
        for mult, acc, cut in lad[m]:
            drop = acc - d
            if fr is None and abs(drop) <= 2.5:
                fr = (mult, drop, cut)
            if rb is None and abs(drop) <= tight:
                rb = (mult, drop, cut)
        if fr is None:
            continue
        if rb is None:
            print(f"{m:24s} {s:5.2f} {'x%-5.2f -%5.1f%%' % (fr[0], fr[2]):>22s} "
                  f"{'none on the ladder':>22s} {'--':>7s}")
            rows.append({"model": m, "sd_seed": round(s, 4),
                         "frontier_mult": fr[0], "frontier_drop": round(fr[1], 3),
                         "frontier_param_cut": fr[2], "robust_mult": "",
                         "robust_drop": "", "robust_param_cut": "", "cost_pts": ""})
            continue
        cost = fr[2] - rb[2]
        print(f"{m:24s} {s:5.2f} {'x%-5.2f -%5.1f%%' % (fr[0], fr[2]):>22s} "
              f"{'x%-5.2f -%5.1f%%' % (rb[0], rb[2]):>22s} {cost:6.1f}pp")
        rows.append({"model": m, "sd_seed": round(s, 4),
                     "frontier_mult": fr[0], "frontier_drop": round(fr[1], 3),
                     "frontier_param_cut": fr[2], "robust_mult": rb[0],
                     "robust_drop": round(rb[1], 3), "robust_param_cut": rb[2],
                     "cost_pts": round(cost, 3)})

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)

    costs = [r["cost_pts"] for r in rows if r["cost_pts"] != ""]
    if costs:
        print(f"\nmedian cost of a {a.k}-sigma margin: {st.median(costs):.1f} "
              f"percentage points of parameter reduction "
              f"(range {min(costs):.1f}-{max(costs):.1f})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
