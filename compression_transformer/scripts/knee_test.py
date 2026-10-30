#!/usr/bin/env python3
"""Is there a feature at `alpha = 1`?

The coarse ladder suggested one, but its log spacing is uneven exactly there, so
a second difference computed on it partly reports the sampling pattern. This
reads the evenly-sampled fine scan instead and asks the question three ways:

  1. the slope of accuracy in `log alpha`, left and right of 1, on the fine grid
  2. a two-segment fit against a one-segment fit over the same window, compared
     by residual sum of squares, with the breakpoint free
  3. whether the free breakpoint lands at 1 across three calibration seeds

A knee at the intrinsic dimension would show as a break located at `alpha = 1`
in all three seeds. A smooth concave curve, which is what the paper reports for
24 checkpoints, shows as a breakpoint that wanders with the seed and a
two-segment fit that barely beats one segment.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0, sum((y - my) ** 2 for y in ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, rss


def main():
    rows = [r for r in csv.DictReader((RESULTS / "sweep.csv").open())
            if r["tag"] == "knee"]
    if not rows:
        print("no fine-scan rows yet")
        return
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[int(r["seed"])][float(r["alpha"])] = float(r["acc"])

    out = {"seeds": {}}
    for sd, curve in sorted(by_seed.items()):
        xs = sorted(curve)
        if len(xs) < 9:                      # a partial scan cannot be fitted
            print(f"seed {sd}: only {len(xs)} points, skipping")
            continue
        lx = [math.log(x) for x in xs]
        ys = [curve[x] for x in xs]

        _, _, rss1 = fit_line(lx, ys)
        best = None
        for k in range(3, len(xs) - 3):                # breakpoint at xs[k]
            _, b1, r1 = fit_line(lx[:k + 1], ys[:k + 1])
            _, b2, r2 = fit_line(lx[k:], ys[k:])
            if best is None or r1 + r2 < best[1]:
                best = (xs[k], r1 + r2, b1, b2)
        brk, rss2, sl_l, sl_r = best

        # Local finite differences on this grid are not interpretable. Adjacent
        # points are 0.05 apart in log alpha and each accuracy carries about
        # 0.19 points of seed noise, so a one-interval slope inherits roughly
        # +/- 4 points per unit log alpha of pure noise, which is the size of the
        # effect being looked for. The comparison that survives that is the
        # average slope over the whole span on each side of alpha = 1.
        li = [i for i, x in enumerate(xs) if x <= 1.0]
        ri = [i for i, x in enumerate(xs) if x >= 1.0]
        avg_l = ((ys[li[-1]] - ys[li[0]]) / (lx[li[-1]] - lx[li[0]])
                 if len(li) > 1 else None)
        avg_r = ((ys[ri[-1]] - ys[ri[0]]) / (lx[ri[-1]] - lx[ri[0]])
                 if len(ri) > 1 else None)
        out["seeds"][str(sd)] = {
            "avg_slope_below_1": round(avg_l, 3) if avg_l else None,
            "avg_slope_above_1": round(avg_r, 3) if avg_r else None,
            "slope_ratio": round(avg_r / avg_l, 3) if avg_l and avg_r else None,
            "n_points": len(xs),
            "breakpoint": brk,
            "slope_left": round(sl_l, 3),
            "slope_right": round(sl_r, 3),
            "rss_one_segment": round(rss1, 3),
            "rss_two_segment": round(rss2, 3),
            "variance_explained_by_break":
                round(1 - rss2 / rss1, 4) if rss1 > 0 else None,
            "curve": [(x, curve[x]) for x in xs],
        }
        print(f"seed {sd}: free breakpoint at alpha = {brk:g} "
              f"(slopes {sl_l:.2f} -> {sl_r:.2f});  "
              f"average slope below/above alpha=1: "
              f"{avg_l:.2f} / {avg_r:.2f}, ratio {avg_r / avg_l:.2f}")

    brks = [v["breakpoint"] for v in out["seeds"].values()]
    ratios = [v["slope_ratio"] for v in out["seeds"].values()
              if v.get("slope_ratio")]
    if ratios:
        out["slope_ratio_across_seeds"] = ratios
        # A knee at the intrinsic dimension means the gain per unit budget drops
        # sharply once each block has d_l units, so the ratio would be well
        # below 1. Near 1 means the curve passes through alpha = 1 unchanged.
        out["knee_at_1_supported"] = bool(max(ratios) < 0.6)
    out["breakpoints"] = brks
    out["breakpoint_at_1_in_all_seeds"] = all(abs(b - 1.0) < 1e-9 for b in brks)
    out["breakpoint_spread"] = [min(brks), max(brks)]
    if len(brks) > 1:
        out["breakpoint_sd"] = round(st.stdev(brks), 4)
    print(f"\nbreakpoints across seeds: {brks}")
    print(f"all at alpha = 1: {out['breakpoint_at_1_in_all_seeds']}")
    (RESULTS / "knee_test.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
