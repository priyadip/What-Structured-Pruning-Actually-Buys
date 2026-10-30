#!/usr/bin/env python3
"""Phase 4: every method at matched width.

All thirteen operators are handed exactly the same `M_l = round(alpha d_l)`, so
whatever separates them is how units are removed and never how many survive. The
budgets are chosen to bracket the method's own frontier, so the comparison is
read where it matters rather than only where everything works or everything has
already collapsed.
"""

from __future__ import annotations

import argparse

import core as C
import harness as H

OUT = C.RESULTS / "baselines.csv"

ORDER = [
    "omp_ls", "fwdbwd_ls", "span_ls", "magnitude_ls", "magnitude_prod_ls",
    "random_ls", "ot_then_ls",
    "omp", "magnitude", "random", "thinet", "thinet_densetgt", "thinet_lsfull",
    "ot_intrafusion", "kmeans_merge",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="24,16,12,8,6,4,2")
    ap.add_argument("--strategies", default=",".join(ORDER))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=65536)
    a = ap.parse_args()

    s = H.Session(calib=a.calib, rows_budget=a.rows)
    alphas = [float(x) for x in a.alphas.split(",")]
    strategies = [x for x in a.strategies.split(",") if x]
    for alpha in alphas:
        H.log(f"alpha {alpha}  widths {s.widths(alpha)}")
        for st in strategies:
            try:
                s.run(st, alpha, out=OUT, tag="matched", seed=a.seed)
            except Exception as exc:                    # a baseline may legitimately fail
                H.log(f"  {st} FAILED at alpha={alpha}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
