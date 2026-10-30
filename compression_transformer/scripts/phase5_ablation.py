#!/usr/bin/env python3
"""Phase 5: what each piece is worth, and the two questions the port raises.

Everything here is a paired contrast at identical width, so nothing is
confounded by how many units survived. Four of the six axes are inherited from
the paper; two are new because a Transformer poses them and a convolution does
not.

  A. correction     omp vs omp_ls, magnitude vs magnitude_ls, ot vs ot_then_ls
  B. selection      omp_ls vs span_ls vs magnitude_ls vs random_ls
  C. absorption     sequential on vs off
  D. intercept      omp_ls vs omp_ls_noint            (RESEARCH R3, GELU)
  E. target         dense vs residual                 (RESEARCH R2, new)

The cloud question of RESEARCH R4 is not an axis here: changing the cloud
changes the widths, so it is compared as a whole frontier in phase3_sweep.py.
"""

from __future__ import annotations

import argparse

import core as C
import harness as H

OUT = C.RESULTS / "ablation.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="16,12,8,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--axes", default="A,B,C,D,E")
    a = ap.parse_args()
    axes = set(a.axes.split(","))
    alphas = [float(x) for x in a.alphas.split(",")]

    s = H.Session()
    for alpha in alphas:
        H.log(f"alpha {alpha}  widths {s.widths(alpha)}")
        if "A" in axes or "B" in axes:
            for st in ("omp_ls", "omp", "span_ls", "magnitude_ls", "magnitude",
                       "random_ls", "random", "ot_then_ls", "ot_intrafusion"):
                s.run(st, alpha, out=OUT, tag="AB", seed=a.seed)
        if "C" in axes:
            s.run("omp_ls", alpha, out=OUT, tag="C_noabsorb", seed=a.seed,
                  sequential=False, note="dense input at every layer")
        if "D" in axes:
            s.run("omp_ls_noint", alpha, out=OUT, tag="D_nointercept", seed=a.seed)
        if "E" in axes:
            s.run("omp_ls", alpha, out=OUT, tag="E_residual", seed=a.seed,
                  target="residual", note="target += a_dense - a_comp")
    # Axis F is deliberately not run here. Changing the cloud changes `d_l` and
    # therefore the widths, so a fixed alpha no longer holds the budget fixed
    # and an accuracy difference would confound operator with size. The three
    # clouds are compared as whole frontiers instead, by running the ladder once
    # per cloud in phase3_sweep.py and reading accuracy against parameters.


if __name__ == "__main__":
    main()
