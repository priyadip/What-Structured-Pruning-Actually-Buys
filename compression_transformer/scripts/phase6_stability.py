#!/usr/bin/env python3
"""Phase 6: how much of the frontier is real.

The frontier is the smallest budget still inside the window, so it is a boundary
point and is systematically optimistic: any run that landed just inside by luck
is kept, any that landed just outside is not. The paper's answer is to report a
second frontier at `|delta| <= 2.5 - 2*sigma`, where sigma is the spread across
calibration seeds at the same budget. The same treatment is applied here.

The calibration-size curve is the other half: if the method underperformed, it
should be shown whether that is the operator or the data it was given.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import core as C
import harness as H

OUT = C.RESULTS / "stability.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="8,10,12,12.125,13,14,16,20")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--calibs", default="512,1024,2048,4096")
    ap.add_argument("--what", default="seeds,calib")
    ap.add_argument("--target", default="dense")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    what = set(a.what.split(","))

    if "seeds" in what:
        s = H.Session()
        # dims are re-measured per seed, so the width itself moves with the seed;
        # that variation is part of what the frontier inherits and is kept in.
        tag = "seed" if a.target == "dense" else f"seed_{a.target}"
        for alpha in alphas:
            for sd in seeds:
                s.run("omp_ls", alpha, out=OUT, tag=tag, seed=sd,
                      target=a.target)
        del s

    if "calib" in what:
        for n in [int(x) for x in a.calibs.split(",")]:
            rows = min(65536, n * 32)
            s = H.Session(calib=n, rows_budget=rows)
            for alpha in (12.125, 16):
                try:
                    s.run("omp_ls", alpha, out=OUT, tag="calib", seed=0,
                          note=f"calib={n}")
                except RuntimeError as exc:
                    H.log(f"  calib={n} alpha={alpha} refused: {exc}")
            del s


if __name__ == "__main__":
    main()
