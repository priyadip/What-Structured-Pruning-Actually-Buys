#!/usr/bin/env python3
"""Phase 3: the ladder and the frontier for the method itself.

A single global `alpha` sets every layer's width through `M_l = round(alpha d_l)`,
so the sweep moves all six layers together and the reported frontier is the
smallest budget still inside the +/- 2.5 window.
"""

from __future__ import annotations

import argparse
import json

import core as C
import harness as H

OUT = C.RESULTS / "sweep.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="omp_ls")
    ap.add_argument("--calib", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cloud", default="neuron")
    ap.add_argument("--target", default="dense")
    ap.add_argument("--alphas", default="")
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--no-bisect", action="store_true")
    a = ap.parse_args()

    s = H.Session(calib=a.calib, rows_budget=a.rows, cloud=a.cloud)
    if a.alphas:
        ladder = [float(x) for x in a.alphas.split(",")]
    else:
        ladder = C.multiplier_ladder(s.dims(a.cloud), [C.HIDDEN_DIM] * C.N_LAYERS)
    H.log(f"dims {[round(d, 2) for d in s.dims(a.cloud)]}")
    H.log(f"ladder {ladder}")

    rows = []
    for alpha in ladder:
        rows.append(s.run(a.strategy, alpha, out=OUT, tag=a.tag, seed=a.seed,
                          cloud=a.cloud, target=a.target))

    hits = [r for r in rows if r["hit"]]
    if not hits:
        H.log("no configuration on the ladder reached the window")
        return
    first = min(hits, key=lambda r: float(r["alpha"]))
    below = [r for r in rows if float(r["alpha"]) < float(first["alpha"])]
    H.log(f"first hit on the ladder: alpha={first['alpha']} "
          f"acc={first['acc']} red_total={first['red_total_pct']}%")

    if not a.no_bisect and below:
        lo = max(float(r["alpha"]) for r in below)
        best = H.bisect(s, a.strategy, lo, float(first["alpha"]), out=OUT,
                        tag=a.tag + "_bisect", seed=a.seed, cloud=a.cloud,
                        target=a.target)
        if best:
            first = best
    H.log(f"FRONTIER alpha* = {first['alpha']}  widths {first['widths']}  "
          f"acc {first['acc']} ({first['delta']:+})  "
          f"red total {first['red_total_pct']}%  encoder {first['red_encoder_pct']}%  "
          f"ffn {first['red_ffn_pct']}%")
    (C.RESULTS / f"frontier_{a.strategy}_{a.cloud}_s{a.seed}.json").write_text(
        json.dumps(first, indent=2))


if __name__ == "__main__":
    main()
