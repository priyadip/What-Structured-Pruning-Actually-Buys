#!/usr/bin/env python3
"""Phase 7: per-layer sensitivity, and where the accuracy actually goes.

A single global `alpha` yields one frontier for the model, so "alpha* for each
of the six layers" only has an unambiguous meaning if each layer is compressed
on its own. That is what this does: six independent ladders, every other layer
left dense, which reads off how much each block can give up in isolation and
which block breaks first.
"""

from __future__ import annotations

import argparse
import json

import core as C
import harness as H

OUT = C.RESULTS / "perlayer.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1,2,4,6,8,12,16,24")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    s = H.Session()
    frontier = {}
    for l in range(C.N_LAYERS):
        H.log(f"layer {l} alone (d = {s.dims()[l]:.2f})")
        hits = []
        for alpha in alphas:
            r = s.run("omp_ls", alpha, out=OUT, tag=f"solo_L{l}", seed=a.seed,
                      layers=[l])
            if r["hit"]:
                hits.append(r)
        if hits:
            best = min(hits, key=lambda r: float(r["alpha"]))
            frontier[str(l)] = {"alpha_star": float(best["alpha"]),
                                "m": json.loads(best["widths"])[l],
                                "acc": best["acc"], "delta": best["delta"],
                                "red_ffn_pct": best["red_ffn_pct"]}
            H.log(f"  layer {l}: alpha* = {best['alpha']}, "
                  f"M = {frontier[str(l)]['m']}, acc {best['acc']}")
        else:
            frontier[str(l)] = None
            H.log(f"  layer {l}: no alpha on the ladder stayed inside the window")
    (C.RESULTS / "perlayer_frontier.json").write_text(json.dumps(frontier, indent=2))


if __name__ == "__main__":
    main()
