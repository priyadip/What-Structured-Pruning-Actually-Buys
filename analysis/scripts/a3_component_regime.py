#!/usr/bin/env python3
"""A3. Which components keep their value inside the deployable window?

A2 showed the selection gap collapses from +7.7 points at deep compression to
+0.57 inside the +/-2.5 window, with no architecture dependence left once the
regime is held fixed. The obvious next question is whether every component
behaves that way, in which case the finding is "nothing matters near dense" and
is trivial, or whether the correction survives, in which case the components
separate into two classes and that separation is the contribution.

Same stratification as A2: bands are set by the distance from dense of the
*better* member of each pair, which is a property of the cell and not of either
operator, so the split does not select on the outcome.
"""

from __future__ import annotations

import json
from pathlib import Path

from load_all import load_all, summarise

OUT = Path(__file__).resolve().parents[1] / "results"
BANDS = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 1e9)]

# (label, better, worse, what it isolates)
CONTRASTS = [
    ("correction | magnitude selection", "magnitude_ls", "magnitude",
     "refit vs none, selection held fixed"),
    ("correction | merging", "ot_then_ls", "ot_intrafusion",
     "refit vs none, producer held fixed"),
    ("selection: OMP vs span", "omp_ls", "span_ls",
     "target-aware vs target-blind, refit held fixed"),
    ("selection: OMP vs magnitude", "omp_ls", "magnitude_ls",
     "data-aware vs weight-norm, refit held fixed"),
    ("selection: OMP vs merging", "omp_ls", "ot_then_ls",
     "select vs merge, refit held fixed"),
    ("refinement: fwd-bwd vs OMP", "fwdbwd_ls", "omp_ls",
     "backward elimination on top of greedy"),
]


def cells(runs, a, b):
    out = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == a}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == b}
        for k in set(ia) & set(ib):
            ra, rb = ia[k], ib[k]
            out.append({"arch": arch, "gap": ra.acc - rb.acc,
                        "best_drop": ra.dense - max(ra.acc, rb.acc)})
    return out


def main():
    runs = load_all()
    res = {}
    hdr = (f"{'contrast':36s} {'<2.5':>9s} {'2.5-5':>9s} {'5-10':>9s} "
           f"{'>10':>9s}   {'n(window)':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for label, a, b, _ in CONTRASTS:
        cs = cells(runs, a, b)
        if not cs:
            continue
        row, cells_by_band = [], {}
        for lo, hi in BANDS:
            s = summarise([c["gap"] for c in cs if lo <= c["best_drop"] < hi])
            cells_by_band[f"{lo}-{hi}"] = s
            row.append(f"{s['mean']:+9.2f}" if s else f"{'-':>9s}")
        nwin = cells_by_band["0.0-2.5"]["n"] if cells_by_band["0.0-2.5"] else 0
        print(f"{label:36s} {' '.join(row)}   {nwin:9d}")
        res[label] = {"a": a, "b": b, "bands": cells_by_band,
                      "by_arch_in_window": {}}
        for arch in ("mlp", "cnn", "transformer"):
            s = summarise([c["gap"] for c in cs
                           if c["arch"] == arch and c["best_drop"] < 2.5])
            if s:
                res[label]["by_arch_in_window"][arch] = s

    print()
    print("Inside the +/-2.5 window, per architecture "
          "(mean, and cells where the first operator wins):")
    print(f"{'contrast':36s} {'mlp':>18s} {'cnn':>18s} {'transformer':>18s}")
    print("-" * 94)
    for label in res:
        cellsr = []
        for arch in ("mlp", "cnn", "transformer"):
            s = res[label]["by_arch_in_window"].get(arch)
            cellsr.append(f"{s['mean']:+7.2f} ({s['pos']:2d}/{s['n']:2d})"
                          if s else f"{'-':>18s}")
        print(f"{label:36s} {cellsr[0]:>18s} {cellsr[1]:>18s} {cellsr[2]:>18s}")

    (OUT / "a3_component_regime.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a3_component_regime.json'}")


if __name__ == "__main__":
    main()
