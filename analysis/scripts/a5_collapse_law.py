#!/usr/bin/env python3
"""A5. Does the selection gap collapse onto a single curve in alpha?

A2 and A4 established that the value of the selection rule depends on the
operating regime. "Distance from dense" is a fine label for a regime but it is
not a mechanism, because it is measured on the outcome. This tests a mechanistic
candidate that is available *before* any accuracy is measured: the budget
multiplier alpha itself, where every layer receives `M_l = round(alpha d_l)`
units and `d_l` is the GRIDE intrinsic dimension.

The argument for alpha. Selecting M columns of U to reconstruct Y is a column
subset selection problem whose residual is controlled by how much of U's energy
lies outside the span of the chosen columns. If U has effective rank ~d, then
once M exceeds d by a comfortable factor almost every subset of size M spans the
directions that carry the signal, and *which* subset was taken stops mattering.
When M is comparable to d, the choice determines the span. So the natural
scale-free coordinate is M/d, which is exactly alpha.

If the gap collapses onto one curve in alpha across MLP, CNN and Transformer,
that is a statement about column subset selection rather than about any of the
three architectures, and it is the sort of claim that transfers.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from load_all import load_all, summarise, pearson, spearman

OUT = Path(__file__).resolve().parents[1] / "results"


def gap_cells(runs, a="omp_ls", b="span_ls"):
    out = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == a}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == b}
        for k in set(ia) & set(ib):
            ra, rb = ia[k], ib[k]
            if ra.mult <= 0:
                continue
            out.append({"arch": arch, "model": ra.model, "alpha": ra.mult,
                        "gap": ra.acc - rb.acc,
                        "drop": ra.dense - max(ra.acc, rb.acc),
                        "n_over_d": ra.n_over_d, "dense": ra.dense})
    return out


def main():
    runs = load_all()
    res = {}
    cs = gap_cells(runs)
    print(f"cells where both omp_ls and span_ls ran at identical widths: {len(cs)}")

    # ---- gap against alpha, pooled and within architecture -----------------
    la = [math.log(c["alpha"]) for c in cs]
    g = [c["gap"] for c in cs]
    print()
    print("=" * 72)
    print("Selection gap against log alpha")
    print("=" * 72)
    print(f"  pooled        n={len(cs):4d}  pearson={pearson(la, g):+.4f}  "
          f"spearman={spearman(la, g):+.4f}")
    res["gap_vs_log_alpha"] = {"pooled": {"n": len(cs),
                                          "pearson": round(pearson(la, g), 4),
                                          "spearman": round(spearman(la, g), 4)}}
    for arch in ("mlp", "cnn", "transformer"):
        x = [math.log(c["alpha"]) for c in cs if c["arch"] == arch]
        y = [c["gap"] for c in cs if c["arch"] == arch]
        if len(x) >= 4 and len(set(x)) > 1:
            print(f"  {arch:12s}  n={len(x):4d}  pearson={pearson(x, y):+.4f}  "
                  f"spearman={spearman(x, y):+.4f}  "
                  f"alpha {math.exp(min(x)):.2f} to {math.exp(max(x)):.2f}")
            res["gap_vs_log_alpha"][arch] = {
                "n": len(x), "pearson": round(pearson(x, y), 4),
                "spearman": round(spearman(x, y), 4),
                "alpha_lo": round(math.exp(min(x)), 3),
                "alpha_hi": round(math.exp(max(x)), 3)}

    # ---- the collapse, binned in alpha -------------------------------------
    print()
    print("=" * 72)
    print("Does the curve coincide across architectures? Binned in alpha")
    print("=" * 72)
    bins = [(0, 2), (2, 3), (3, 5), (5, 8), (8, 13), (13, 25), (25, 1e9)]
    print(f"{'alpha bin':14s} {'mlp':>16s} {'cnn':>16s} {'transformer':>16s} "
          f"{'pooled':>16s}")
    table = {}
    for lo, hi in bins:
        sel = [c for c in cs if lo <= c["alpha"] < hi]
        if not sel:
            continue
        row, entry = [], {}
        for arch in ("mlp", "cnn", "transformer"):
            s = summarise([c["gap"] for c in sel if c["arch"] == arch])
            row.append(f"{s['mean']:+7.2f} (n{s['n']:3d})" if s else f"{'-':>16s}")
            if s:
                entry[arch] = s
        sp = summarise([c["gap"] for c in sel])
        entry["pooled"] = sp
        lab = f"[{lo:4.1f},{hi:5.1f})" if hi < 1e9 else f"[{lo:4.1f},  inf)"
        print(f"{lab:14s} {row[0]:>16s} {row[1]:>16s} {row[2]:>16s} "
              f"{sp['mean']:+7.2f} (n{sp['n']:3d})")
        table[lab.strip()] = entry
    res["binned"] = table

    # ---- how well does alpha do against the outcome-based label? -----------
    d = [c["drop"] for c in cs]
    print()
    print("Predictors of the gap, compared on the same 188 cells:")
    print(f"  log alpha            spearman = {spearman(la, g):+.4f}   "
          f"(known before any evaluation)")
    print(f"  distance from dense  spearman = {spearman(d, g):+.4f}   "
          f"(measured on the outcome)")
    nd = [math.log(c["n_over_d"]) for c in cs if c["n_over_d"] > 0]
    gnd = [c["gap"] for c in cs if c["n_over_d"] > 0]
    print(f"  log mean N/d         spearman = {spearman(nd, gnd):+.4f}   "
          f"(a property of the dense net alone)")
    res["predictors"] = {"log_alpha": round(spearman(la, g), 4),
                         "distance_from_dense": round(spearman(d, g), 4),
                         "log_n_over_d": round(spearman(nd, gnd), 4)}

    # ---- same test for the correction, which should NOT collapse to zero ---
    print()
    print("=" * 72)
    print("The correction, for contrast: does it also vanish at large alpha?")
    print("=" * 72)
    for a, b, lab in (("magnitude_ls", "magnitude", "refit | magnitude"),
                      ("ot_then_ls", "ot_intrafusion", "refit | merging")):
        cc = gap_cells(runs, a, b)
        if not cc:
            continue
        x = [math.log(c["alpha"]) for c in cc]
        y = [c["gap"] for c in cc]
        print(f"  {lab:20s} n={len(cc):4d}  spearman(log alpha, gap) = "
              f"{spearman(x, y):+.4f}")
        row = []
        for lo, hi in bins:
            s = summarise([c["gap"] for c in cc if lo <= c["alpha"] < hi])
            row.append(f"{s['mean']:+6.1f}" if s else "     -")
        print(f"  {'':20s} by alpha bin: {' '.join(row)}")
        res.setdefault("correction_vs_alpha", {})[lab] = {
            "n": len(cc), "spearman": round(spearman(x, y), 4),
            "by_bin": row}

    (OUT / "a5_collapse_law.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a5_collapse_law.json'}")


if __name__ == "__main__":
    main()
