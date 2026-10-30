#!/usr/bin/env python3
"""A2. The selection gap is a function of the operating regime, not of the
architecture and not of N/d.

A1 pooled every budget on the ladder, including multipliers where the network
has been reduced to near chance. A contrast measured there says nothing about
the regime anyone would deploy: at 10% accuracy on CIFAR-10 both operators have
failed and the difference between them is noise on a destroyed network. The
Transformer ablation, by contrast, only ever ran budgets near its frontier, so
A1's cross-architecture comparison was between different parts of the curve.

This stratifies by how far the *better* of the two operators sits from dense
accuracy, which is a property of the cell rather than of either operator, and so
does not select on the outcome being compared.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from load_all import load_all, summarise, pearson, spearman

OUT = Path(__file__).resolve().parents[1] / "results"
BANDS = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 1e9)]


def cells(runs, a="omp_ls", b="span_ls"):
    out = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == a}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == b}
        for k in set(ia) & set(ib):
            ra, rb = ia[k], ib[k]
            best_drop = ra.dense - max(ra.acc, rb.acc)      # >= 0, in points
            out.append({"arch": arch, "model": ra.model, "mult": ra.mult,
                        "gap": ra.acc - rb.acc, "best_drop": best_drop,
                        "n_over_d": ra.n_over_d, "dense": ra.dense,
                        "acc_a": ra.acc, "acc_b": rb.acc,
                        "cut": ra.params_cut})
    return out


def main():
    runs = load_all()
    res = {}

    for pair in (("omp_ls", "span_ls"), ("omp_ls", "magnitude_ls"),
                 ("omp_ls", "ot_then_ls")):
        a, b = pair
        cs = cells(runs, a, b)
        if not cs:
            continue
        print("=" * 78)
        print(f"{a} minus {b}, stratified by how far the better one is from dense")
        print("=" * 78)
        print(f"{'band (points below dense)':28s} {'n':>5s} {'mean':>8s} "
              f"{'sd':>7s} {'archs':>16s}")
        rows = {}
        for lo, hi in BANDS:
            v = [c for c in cs if lo <= c["best_drop"] < hi]
            s = summarise([c["gap"] for c in v])
            if not s:
                continue
            arches = sorted({c["arch"] for c in v})
            lab = f"  [{lo:5.1f}, {hi:5.1f})" if hi < 1e9 else f"  [{lo:5.1f},   inf)"
            print(f"{lab:28s} {s['n']:5d} {s['mean']:+8.3f} {s['sd']:7.3f} "
                  f"{','.join(x[:3] for x in arches):>16s}")
            rows[f"{lo}-{hi}"] = dict(s, archs=arches)
        res[f"{a}_minus_{b}"] = rows

        # inside the deployable band only, split by architecture
        print(f"\n  inside the +/-2.5 window only, by architecture:")
        band = [c for c in cs if c["best_drop"] < 2.5]
        for arch in ("mlp", "cnn", "transformer"):
            s = summarise([c["gap"] for c in band if c["arch"] == arch])
            if s:
                print(f"    {arch:12s} n={s['n']:4d}  mean={s['mean']:+7.3f}  "
                      f"sd={s['sd']:6.3f}  t={s['t']}  wins {s['pos']}/{s['n']}")
                res.setdefault(f"{a}_minus_{b}_window", {})[arch] = s
        allw = summarise([c["gap"] for c in band])
        if allw:
            print(f"    {'ALL':12s} n={allw['n']:4d}  mean={allw['mean']:+7.3f}  "
                  f"sd={allw['sd']:6.3f}  t={allw['t']}  wins {allw['pos']}/{allw['n']}")
            res.setdefault(f"{a}_minus_{b}_window", {})["all"] = allw
        print()

    # ---- the same question as a continuous relationship --------------------
    cs = cells(runs, "omp_ls", "span_ls")
    x = [c["best_drop"] for c in cs]
    y = [c["gap"] for c in cs]
    print("=" * 78)
    print("Selection gap against distance from dense, continuous")
    print("=" * 78)
    print(f"  n={len(cs)}  pearson r = {pearson(x, y):+.4f}  "
          f"spearman = {spearman(x, y):+.4f}")
    res["gap_vs_distance_from_dense"] = {
        "n": len(cs), "pearson": round(pearson(x, y), 4),
        "spearman": round(spearman(x, y), 4)}
    for arch in ("mlp", "cnn"):
        px = [c["best_drop"] for c in cs if c["arch"] == arch]
        py = [c["gap"] for c in cs if c["arch"] == arch]
        print(f"    {arch:12s} n={len(px):4d} r={pearson(px, py):+.4f} "
              f"spearman={spearman(px, py):+.4f}")
        res["gap_vs_distance_from_dense"][arch] = {
            "n": len(px), "pearson": round(pearson(px, py), 4),
            "spearman": round(spearman(px, py), 4)}

    # ---- and re-test N/d *within* the deployable band ----------------------
    band = [c for c in cs if c["best_drop"] < 2.5]
    bx = [math.log(c["n_over_d"]) for c in band if c["n_over_d"] > 0]
    by = [c["gap"] for c in band if c["n_over_d"] > 0]
    print()
    print("N/d re-tested inside the window only "
          "(A1's positive correlation was a regime artefact):")
    print(f"  n={len(bx)}  pearson r = {pearson(bx, by):+.4f}  "
          f"spearman = {spearman(bx, by):+.4f}")
    res["nd_within_window"] = {"n": len(bx),
                               "pearson": round(pearson(bx, by), 4),
                               "spearman": round(spearman(bx, by), 4)}

    (OUT / "a2_regime.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a2_regime.json'}")


if __name__ == "__main__":
    main()
