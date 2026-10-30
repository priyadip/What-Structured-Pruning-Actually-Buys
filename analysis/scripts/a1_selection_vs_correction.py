#!/usr/bin/env python3
"""A1. Does the selection rule matter, and does the answer depend on N/d?

The paper's headline decomposition credits three components: a GRIDE budget,
target-aware selection by group OMP, and a least-squares consumer refit. On
DistilBERT the selection term measured 0.08 points against target-blind
spanning, which was written up as a Transformer-specific reversal. This asks
whether that is true, by computing the same contrast on the MLP and CNN runs
that were already on disk.

Everything is a paired difference at identical per-layer widths: both members of
every pair ran on the same checkpoint at the same multiplier, so no comparison
is confounded by how many units survived.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from load_all import load_all, paired, summarise, pearson, spearman

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    runs = load_all()
    res = {}

    # ---- the contrast that motivated this whole analysis -------------------
    print("=" * 74)
    print("OMP (target-aware) minus SPAN (target-blind), both with the refit")
    print("=" * 74)
    per_arch = {}
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        d = paired(sub, "omp_ls", "span_ls")
        s = summarise([x for _, x in d])
        per_arch[arch] = {"summary": s, "cells": len(d)}
        if s:
            print(f"  {arch:12s} n={s['n']:4d}  mean={s['mean']:+7.3f}  "
                  f"sd={s['sd']:6.3f}  t={s['t']}  omp wins {s['pos']}/{s['n']}")
    res["omp_minus_span"] = per_arch

    # ---- the correction contrast, for scale --------------------------------
    print()
    print("=" * 74)
    print("The refit, holding selection fixed")
    print("=" * 74)
    refit_pairs = [("magnitude_ls", "magnitude"), ("ot_then_ls", "ot_intrafusion")]
    res["refit"] = {}
    for a, b in refit_pairs:
        for arch in ("mlp", "cnn", "transformer"):
            sub = [r for r in runs if r.arch == arch]
            d = paired(sub, a, b)
            s = summarise([x for _, x in d])
            if s:
                print(f"  {arch:12s} {a} - {b}: n={s['n']:4d} "
                      f"mean={s['mean']:+7.3f}  t={s['t']}")
                res["refit"][f"{arch}:{a}-{b}"] = s

    # ---- is the selection gap a function of N/d? ---------------------------
    # One point per (checkpoint, budget) cell where both operators ran. N/d is
    # the mean over compressible units of nominal width over GRIDE dimension,
    # the same scalar the paper regresses the achievable cut on.
    print()
    print("=" * 74)
    print("Does the selection gap shrink as N/d grows?")
    print("=" * 74)
    pts = []
    for arch in ("mlp", "cnn", "transformer"):
        sub = [r for r in runs if r.arch == arch]
        ia = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == "omp_ls"}
        ib = {(r.model, r.mult, r.phase): r for r in sub if r.strategy == "span_ls"}
        for k in set(ia) & set(ib):
            nd = ia[k].n_over_d
            if nd == nd and nd > 0:
                pts.append({"arch": arch, "model": ia[k].model, "mult": ia[k].mult,
                            "n_over_d": nd, "gap": ia[k].acc - ib[k].acc,
                            "dense": ia[k].dense})
    import math
    x = [math.log(p["n_over_d"]) for p in pts]
    y = [p["gap"] for p in pts]
    print(f"  cells with both operators: {len(pts)}")
    print(f"  pearson  r(log N/d, gap) = {pearson(x, y):+.4f}")
    print(f"  spearman r(log N/d, gap) = {spearman(x, y):+.4f}")
    res["gap_vs_nd"] = {"n": len(pts), "pearson": round(pearson(x, y), 4),
                        "spearman": round(spearman(x, y), 4)}

    # per-architecture, to check the pooled correlation is not just the
    # architecture label in disguise
    print("\n  within each architecture:")
    for arch in ("mlp", "cnn", "transformer"):
        px = [math.log(p["n_over_d"]) for p in pts if p["arch"] == arch]
        py = [p["gap"] for p in pts if p["arch"] == arch]
        if len(px) >= 3:
            print(f"    {arch:12s} n={len(px):4d}  r={pearson(px, py):+.4f}  "
                  f"spearman={spearman(px, py):+.4f}  "
                  f"N/d range {math.exp(min(px)):.1f}-{math.exp(max(px)):.1f}")
            res.setdefault("gap_vs_nd_within", {})[arch] = {
                "n": len(px), "pearson": round(pearson(px, py), 4),
                "spearman": round(spearman(px, py), 4),
                "nd_lo": round(math.exp(min(px)), 2),
                "nd_hi": round(math.exp(max(px)), 2)}

    # binned, since a correlation on a bounded quantity is easy to over-read
    print("\n  binned by N/d:")
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 100)]
    res["gap_by_nd_bin"] = {}
    for lo, hi in bins:
        v = [p["gap"] for p in pts if lo <= p["n_over_d"] < hi]
        s = summarise(v)
        if s:
            arches = sorted({p["arch"] for p in pts if lo <= p["n_over_d"] < hi})
            print(f"    N/d in [{lo:3d},{hi:3d})  n={s['n']:4d}  "
                  f"mean gap={s['mean']:+7.3f}  sd={s['sd']:6.3f}  "
                  f"archs={','.join(a[:3] for a in arches)}")
            res["gap_by_nd_bin"][f"{lo}-{hi}"] = dict(s, archs=arches)

    (OUT / "a1_selection_vs_correction.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a1_selection_vs_correction.json'}")


if __name__ == "__main__":
    main()
