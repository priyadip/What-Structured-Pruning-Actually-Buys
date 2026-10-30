#!/usr/bin/env python3
"""A6. Both contrasts, measured at each checkpoint's own frontier.

A5 showed that the selection gap and the correction gap both shrink as the
budget loosens, so a claim of the form "selection does not matter but correction
does" is only meaningful if the two are read at the same point on the curve.
The natural such point is each checkpoint's frontier: the smallest multiplier
whose best operator is still inside the +/-2.5 window. That is the configuration
the study would actually ship, and it is defined per checkpoint rather than by a
global budget, so it controls for regime exactly.

This is the comparison the paper should have made and did not.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from load_all import load_all, summarise

OUT = Path(__file__).resolve().parents[1] / "results"
PAIRS = [
    ("correction", "magnitude_ls", "magnitude"),
    ("correction (merge)", "ot_then_ls", "ot_intrafusion"),
    ("selection: target vs span", "omp_ls", "span_ls"),
    ("selection: OMP vs magnitude", "omp_ls", "magnitude_ls"),
    ("select vs merge", "omp_ls", "ot_then_ls"),
    ("backward elimination", "fwdbwd_ls", "omp_ls"),
]


def main():
    runs = load_all()
    cells = defaultdict(dict)
    for r in runs:
        if r.arch == "transformer" and r.phase not in ("matched", "ablation"):
            continue
        cells[(r.arch, r.model, r.mult)][r.strategy] = r

    # frontier multiplier per checkpoint: smallest mult whose best operator is
    # inside the window, using only cells where at least four operators ran so
    # "best" is not decided by which happened to be present
    frontier = {}
    for (arch, model, mult), ops in cells.items():
        if len(ops) < 4:
            continue
        best = max(ops.values(), key=lambda r: r.acc)
        if (best.dense - best.acc) <= 2.5:
            key = (arch, model)
            if key not in frontier or mult < frontier[key][0]:
                frontier[key] = (mult, ops)

    print(f"checkpoints with an identified frontier: {len(frontier)}")
    print()
    hdr = f"{'checkpoint':30s} {'alpha*':>7s} " + " ".join(
        f"{lab.split(':')[0][:11]:>12s}" for lab, _, _ in PAIRS)
    print(hdr)
    print("-" * len(hdr))

    collected = defaultdict(list)
    per_ckpt = {}
    for (arch, model), (mult, ops) in sorted(frontier.items()):
        row, entry = [], {"alpha_star": mult}
        for lab, a, b in PAIRS:
            if a in ops and b in ops:
                v = ops[a].acc - ops[b].acc
                collected[lab].append(v)
                entry[lab] = round(v, 3)
                row.append(f"{v:+12.2f}")
            else:
                row.append(f"{'-':>12s}")
        print(f"{arch[:3]}/{model:26s} {mult:7.2f} " + " ".join(row))
        per_ckpt[f"{arch}/{model}"] = entry

    print()
    print("=" * 78)
    print("Pooled over checkpoints, every contrast read at that checkpoint's frontier")
    print("=" * 78)
    print(f"{'contrast':30s} {'n':>4s} {'mean':>9s} {'sd':>8s} {'t':>7s} "
          f"{'wins':>8s}")
    summary = {}
    for lab, _, _ in PAIRS:
        s = summarise(collected[lab])
        if s:
            print(f"{lab:30s} {s['n']:4d} {s['mean']:+9.2f} {s['sd']:8.2f} "
                  f"{str(s['t']):>7s} {s['pos']:4d}/{s['n']:<3d}")
            summary[lab] = s

    # the headline ratio
    corr = summary.get("correction (merge)") or summary.get("correction")
    sel = summary.get("selection: target vs span")
    if corr and sel and sel["mean"] != 0:
        ratio = corr["mean"] / sel["mean"]
        print()
        print(f"  correction is {ratio:.0f}x the selection effect at the frontier")
        summary["ratio_correction_over_selection"] = round(ratio, 1)

    res = {"per_checkpoint": per_ckpt, "pooled": summary,
           "n_checkpoints": len(frontier)}
    (OUT / "a6_at_the_frontier.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a6_at_the_frontier.json'}")


if __name__ == "__main__":
    main()
