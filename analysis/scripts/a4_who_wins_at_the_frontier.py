#!/usr/bin/env python3
"""A4. At the budget you would actually ship, which operator wins?

Every study in this repository reports its ranking pooled over the whole
multiplier ladder, and every one of them concludes that group OMP against the
consumer's target is the best selection rule. But the ladder spends most of its
rungs in a regime nobody would deploy: on `imagenet32_vgg19_bn` the ×1.5 rung
scores 0.57% against a dense 49.06%. A mean taken over that ladder is dominated
by cells where the network has already been destroyed.

This asks the question the ladder average cannot: at each model's *frontier*, the
smallest budget still inside the +/-2.5 window, which operator is actually best,
and by how much is it better than the rest of the refit family?

The CNN study's own generated summary already contains the answer in plain sight
(`cnn_compression_summary.md` lists `S3_span_ls` as the winner at three model
frontiers and bare `S1_magnitude` at three light budgets), but no analysis in the
repository ever aggregated it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from load_all import load_all, summarise

OUT = Path(__file__).resolve().parents[1] / "results"
REFIT_FAMILY = ["omp_ls", "span_ls", "magnitude_ls", "fwdbwd_ls", "ot_then_ls",
                "augdict_omp", "augdict_fwdbwd", "svd_ls", "iterative_ls",
                "fisher_omp", "fisher_span_ls", "random_ls"]


def main():
    runs = load_all()
    res = {}

    # ---- cells: (arch, model, multiplier) with at least four operators ------
    cells = defaultdict(dict)
    for r in runs:
        if r.arch == "transformer" and r.phase not in ("matched", "ablation"):
            continue
        cells[(r.arch, r.model, r.mult)][r.strategy] = r
    cells = {k: v for k, v in cells.items() if len(v) >= 4}
    print(f"cells with >= 4 operators at identical width: {len(cells)}")

    # ---- who wins, split by whether the cell is deployable -----------------
    win_in, win_out = defaultdict(int), defaultdict(int)
    spread_in, spread_out = [], []
    n_in = n_out = 0
    for (arch, model, mult), ops in cells.items():
        best = max(ops.values(), key=lambda r: r.acc)
        inside = (best.dense - best.acc) <= 2.5
        fam = [r.acc for s, r in ops.items() if s in REFIT_FAMILY]
        if inside:
            n_in += 1
            win_in[best.strategy] += 1
            if len(fam) >= 2:
                spread_in.append(max(fam) - min(fam))
        else:
            n_out += 1
            win_out[best.strategy] += 1
            if len(fam) >= 2:
                spread_out.append(max(fam) - min(fam))

    print()
    print("=" * 72)
    print("Which operator takes the cell?")
    print("=" * 72)
    print(f"{'operator':22s} {'inside window':>16s} {'outside window':>16s}")
    for s in sorted(set(win_in) | set(win_out),
                    key=lambda s: -(win_in[s] + win_out[s])):
        print(f"{s:22s} {win_in[s]:8d} /{n_in:4d}    {win_out[s]:8d} /{n_out:4d}")
    res["winners"] = {"inside": dict(win_in), "outside": dict(win_out),
                      "n_inside": n_in, "n_outside": n_out}

    si, so = summarise(spread_in), summarise(spread_out)
    print()
    print("Spread within the refit family (best minus worst, points):")
    print(f"  inside the window : n={si['n']:4d}  mean={si['mean']:6.2f}  "
          f"median={sorted(spread_in)[len(spread_in) // 2]:6.2f}  max={max(spread_in):6.2f}")
    print(f"  outside           : n={so['n']:4d}  mean={so['mean']:6.2f}  "
          f"median={sorted(spread_out)[len(spread_out) // 2]:6.2f}  max={max(spread_out):6.2f}")
    res["refit_family_spread"] = {"inside": si, "outside": so}

    # ---- the frontier cell of every checkpoint -----------------------------
    print()
    print("=" * 72)
    print("Per-checkpoint frontier: smallest budget inside the window")
    print("=" * 72)
    by_model = defaultdict(list)
    for (arch, model, mult), ops in cells.items():
        best = max(ops.values(), key=lambda r: r.acc)
        if (best.dense - best.acc) <= 2.5:
            by_model[(arch, model)].append((mult, ops))
    print(f"{'checkpoint':30s} {'mult':>6s} {'winner':>16s} {'gap to omp_ls':>14s}")
    gaps = []
    frontier_rows = {}
    for (arch, model), lst in sorted(by_model.items()):
        mult, ops = min(lst, key=lambda t: t[0])
        best = max(ops.values(), key=lambda r: r.acc)
        gap = (best.acc - ops["omp_ls"].acc) if "omp_ls" in ops else float("nan")
        if gap == gap:
            gaps.append(gap)
        print(f"{arch[:3]}/{model:26s} {mult:6.2f} {best.strategy:>16s} "
              f"{gap:+14.2f}")
        frontier_rows[f"{arch}/{model}"] = {
            "mult": mult, "winner": best.strategy, "acc": best.acc,
            "dense": best.dense, "gap_to_omp_ls": round(gap, 3) if gap == gap else None,
            "n_ops": len(ops)}
    res["frontier_winners"] = frontier_rows
    g = summarise(gaps)
    nonomp = sum(1 for v in frontier_rows.values() if v["winner"] != "omp_ls")
    print()
    print(f"  omp_ls is NOT the winner at {nonomp} of {len(frontier_rows)} "
          f"checkpoint frontiers")
    print(f"  mean amount omp_ls gives away at its own frontier: "
          f"{g['mean']:+.3f} points (max {max(gaps):.2f})")
    res["frontier_summary"] = {"n_checkpoints": len(frontier_rows),
                               "omp_not_winner": nonomp, "gap": g}

    (OUT / "a4_frontier_winners.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'a4_frontier_winners.json'}")


if __name__ == "__main__":
    main()
