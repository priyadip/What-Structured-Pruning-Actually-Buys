#!/usr/bin/env python3
"""E1 -- seed variance on the frontier configurations.

Every configuration in the paper ran once. The compression itself is
deterministic given a calibration set (closed-form solves, greedy selection with
a fixed tie-break), so a seed changes exactly two things:

  * which calibration images are drawn, and
  * the GRIDE subsample, hence the measured `d_l`, hence the widths.

The second is the interesting one. If `d_l` moves enough that
`round(alpha * d_l)` changes, the *architecture* changes, not just the fit --
so this measures the stability of the width rule, not only of the least-squares
solve. Both are reported.

    python3 e1_seed_variance.py --seeds 0 1 2 [--models ...]

Writes `results/e1_seed_variance.csv` and prints the per-model spread.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import harness as H

OUT = H.RESULTS / "e1_seed_variance.csv"
FIELDS = ["model", "dataset", "seed", "strategy", "multiplier", "acc", "dense_acc",
          "drop", "params", "macs", "param_cut", "mac_cut", "widths", "gride",
          "calib", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    fr = H.frontiers()
    tags = a.models or sorted(fr)
    missing = [t for t in tags if t not in fr]
    if missing:
        raise SystemExit(f"no saved frontier for {missing}")

    H.redirect_csv(H.RESULTS / "e1_driver_log.csv")
    rows = []
    for tag in tags:
        f = fr[tag]
        print(f"\n=== {tag}: {f.strategy} x{f.multiplier} "
              f"(paper: {f.compressed_acc:.2f}%, {f.dense_acc:.2f}% dense)")
        for seed in a.seeds:
            t0 = time.time()
            args = H.base_args(seed=seed)
            if a.device:
                args.device = a.device
            sess = None
            try:
                sess = H.make_session(tag, args)
                r = H.run_config(sess, f.strategy, f.multiplier, stage="e1_seed")
                if r is None:
                    print(f"  seed {seed}: FAILED")
                    continue
                dims = [round(sess.dims_all[n][args.id_space], 3)
                        for n in sess.unit_order]
                rows.append({
                    "model": tag, "dataset": f.dataset, "seed": seed,
                    "strategy": f.strategy, "multiplier": f.multiplier,
                    "acc": round(r.acc, 4), "dense_acc": round(sess.dense_acc, 4),
                    "drop": round(r.drop, 4), "params": r.params, "macs": r.macs,
                    "param_cut": round(100 * (1 - r.params / sess.dense_params), 3),
                    "mac_cut": round(100 * (1 - r.macs / sess.dense_macs), 3),
                    "widths": json.dumps(r.widths), "gride": json.dumps(dims),
                    "calib": sess.calib_n, "seconds": round(time.time() - t0, 1),
                })
                print(f"  seed {seed}: {r.acc:6.2f}%  ({r.drop:+.2f})  "
                      f"params -{rows[-1]['param_cut']:.1f}%  widths {r.widths}")
            finally:
                del sess
                H.free()
        _flush(rows)

    report(rows)


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)

    print("\n" + "=" * 78)
    print("E1 -- seed variance at the frontier configuration")
    print("=" * 78)
    print(f"{'model':24s} {'n':>2s} {'acc mean':>9s} {'sd':>6s} {'range':>13s} "
          f"{'cut sd':>7s} {'widths stable?':>15s}")
    accs_sd, cuts_sd = [], []
    for m, rs in sorted(by.items()):
        av = [r["acc"] for r in rs]
        cv = [r["param_cut"] for r in rs]
        same = len({r["widths"] for r in rs}) == 1
        accs_sd.append(H.sd(av))
        cuts_sd.append(H.sd(cv))
        print(f"{m:24s} {len(rs):2d} {H.mean(av):9.2f} {H.fmt(H.sd(av)):>6s} "
              f"[{min(av):5.2f},{max(av):6.2f}] {H.fmt(H.sd(cv)):>7s} "
              f"{'yes' if same else 'NO':>15s}")

    print("-" * 78)
    print(f"median accuracy s.d. across models : {sorted(accs_sd)[len(accs_sd)//2]:.3f} pts")
    print(f"max accuracy s.d.                  : {max(accs_sd):.3f} pts")
    print(f"max parameter-cut s.d.             : {max(cuts_sd):.3f} pts")

    inside = sum(1 for rs in by.values() for r in rs if abs(r["drop"]) <= 2.5)
    total = sum(len(rs) for rs in by.values())
    print(f"runs inside the +/-2.5 window      : {inside}/{total}")
    worst = max((r for rs in by.values() for r in rs), key=lambda r: abs(r["drop"]))
    print(f"worst drop over all seeds          : {worst['drop']:+.2f} "
          f"({worst['model']}, seed {worst['seed']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
