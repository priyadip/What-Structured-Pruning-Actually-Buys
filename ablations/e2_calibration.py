#!/usr/bin/env python3
"""E2 -- how much does the calibration set size matter on a CNN?

The paper says the size is *derived* rather than chosen: `plan_calibration` does
a shape-only dry run and the driver refuses to fit any unit below `2x`
over-determination, because deep convolutions contribute only
`H_out * W_out` rows per image and an under-determined refit returns a
null-space direction that still evaluates to a plausible accuracy.

"Derived" is a stronger argument than an ablation, but it is not the same thing,
and a reviewer may read it as "not checked". This checks it. The knob is
`--od-factor`, the over-determination target that sets the calibration size, so
the sweep is in the units the rule is actually stated in:

    od-factor 2   the floor the driver will accept
    od-factor 4   the production default
    od-factor 8   2x the default
    od-factor 16  4x the default

Two things are reported: accuracy against calibration size at a fixed budget,
and whether the measured GRIDE dimensions (hence the widths) move with it.

    python3 e2_calibration.py [--models ...] [--od-factors ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import harness as H

OUT = H.RESULTS / "e2_calibration.csv"
FIELDS = ["model", "dataset", "od_factor", "calib", "multiplier", "strategy",
          "acc", "dense_acc", "drop", "params", "param_cut", "widths", "gride",
          "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["cifar10_vgg11_bn", "cifar10_resnet18"])
    ap.add_argument("--od-factors", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fr = H.frontiers()
    H.redirect_csv(H.RESULTS / "e2_driver_log.csv")
    rows = []
    for tag in a.models:
        f = fr.get(tag)
        if f is None:
            print(f"!! no saved frontier for {tag}; skipping")
            continue
        print(f"\n=== {tag}: {f.strategy} x{f.multiplier} "
              f"(paper used calib {f.calib})")
        for od in a.od_factors:
            t0 = time.time()
            sess = None
            try:
                args = H.base_args(seed=a.seed, od_factor=od)
                sess = H.make_session(tag, args)
                r = H.run_config(sess, f.strategy, f.multiplier, stage="e2_calib")
                if r is None:
                    print(f"  od {od}: FAILED (likely under-determined)")
                    continue
                dims = [round(sess.dims_all[n][args.id_space], 3)
                        for n in sess.unit_order]
                rows.append({
                    "model": tag, "dataset": f.dataset, "od_factor": od,
                    "calib": sess.calib_n, "multiplier": f.multiplier,
                    "strategy": f.strategy, "acc": round(r.acc, 4),
                    "dense_acc": round(sess.dense_acc, 4), "drop": round(r.drop, 4),
                    "params": r.params,
                    "param_cut": round(100 * (1 - r.params / sess.dense_params), 3),
                    "widths": json.dumps(r.widths), "gride": json.dumps(dims),
                    "seconds": round(time.time() - t0, 1)})
                print(f"  od {od:5.1f}  calib {sess.calib_n:6d}  {r.acc:6.2f}%  "
                      f"({r.drop:+.2f})  widths {r.widths}")
            except Exception as exc:
                print(f"  od {od}: {type(exc).__name__}: {exc}")
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

    print("\n" + "=" * 72)
    print("E2 -- accuracy against calibration size, budget held fixed")
    print("=" * 72)
    for m, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["calib"])
        base = next((r for r in rs if r["od_factor"] == 4.0), rs[0])
        print(f"\n{m}   (production default is od 4 -> calib {base['calib']})")
        print(f"  {'od':>5s} {'calib':>7s} {'acc':>7s} {'vs default':>11s} "
              f"{'widths == default?':>19s}")
        for r in rs:
            same = r["widths"] == base["widths"]
            print(f"  {r['od_factor']:5.1f} {r['calib']:7d} {r['acc']:7.2f} "
                  f"{r['acc'] - base['acc']:+11.2f} {('yes' if same else 'NO'):>19s}")
        span = max(r["acc"] for r in rs) - min(r["acc"] for r in rs)
        print(f"  full-range accuracy spread: {span:.2f} pts")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
