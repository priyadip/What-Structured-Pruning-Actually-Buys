#!/usr/bin/env python3
"""E5 -- a joint search over (channel multiplier, patch rank multiplier).

The paper's composition result tests only *sequencing*: run channel compression,
then run patch factorisation on its output. Chaining beats the better single
stage in 3 of 9 models, and the paper gives two reasons -- the two stages share
the +/-2.5 tolerance, and channel removal lowers `C_out` hence the break-even
rank `r* = F*C_out/(F+C_out)` that factorisation must stay under.

Neither reason is specific to *sequencing*. A joint choice of both multipliers
might land somewhere the sequential search cannot reach, because the sequential
search commits to a channel budget before it knows what the rank stage will
want. This runs the 2-D grid.

The comparison is at matched cost: for every (alpha_ch, alpha_r) that stays in
the window, we record the MAC and parameter reduction, and ask whether any joint
point dominates the best single-mechanism point on both axes.

    python3 e5_joint_budget.py [--models ...]

Note this reuses the *channel* driver for stage 1 and the patch factoriser for
stage 2, applied to stage 1's output in memory, so no artefact is written and
the paper's CSVs are untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

import harness as H

DRF = H.ROOT / "compression_cnn" / "data_reduction_and_fit"
if str(DRF) not in sys.path:
    sys.path.insert(0, str(DRF))

import factorize as FZ                                  # noqa: E402
from common import gride_conv as GC                     # noqa: E402
from common import surgery as SG                        # noqa: E402
from common.factored import factor_module               # noqa: E402
from common.rebuild import count_macs, count_params     # noqa: E402
from common.discovery import evaluate                   # noqa: E402

OUT = H.RESULTS / "e5_joint_budget.csv"
FIELDS = ["model", "dataset", "alpha_ch", "alpha_r", "acc", "dense_acc", "drop",
          "in_window", "params", "macs", "param_cut", "mac_cut",
          "ch_widths", "ranks", "seconds"]


def _stage_b(sess, model, alpha_r):
    """Apply Stage-B patch factorisation to `model` at rank multiplier `alpha_r`.

    Mirrors `run_input_compression_cnn.Session.factorize` exactly -- probe walk
    to measure each layer's patch GRIDE dimension, then a second walk that
    replaces each consumer by `conv_P(k x k, r) + conv_B(1 x 1)` with
    `r = round(alpha_r * d_patch)` clipped to the layer's own full rank. The one
    difference is that the object being walked is the *channel-compressed*
    network rather than the dense one, which is the whole point: the patch
    dimensions are re-measured on what stage 1 actually produced.
    """
    from common.rebuild import count_macs as _cm      # noqa: F401  (kept local)

    dims, order, full = [], [], []

    def probe(ctx):
        dims.append(GC.gride_id(
            ctx.U.double().cpu().numpy()[: sess.args.gride_patches],
            seed=sess.args.seed))
        order.append(ctx.cons_path)
        full.append(ctx.gs * ctx.C)
        return ctx.A

    SG.Walker(model, SG.clone_eval(model), sess.ck.family, sess.X,
              sess.args.patch_budget, sess.args.ridge, sess.args.seed).walk(probe)

    ranks = [int(min(max(round(alpha_r * d), 1), F)) for d, F in zip(dims, full)]
    out = SG.clone_eval(model)
    got, it = [], iter(range(len(ranks)))
    fz = FZ.BY_NAME["F1_rrr"]

    def decide(ctx):
        i = next(it)
        r = int(ranks[i])
        if r >= full[i]:
            got.append(full[i])
            return ctx.A
        ctx.require_overdetermined(2.0)
        P, B, c = fz.apply(ctx, r, None)
        assert P.shape[0] == r, f"F1_rrr produced rank {P.shape[0]} != {r}"
        SG.set_module(out, ctx.cons_path, factor_module(ctx.consumer, P, B, c))
        got.append(r)
        return ctx.A

    SG.Walker(model, out, sess.ck.family, sess.X, sess.args.patch_budget,
              sess.args.ridge, sess.args.seed).walk(decide)
    return out, got, [round(d, 2) for d in dims]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["cifar10_vgg11_bn", "cifar10_vgg16_bn",
                             "fashionmnist_vgg11_bn"])
    ap.add_argument("--alpha-ch", type=float, nargs="+",
                    default=[3.0, 4.0, 5.0, 6.0, 8.0, 12.0])
    ap.add_argument("--alpha-r", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 3.0, 4.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    H.redirect_csv(H.RESULTS / "e5_driver_log.csv")
    rows = []
    for tag in a.models:
        args = H.base_args(seed=a.seed)
        sess = None
        try:
            sess = H.make_session(tag, args)
            print(f"\n=== {tag}  dense {sess.dense_acc:.2f}%")
            for ach in a.alpha_ch:
                # stage 1: channels only (alpha_r = infinity means "no stage 2")
                t0 = time.time()
                r1 = H.run_config(sess, "S4_omp_ls", ach, stage="e5_stage1")
                if r1 is None:
                    continue
                rows.append(_row(tag, sess, ach, float("inf"), r1.acc,
                                 r1.params, r1.macs, r1.widths, None,
                                 time.time() - t0))
                print(f"  ch x{ach:<5.1f} rank --    {r1.acc:6.2f}% "
                      f"({r1.drop:+.2f})  P -{rows[-1]['param_cut']:.1f}% "
                      f"M -{rows[-1]['mac_cut']:.1f}%")

                for ar in a.alpha_r:
                    t0 = time.time()
                    try:
                        acc, p, mm, ranks = _joint(sess, ach, ar)
                    except Exception as exc:
                        print(f"  ch x{ach:<5.1f} rank x{ar:<4.1f} "
                              f"FAILED {type(exc).__name__}: {exc}")
                        continue
                    rows.append(_row(tag, sess, ach, ar, acc, p, mm,
                                     r1.widths, ranks, time.time() - t0))
                    q = rows[-1]
                    print(f"  ch x{ach:<5.1f} rank x{ar:<4.1f} {acc:6.2f}% "
                          f"({acc - sess.dense_acc:+.2f})  P -{q['param_cut']:.1f}% "
                          f"M -{q['mac_cut']:.1f}%{'  IN' if q['in_window'] else ''}")
                    _flush(rows)
        finally:
            del sess
            H.free()
    report(rows)


def _joint(sess, alpha_ch, alpha_r):
    """Channel-compress at `alpha_ch`, then patch-factorise the result at
    `alpha_r`, and evaluate. Both stages closed-form, no backward pass."""
    import strategies as ST
    comp, _ = sess.compress(ST.BY_NAME["S4_omp_ls"], alpha_ch)
    model, ranks, _ = _stage_b(sess, comp, alpha_r)
    acc = evaluate(model, sess.test_loader, sess.dev)
    p = count_params(model)
    mm = count_macs(model, sess.in_shape, sess.dev)
    del comp, model
    H.free()
    return acc, p, mm, ranks


def _row(tag, sess, ach, ar, acc, p, mm, widths, ranks, secs):
    return {
        "model": tag, "dataset": sess.ck.dataset, "alpha_ch": ach,
        "alpha_r": ar if ar == ar and ar != float("inf") else "",
        "acc": round(acc, 4), "dense_acc": round(sess.dense_acc, 4),
        "drop": round(acc - sess.dense_acc, 4),
        "in_window": int(abs(acc - sess.dense_acc) <= sess.tol),
        "params": p, "macs": mm,
        "param_cut": round(100 * (1 - p / sess.dense_params), 3),
        "mac_cut": round(100 * (1 - mm / sess.dense_macs), 3),
        "ch_widths": json.dumps(widths), "ranks": json.dumps(ranks) if ranks else "",
        "seconds": round(secs, 1)}


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    print("\n" + "=" * 76)
    print("E5 -- does a JOINT budget beat the best single mechanism?")
    print("=" * 76)
    wins = 0
    for m, rs in sorted(by.items()):
        ok = [r for r in rs if r["in_window"]]
        if not ok:
            print(f"{m}: nothing in window")
            continue
        single = [r for r in ok if r["alpha_r"] == ""]
        joint = [r for r in ok if r["alpha_r"] != ""]
        bs = max(single, key=lambda r: r["mac_cut"], default=None)
        bj = max(joint, key=lambda r: r["mac_cut"], default=None)
        if bs is None or bj is None:
            print(f"{m}: incomplete")
            continue
        better = bj["mac_cut"] > bs["mac_cut"]
        wins += better
        print(f"{m:22s} channels-only -{bs['mac_cut']:5.1f}% MACs "
              f"(ch x{bs['alpha_ch']})   joint -{bj['mac_cut']:5.1f}% MACs "
              f"(ch x{bj['alpha_ch']}, r x{bj['alpha_r']})   "
              f"{'JOINT WINS' if better else 'no gain'}")
    print(f"\njoint beats the best single mechanism in {wins}/{len(by)} models")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
