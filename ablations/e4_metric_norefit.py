#!/usr/bin/env python3
"""E4 -- the Gauss-Newton metric with the downstream refit disabled.

This is the experiment the project has named as decisive in three separate
documents over two months and never run.

**The standing hypothesis.** Weighting a layer-wise fit by the logit-pullback
metric `Omega = E[J^T J]` helps when nothing downstream can compensate, and
hurts or ties when a closed-form refit follows, because the metric and the refit
then correct the same thing twice. Evidence so far is only consistent with it:

    MLP input reduction, hidden layers frozen   metric HELPS   (gap .05 vs .47)
    MLP hidden layers, consumer refitted        metric HURTS   (~-1 pt)
    CNN channels, consumer refitted             metric HURTS   (-0.27, 4 cells)
    CNN patch rank, only the 1x1 mix refitted   DEAD HEAT      (-0.02, 169 cells)

Every CNN condition tested so far has *some* refit in it. This runs the missing
cell: selection scored in the metric, and then **no refit at all** -- the
consumer keeps its original weights restricted to the kept channels, exactly as
a pruning method with no reconstruction step would.

Four strategies, a 2x2 over (metric / plain) x (refit / no refit):

    M1_plain_norefit    Frobenius selection, no refit      <- baseline for the frozen arm
    M2_metric_norefit   metric selection,    no refit      <- the missing cell
    M3_plain_refit      Frobenius selection, our refit     (= S4_omp_ls)
    M4_metric_refit     metric selection,    our refit     (= N1_metric_omp)

The hypothesis predicts M2 - M1 > 0 while M4 - M3 <= 0. If both are negative the
hypothesis is wrong and the paper's Discussion paragraph becomes a plain
negative result.

    python3 e4_metric_norefit.py [--models ...] [--mults ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import torch
import torch.nn as nn

import harness as H
from common import linalg as LA          # noqa: E402
import strategies as ST                  # noqa: E402

OUT = H.RESULTS / "e4_metric_norefit.csv"
FIELDS = ["model", "dataset", "multiplier", "strategy", "metric", "refit",
          "acc", "dense_acc", "drop", "params", "param_cut", "widths", "seconds"]


def _metric_scored_stats(ctx):
    """`(G, P_omega)` with the cross-product taken in the metric the frozen tail
    induces on this layer's output.

    The selection objective is `|| Y - U_S U_S^+ Y ||^2_Omega`, i.e. the residual
    measured in `Omega` rather than in Frobenius norm. Writing `Omega = L^T L`
    (symmetric factor via eigen-decomposition, negatives clamped -- `Omega` is
    PSD by construction but its estimate can have small negative eigenvalues),
    the objective is the plain one on the rotated target `Y L^T`, so only the
    cross-product `P` changes and `G` is untouched.
    """
    G, P, us, ys, n = ctx.stats
    Om = getattr(ctx, "metric", None)
    if Om is None:
        return G, P
    Om = LA.f64(Om)
    Om = 0.5 * (Om + Om.t())
    w, V = torch.linalg.eigh(Om)
    L = V @ torch.diag(w.clamp(min=0).sqrt()) @ V.t()      # symmetric PSD factor
    return G, LA.f64(P) @ L


def _no_refit(ctx, keep):
    """Keep the consumer's own weights on the surviving channels. This is what a
    pruning method without a reconstruction step does, and it is the condition
    the hypothesis is about."""
    V = ctx.consumer.weight.detach()
    Vf = V.reshape(V.shape[0], ctx.C, -1)[:, keep].reshape(V.shape[0], -1)
    return Vf.clone(), ctx.consumer.bias.detach().clone()


class PlainNoRefit(ST.Strategy):
    name = "M1_plain_norefit"
    note = "OMP selection against U^T Y, consumer weights kept as-is (no refit)"
    family = "select"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = LA.group_select(G, P, ctx.gs, m)
        Vf, c = _no_refit(ctx, keep)
        return (*ST._rows(ctx, keep), Vf, c)


class MetricNoRefit(ST.Strategy):
    name = "M2_metric_norefit"
    note = "OMP selection in the logit-pullback metric, no refit -- the missing cell"
    family = "select"
    needs_metric = True

    def apply(self, ctx, m):
        G, Pw = _metric_scored_stats(ctx)
        keep = LA.group_select(G, Pw, ctx.gs, m)
        Vf, c = _no_refit(ctx, keep)
        return (*ST._rows(ctx, keep), Vf, c)


class PlainRefit(ST.Strategy):
    name = "M3_plain_refit"
    note = "OMP selection + consumer refit (identical to S4_omp_ls; run for pairing)"
    family = "select"

    def apply(self, ctx, m):
        G, P, us, ys, n = ctx.stats
        keep = LA.group_select(G, P, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*ST._rows(ctx, keep), V, c)


class MetricRefit(ST.Strategy):
    name = "M4_metric_refit"
    note = "metric-scored selection + consumer refit"
    family = "select"
    needs_metric = True

    def apply(self, ctx, m):
        G, Pw = _metric_scored_stats(ctx)
        keep = LA.group_select(G, Pw, ctx.gs, m)
        V, c = ctx.ls(keep)
        return (*ST._rows(ctx, keep), V, c)


VARIANTS = [PlainNoRefit(), MetricNoRefit(), PlainRefit(), MetricRefit()]
IS_METRIC = {"M2_metric_norefit", "M4_metric_refit"}
IS_REFIT = {"M3_plain_refit", "M4_metric_refit"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["cifar10_vgg11_bn", "cifar10_resnet18",
                             "cifar100_vgg11_bn", "fashionmnist_vgg16_bn"])
    ap.add_argument("--mults", type=float, nargs="*",
                    default=[2.0, 3.0, 4.0, 6.0, 8.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    for v in VARIANTS:
        H.register(v)
    H.redirect_csv(H.RESULTS / "e4_driver_log.csv")

    rows = []
    for tag in a.models:
        args = H.base_args(seed=a.seed)
        sess = None
        try:
            sess = H.make_session(tag, args)
            print(f"\n=== {tag}  dense {sess.dense_acc:.2f}%")
            for mlt in a.mults:
                for v in VARIANTS:
                    t0 = time.time()
                    r = H.run_config(sess, v.name, mlt, stage="e4_metric")
                    if r is None:
                        continue
                    rows.append({
                        "model": tag, "dataset": sess.ck.dataset, "multiplier": mlt,
                        "strategy": v.name,
                        "metric": int(v.name in IS_METRIC),
                        "refit": int(v.name in IS_REFIT),
                        "acc": round(r.acc, 4), "dense_acc": round(sess.dense_acc, 4),
                        "drop": round(r.drop, 4), "params": r.params,
                        "param_cut": round(100 * (1 - r.params / sess.dense_params), 3),
                        "widths": json.dumps(r.widths),
                        "seconds": round(time.time() - t0, 1)})
                _flush(rows)
        finally:
            del sess
            H.free()
    report(rows)


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["multiplier"]), {})[r["strategy"]] = r["acc"]

    def paired(a, b):
        d = [c[a] - c[b] for c in cells.values() if a in c and b in c]
        if not d:
            return None
        n = len(d)
        m = sum(d) / n
        var = sum((x - m) ** 2 for x in d) / n
        se = (var / n) ** 0.5 if n > 1 else float("nan")
        return {"mean": m, "se": se, "t": m / se if se else float("nan"),
                "wins": sum(x > 0 for x in d), "n": n}

    print("\n" + "=" * 72)
    print("E4 -- does the output metric help once nothing downstream can fix it?")
    print("=" * 72)
    frozen = paired("M2_metric_norefit", "M1_plain_norefit")
    refit = paired("M4_metric_refit", "M3_plain_refit")
    for label, s in (("NO refit downstream  (M2 - M1)", frozen),
                     ("refit downstream     (M4 - M3)", refit)):
        if s:
            print(f"{label}: {s['mean']:+7.3f}  se {s['se']:.3f}  "
                  f"t {s['t']:+5.2f}  wins {s['wins']}/{s['n']}")
    if frozen and refit:
        print("-" * 72)
        # The hypothesis says the metric *helps* when frozen -- a positive
        # effect, not merely a non-negative one. A mean of +0.15 with t = +0.11
        # is a null result and must not be scored as a confirmation, so the test
        # requires the frozen arm to be distinguishable from zero.
        helps = frozen["mean"] > 0 and abs(frozen["t"]) > 2.0
        ordered = frozen["mean"] > refit["mean"]
        print(f"frozen arm distinguishable from zero: "
              f"{'yes' if helps else 'NO (t = %+.2f)' % frozen['t']}")
        print(f"predicted ordering (frozen > refitted): "
              f"{'yes' if ordered else 'no'}")
        print("hypothesis (metric HELPS when frozen, hurts/ties with a refit): "
              f"{'SUPPORTED' if (helps and ordered) else 'NOT SUPPORTED'}")
        print(f"difference of differences: {frozen['mean'] - refit['mean']:+.3f} pts")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
