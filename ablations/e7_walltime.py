#!/usr/bin/env python3
"""E7: do the MAC reductions translate into wall-clock?

MACs are counted analytically. Structured channel removal generally does
translate (unlike unstructured sparsity, which needs sparse kernels), but the
paper has never measured it, and the *second* mechanism has a specific reason to
lag: patch-rank factorisation replaces one convolution with two, so it trades
arithmetic for an extra kernel launch. At small batch sizes that overhead can
swallow the saving entirely.

This times the saved artefacts against their dense sources:

    batch 1     latency-bound; kernel-launch overhead is visible here
    batch 128   throughput-bound; arithmetic should dominate

and reports, for each, the speedup next to the MAC reduction the paper claims.
CPU is included because that is where a launch-overhead story would be
different, and because many deployments of a compressed CIFAR-scale model are
not on a GPU.

    python3 e7_walltime.py [--iters 100] [--devices cuda cpu]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import harness as H
from common.discovery import load_checkpoint          # noqa: E402
from common.foldbn import fold_bn                     # noqa: E402
from common.rebuild import load_compressed, count_macs, count_params  # noqa: E402

HLC = H.HLC / "results"
DRF = H.ROOT / "compression_cnn" / "data_reduction_and_fit" / "results"
OUT = H.RESULTS / "e7_walltime.csv"
FIELDS = ["model", "mechanism", "device", "batch", "dense_ms", "comp_ms",
          "speedup", "mac_cut", "param_cut", "iters"]


@torch.no_grad()
def timeit(model, x, iters: int, warmup: int = 15) -> float:
    """Median forward time in milliseconds. Median, not mean: a single
    scheduler hiccup should not move the number."""
    dev = x.device.type
    model.eval()
    for _ in range(warmup):
        model(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def artefacts():
    """(tag, mechanism, arch.json) for every saved compressed model.

    For each (model, mechanism) both saved artefacts are considered and the one
    with the **larger MAC reduction** is timed, because a speedup measurement on
    a configuration that barely compresses says nothing. `*_dense_*` in the
    patch-rank study means "Stage B applied to the dense network", which is what
    Table 1 reports; compositions (`hidden_*`) are excluded.

    Artefacts that are not a genuine reduction are skipped rather than reported
    as a slowdown -- four `*_dense_frontier` files hold Stage-A configurations
    from before the frontier rule required shrinkage, and timing those would
    measure the wrong thing."""
    picks: dict[tuple[str, str], tuple[Path, float]] = {}
    for d, mech, pats in ((HLC, "channels", ("*_frontier.arch.json", "*_best.arch.json")),
                          (DRF, "patch", ("*_dense_frontier.arch.json",
                                          "*_dense_best.arch.json"))):
        for pat in pats:
            for p in sorted(d.glob(pat)):
                suffix = pat.replace("*", "")
                tag = p.name[: -len(suffix)]
                try:
                    m = json.loads(p.read_text())
                except Exception:
                    continue
                cut = 1 - m["macs"] / m["dense_macs"]
                if cut <= 0:                       # not a compression; skip
                    continue
                key = (tag, mech)
                if key not in picks or cut > picks[key][1]:
                    picks[key] = (p, cut)
    return [(tag, mech, p) for (tag, mech), (p, _) in sorted(picks.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 128])
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()

    devices = a.devices or (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    rows = []
    seen_dense: dict[tuple[str, str, int], float] = {}

    for tag, mech, arch in artefacts():
        if a.models and tag not in a.models:
            continue
        try:
            comp, meta = load_compressed(arch, "cpu")
        except Exception as exc:
            print(f"!! {tag}/{mech}: could not rebuild ({type(exc).__name__}: {exc})")
            continue
        path, mname = H.checkpoint_for(tag)
        ck = load_checkpoint(path, mname, "cpu")
        dense = fold_bn(ck.model, ck.family)
        shape = tuple(meta["input_shape"])
        mac_cut = 100 * (1 - meta["macs"] / meta["dense_macs"])
        par_cut = 100 * (1 - meta["params"] / meta["dense_params"])

        for dev in devices:
            d = dense.to(dev).eval()
            c = comp.to(dev).eval()
            for bs in a.batches:
                x = torch.randn(bs, *shape, device=dev)
                key = (tag, dev, bs)
                if key not in seen_dense:
                    seen_dense[key] = timeit(d, x, a.iters)
                dm = seen_dense[key]
                cm = timeit(c, x, a.iters)
                rows.append({
                    "model": tag, "mechanism": mech, "device": dev, "batch": bs,
                    "dense_ms": round(dm, 4), "comp_ms": round(cm, 4),
                    "speedup": round(dm / cm, 4),
                    "mac_cut": round(mac_cut, 2), "param_cut": round(par_cut, 2),
                    "iters": a.iters})
                print(f"{tag:24s} {mech:8s} {dev:4s} b{bs:<4d} "
                      f"dense {dm:8.3f}ms  comp {cm:8.3f}ms  "
                      f"{dm/cm:5.2f}x  (MACs -{mac_cut:.1f}%)")
            d.to("cpu")
            c.to("cpu")
        del comp, dense, ck
        H.free()
        _flush(rows)

    report(rows)


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    print("\n" + "=" * 76)
    print("E7 -- measured speedup against the analytic MAC reduction")
    print("=" * 76)
    print(f"{'mechanism':10s} {'device':7s} {'batch':>5s} {'n':>3s} "
          f"{'median MAC cut':>15s} {'implied':>8s} {'measured':>9s} {'realised':>9s}")
    groups = {}
    for r in rows:
        groups.setdefault((r["mechanism"], r["device"], r["batch"]), []).append(r)
    for (mech, dev, bs), rs in sorted(groups.items()):
        cuts = sorted(r["mac_cut"] for r in rs)
        cut = cuts[len(cuts) // 2]
        implied = 1.0 / (1.0 - cut / 100.0)
        got = statistics.median(r["speedup"] for r in rs)
        print(f"{mech:10s} {dev:7s} {bs:5d} {len(rs):3d} {cut:14.1f}% "
              f"{implied:7.2f}x {got:8.2f}x {100*(got-1)/(implied-1) if implied>1 else float('nan'):8.0f}%")
    print("\n'realised' is the fraction of the arithmetic speedup that actually")
    print("appears in wall-clock. Below 100% means overhead the MAC count misses.")
    print(f"\nwrote {OUT}")



if __name__ == "__main__":
    main()
