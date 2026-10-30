#!/usr/bin/env python3
"""Reducing what a CNN reads: GRIDE-budgeted patch-space factorisation.

AIM
---
The MLP counterpart of this study (`../../explore_data_reduction_and_fit`) cut
the input dimension of a compressed MLP from `D` to `r = round(multiplier *
GRIDE d_input)` and refitted the first layer in closed form, with every hidden
layer frozen. This carries that over to VGG and ResNet.

THE RULE, unchanged
-------------------
Every reduced dimension comes from GRIDE measured at runtime:

        r_l = round(multiplier * d_l),   d_l = GRIDE ID of layer l's patch cloud

Nothing is hardcoded, no PCA variance threshold, no fixed rank.

WHAT "THE INPUT" IS FOR A CONVOLUTION
-------------------------------------
An MLP's first layer reads the whole flattened image, so `d_input` was the
intrinsic dimension of the image manifold in `R^(3*H*W)`. A conv layer reads a
`C_in*k*k` **patch** and never anything larger. For the first conv that cloud is
the raw **RGB patch manifold**, and its dimension is a different -- much smaller
-- number than the image manifold's. Both are measured and reported, because the
gap between them is the whole reason the port is not mechanical.

Two stages, and the honest difference between them
--------------------------------------------------
STAGE A  first conv only. This is the faithful port: the hidden layers stay
         frozen exactly as in the MLP study, and only what the network reads
         from the image is reduced. On a CNN it is worth almost nothing --
         a VGG11's first conv is 1 728 of 9.2M parameters -- and saying so is
         the result, not a failure.

STAGE B  every conv. The same operation, applied to every layer's patch space.
         This refits each layer's `1x1` mixing matrix, so it goes beyond "hidden
         layers frozen" -- it is still closed form and still zero backward
         passes, but it is a generalisation and is labelled as one. This is
         where the compression is, and unlike channel compression it cuts
         parameters and multiply-accumulates by the *same* factor, because a
         conv reuses its weights at every spatial position.

    python3 run_input_compression_cnn.py --models cifar10_vgg11_bn
    python3 run_input_compression_cnn.py --models cifar10_vgg11_bn --source dense
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent
PROJECT_ROOT = EXT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(EXT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import gride_conv as GC              # noqa: E402
from common import surgery as SG                 # noqa: E402
from common.discovery import (                   # noqa: E402
    calibration_batch, discover, evaluate, load_checkpoint, loaders_for)
from common.factored import break_even_rank, factor_module   # noqa: E402
from common.foldbn import fold_bn                # noqa: E402
from common.rebuild import (                     # noqa: E402
    count_macs, count_params, load_compressed, save_compressed)

import factorize as FZ                           # noqa: E402
from common.jacobian_metric import metric_pair, pullback_metrics  # noqa: E402

RESULTS = HERE / "results"
HIDDEN_RESULTS = EXT_ROOT / "hidden_layer_compression" / "results"
CSV_PATH = RESULTS / "input_compression_cnn.csv"
PROGRESS = HERE / "PROGRESS.md"

CSV_FIELDS = [
    "timestamp", "model_name", "dataset", "family", "source", "stage",
    "strategy", "multiplier", "ranks", "break_even_ranks", "patch_gride_dims",
    "d_rgb_patch", "d_rgb_patch_grey", "d_image", "rgb_patch_dim",
    "compressed_acc", "base_acc", "dense_acc", "drop_vs_dense", "drop_vs_base",
    "target_hit", "params", "param_reduction_pct", "macs", "mac_reduction_pct",
    "dense_params", "dense_macs", "phase", "calib", "ridge", "seconds", "note",
]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def csv_append(row):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fresh = not CSV_PATH.exists()
    if not fresh:
        # A header that disagrees with CSV_FIELDS is silent corruption: DictWriter
        # writes values in the *current* field order against the *old* header, so
        # every column past the difference shifts and `target_hit` starts reading
        # somebody else's value. Rotate instead of appending into a mismatch.
        with CSV_PATH.open() as fh:
            head = next(csv.reader(fh), [])
        if head and head != CSV_FIELDS:
            bak = CSV_PATH.with_suffix(f".csv.schema{len(head)}")
            CSV_PATH.rename(bak)
            print(f"[csv] header/schema mismatch; previous file moved to {bak.name}")
            fresh = True
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def progress(t):
    with PROGRESS.open("a") as f:
        f.write(t + "\n")


@dataclass
class Run:
    strategy: str
    multiplier: float
    ranks: list[int]
    acc: float
    params: int
    macs: int
    hit: bool
    stage: str
    seconds: float = 0.0


# ===========================================================================
class Session:
    def __init__(self, ck, args, base_model=None, base_meta=None):
        self.ck, self.args = ck, args
        self.dev, self.tol = args.device, args.tol
        self.base_model, self.base_meta = base_model, base_meta
        self.runs: list[Run] = []
        self.best: Run | None = None
        self.frontier: Run | None = None

    # -- phase 0 -----------------------------------------------------------
    def setup(self):
        a = self.args
        self.train_loader, self.test_loader = loaders_for(self.ck, a.batch_size, a.workers)
        self.dense_acc = evaluate(self.ck.model, self.test_loader, self.dev)

        if self.base_model is None:                       # operate on the dense net
            self.base = fold_bn(self.ck.model, self.ck.family).to(self.dev)
            self.source = "dense"
        else:
            self.base = self.base_model.to(self.dev)
            self.source = f"hidden_{self.base_meta['kind']}"
        self.base_acc = evaluate(self.base, self.test_loader, self.dev)

        probe = calibration_batch(self.train_loader, 8, self.dev, seed=a.seed)
        self.in_shape = tuple(probe.shape[1:])
        need, info = SG.plan_calibration(self.base, self.ck.family, probe, a.od_factor)
        self.calib_n = max(a.calib, need)
        self.unit_info = info
        self.X = calibration_batch(self.train_loader, self.calib_n, self.dev, seed=a.seed)

        self.dense_params, self.dense_macs = self.ck.n_params, count_macs(
            fold_bn(self.ck.model, self.ck.family).to(self.dev), self.in_shape, self.dev)
        self.base_params = count_params(self.base)
        self.base_macs = count_macs(self.base, self.in_shape, self.dev)

        log(f"AIM: {self.ck.tag} [{self.source}] base {self.base_acc:.2f}%, "
            f"dense {self.dense_acc:.2f}% -> stay within +/-{self.tol} of dense")
        log(f"  {len(info)} factorisable layers, calib {self.calib_n}, "
            f"params {self.base_params:,} of {self.dense_params:,}, "
            f"MACs {self.base_macs/1e6:.1f}M of {self.dense_macs/1e6:.1f}M")

    # -- phase 1: GRIDE on what each layer reads ---------------------------
    def gride_profile(self):
        t0 = time.time()
        rgb = GC.rgb_patch_id(self.X[: min(2000, self.calib_n)], k=3, stride=2,
                              seed=self.args.seed)
        self.d_rgb, self.d_image = rgb["patch"], rgb["image"]
        self.d_grey = rgb.get("patch_gray")
        log(f"  RGB input manifold: image D={int(np.prod(self.in_shape))} "
            f"d_image={self.d_image:.2f} | 3x3x3 patch D={int(rgb['patch_dim'])} "
            f"d_patch={self.d_rgb:.2f}"
            + (f" (greyscale {rgb['patch_gray']:.2f})" if "patch_gray" in rgb else ""))

        dims, order, be = [], [], []
        def probe(ctx):
            dims.append(GC.gride_id(ctx.U.double().cpu().numpy()[
                :self.args.gride_points], seed=self.args.seed))
            order.append(ctx.cons_path)
            be.append(break_even_rank(ctx.consumer, ctx.gs))
            return ctx.A

        SG.Walker(self.base, SG.clone_eval(self.base), self.ck.family, self.X,
                  self.args.patch_budget, self.args.ridge, self.args.seed).walk(probe)
        self.dims, self.layer_order, self.break_even = dims, order, be
        self.full_ranks = [i["gs"] * i["C"] for i in self.unit_info]
        log(f"  patch GRIDE in {time.time()-t0:.0f}s: "
            + ", ".join(f"{n.split('.')[-1]}:{d:.1f}/{F}(be {b:.0f})"
                        for n, d, F, b in zip(order, dims, self.full_ranks, be)))
        self.mult_grid = GC.multiplier_grid(dims, self.full_ranks)
        log(f"  multiplier ladder: {self.mult_grid}")

    def metrics(self):
        if getattr(self, "_metrics", None) is None:
            log("  computing logit-pullback metrics (forward-only recursion)")
            raw = pullback_metrics(self.base, self.ck.family, self.X, self.args.chunk)
            self._metrics = {k: metric_pair(v) for k, v in raw.items()}
        return self._metrics

    # -- the factorisation -------------------------------------------------
    def factorize(self, fz, multiplier, stage):
        ranks = [int(np.clip(round(multiplier * d), 1, F))
                 for d, F in zip(self.dims, self.full_ranks)]
        comp = SG.clone_eval(self.base)
        met = self.metrics() if fz.needs_metric else None
        got, it = [], iter(range(len(ranks)))

        def decide(ctx):
            i = next(it)
            r = int(ranks[i])
            if stage == "A" and i > 0:
                got.append(self.full_ranks[i])
                return ctx.A
            if r >= self.full_ranks[i]:
                got.append(self.full_ranks[i])
                return ctx.A
            ctx.require_overdetermined(2.0)
            m = met.get(ctx.cons_path) if met else None
            P, B, c = fz.apply(ctx, r, m)
            assert P.shape[0] == r, f"{fz.name} produced rank {P.shape[0]} != {r}"
            SG.set_module(comp, ctx.cons_path, factor_module(ctx.consumer, P, B, c))
            got.append(r)
            return ctx.A

        SG.Walker(self.base, comp, self.ck.family, self.X, self.args.patch_budget,
                  self.args.ridge, self.args.seed, self.args.chunk).walk(decide)
        return comp, got

    def evaluate_config(self, fz, multiplier, stage, phase, note=""):
        t0 = time.time()
        try:
            comp, ranks = self.factorize(fz, multiplier, stage)
        except Exception as exc:
            # A dropped configuration must still leave a trace: a summary built
            # only from configurations that happened to succeed is biased towards
            # the cheap end of the ladder.
            log(f"    {fz.name} x{multiplier} [{stage}]: FAILED {type(exc).__name__}: {exc}")
            if self.args.debug:
                traceback.print_exc()
            torch.cuda.empty_cache()
            csv_append(self._row(
                Run(fz.name, multiplier, [], float("nan"), 0, 0, False, stage,
                    time.time() - t0), phase,
                note=f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"))
            return None
        acc = evaluate(comp, self.test_loader, self.dev)
        p, mm = count_params(comp), count_macs(comp, self.in_shape, self.dev)
        hit = abs(acc - self.dense_acc) <= self.tol
        r = Run(fz.name, multiplier, ranks, acc, p, mm, hit, stage, time.time() - t0)
        self.runs.append(r)
        csv_append(self._row(r, phase, note))
        log(f"    {fz.name:18s} x{multiplier:<6.2f} [{stage}] {acc:6.2f}%  "
            f"(dense {acc-self.dense_acc:+.2f}, base {acc-self.base_acc:+.2f})  "
            f"params {100*(p/self.dense_params-1):+5.1f}%  "
            f"MACs {100*(mm/self.dense_macs-1):+5.1f}%  {'HIT' if hit else ''}")
        if self.best is None or acc > self.best.acc:
            self.best = r
        # A configuration only counts as a frontier if it is actually a
        # compression. Ranks above the break-even point make the factorised
        # layer *larger* than the dense one, and a Stage-A row barely touches the
        # model at all -- either can clear the window while saving nothing, and
        # reporting that as a frontier would be meaningless.
        real = mm < self.dense_macs and p < self.dense_params and r.stage == "B"
        if hit and real and (self.frontier is None or mm < self.frontier.macs):
            self.frontier = r
        del comp
        torch.cuda.empty_cache()
        return r

    def _row(self, r, phase, note=""):
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.ck.tag, "dataset": self.ck.dataset,
            "family": self.ck.family, "source": self.source, "stage": r.stage,
            "strategy": r.strategy, "multiplier": r.multiplier,
            "ranks": json.dumps(r.ranks),
            "break_even_ranks": json.dumps([round(b, 1) for b in self.break_even]),
            "patch_gride_dims": json.dumps([round(d, 3) for d in self.dims]),
            "d_rgb_patch": f"{self.d_rgb:.4f}",
            "d_rgb_patch_grey": (f"{self.d_grey:.4f}" if self.d_grey else ""),
            "d_image": f"{self.d_image:.4f}",
            "rgb_patch_dim": 27,
            "compressed_acc": f"{r.acc:.4f}", "base_acc": f"{self.base_acc:.4f}",
            "dense_acc": f"{self.dense_acc:.4f}",
            "drop_vs_dense": f"{r.acc - self.dense_acc:.4f}",
            "drop_vs_base": f"{r.acc - self.base_acc:.4f}",
            "target_hit": int(r.hit), "params": r.params, "macs": r.macs,
            "param_reduction_pct": (f"{100*(1-r.params/self.dense_params):.2f}"
                                    if r.params else ""),
            "mac_reduction_pct": (f"{100*(1-r.macs/self.dense_macs):.2f}"
                                  if r.macs else ""),
            "dense_params": self.dense_params, "dense_macs": self.dense_macs,
            "phase": phase, "calib": self.calib_n, "ridge": self.args.ridge,
            "seconds": f"{r.seconds:.1f}", "note": note,
        }

    # -- driver ------------------------------------------------------------
    def run(self):
        self.setup()
        self.gride_profile()
        pool = FZ.ALL if not self.args.strategies else [FZ.BY_NAME[s] for s in self.args.strategies]

        log("  stage A: first conv only (the faithful input-reduction port)")
        for mlt in self.args.stage_a_mults or [1.0, 2.0, 4.0]:
            for fz in pool[:2]:
                self.evaluate_config(fz, mlt, "A", phase=1)

        log("  stage B: every layer's patch space")
        picks = self.args.tourney_mults or sorted(
            {self.mult_grid[len(self.mult_grid) // 3], self.mult_grid[-1]})
        for mlt in picks:
            for fz in pool:
                self.evaluate_config(fz, mlt, "B", phase=2)
        rank = _rank(self.runs, [f.name for f in pool])
        log("  factoriser ranking (mean gap at equal budget):")
        for n, g, w, c in rank:
            log(f"    {n:18s} gap {g:+.2f}  best-in-cell {w}/{c}")
        self.ranking = rank

        lead = [FZ.BY_NAME[rank[0][0]]] + [FZ.BY_NAME[r[0]] for r in rank[1:self.args.top_k]]
        log(f"  stage B ladder with {lead[0].name}")
        hit_idx = None
        for i, mlt in enumerate(self.mult_grid):
            r = self.evaluate_config(lead[0], mlt, "B", phase=3)
            if r and r.hit:
                hit_idx = i
                break
        if hit_idx is not None:
            for fz in lead[1:]:
                for j in range(max(0, hit_idx - 2), hit_idx + 1):
                    r = self.evaluate_config(fz, self.mult_grid[j], "B", phase=3)
                    if r and r.hit:
                        hit_idx = min(hit_idx, j)
                        break
            lo = self.mult_grid[hit_idx - 1] if hit_idx else max(0.5, self.mult_grid[0] / 2)
            hi = self.mult_grid[hit_idx]
            for _ in range(self.args.bisect_rounds):
                mid = round((lo + hi) / 2, 2)
                if mid <= lo or mid >= hi:
                    break
                got = [self.evaluate_config(fz, mid, "B", phase=4) for fz in lead[:2]]
                if any(r and r.hit for r in got):
                    hi = mid
                else:
                    lo = mid
        self.finalize()

    def finalize(self):
        keep = []
        if self.frontier:
            keep.append(("frontier", self.frontier))
        if self.best and self.best is not self.frontier:
            keep.append(("best", self.best))
        for kind, r in keep:
            comp, ranks = self.factorize(FZ.BY_NAME[r.strategy], r.multiplier, r.stage)
            acc = evaluate(comp, self.test_loader, self.dev)
            p, mm = count_params(comp), count_macs(comp, self.in_shape, self.dev)
            meta = {
                "source_checkpoint": self.ck.path.name, "model_name": self.ck.model_name,
                "dataset": self.ck.dataset, "family": self.ck.family, "bn_folded": True,
                "kind": kind, "pipeline": self.source, "stage": r.stage,
                "strategy": r.strategy, "multiplier": r.multiplier,
                "layer_order": self.layer_order, "ranks": ranks,
                "full_ranks": self.full_ranks,
                "break_even_ranks": self.break_even,
                "patch_gride_dims": self.dims,
                "d_rgb_patch": self.d_rgb, "d_image": self.d_image,
                "hidden_source": (self.base_meta or {}).get("kind"),
                "hidden_widths": (self.base_meta or {}).get("widths"),
                "dense_acc": self.dense_acc, "base_acc": self.base_acc,
                "compressed_acc": acc,
                "dense_params": self.dense_params, "params": p,
                "dense_macs": self.dense_macs, "macs": mm,
                "tol": self.tol, "calib": self.calib_n, "seed": self.args.seed,
                "input_shape": list(self.in_shape),
            }
            tag = f"{self.ck.tag}_{self.source}_{kind}"
            save_compressed(comp, meta, RESULTS, tag)
            log(f"  saved {tag}: {acc:.2f}% (dense {acc-self.dense_acc:+.2f}), "
                f"params -{100*(1-p/self.dense_params):.1f}%, "
                f"MACs -{100*(1-mm/self.dense_macs):.1f}%")
            del comp
            torch.cuda.empty_cache()

        b = self.frontier or self.best
        progress(f"\n### {datetime.now():%Y-%m-%d %H:%M} — {self.ck.tag} [{self.source}]\n\n"
                 f"- d_image {self.d_image:.2f}, d_rgb_patch {self.d_rgb:.2f} of 27\n"
                 f"- patch GRIDE per layer {[round(d,1) for d in self.dims]}\n"
                 f"- {len(self.runs)} configurations\n"
                 f"- best: {b.strategy} x{b.multiplier} stage {b.stage} -> {b.acc:.2f}% "
                 f"(dense {b.acc-self.dense_acc:+.2f}), params "
                 f"-{100*(1-b.params/self.dense_params):.1f}%, MACs "
                 f"-{100*(1-b.macs/self.dense_macs):.1f}%"
                 f"{'' if b.hit else '  **AIM NOT MET**'}\n")


def _rank(runs, names):
    cells = {}
    for r in runs:
        if r.stage != "B":
            continue
        cells.setdefault(r.multiplier, {})
        cells[r.multiplier][r.strategy] = max(cells[r.multiplier].get(r.strategy, -1e9), r.acc)
    gaps = {n: [] for n in names}
    wins = {n: 0 for n in names}
    seen = {n: 0 for n in names}
    for d in cells.values():
        top = max(d.values())
        for n, a in d.items():
            gaps[n].append(a - top)
            seen[n] += 1
            wins[n] += int(a >= top - 1e-9)
    out = [(n, float(np.mean(g)), wins[n], seen[n]) for n, g in gaps.items() if seen[n]]
    return sorted(out, key=lambda t: -t[1])


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--source", default="hidden", choices=["hidden", "dense"],
                    help="'hidden' factorises on top of hidden_layer_compression's artefact")
    ap.add_argument("--hidden-kind", default="frontier", choices=["frontier", "best"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--calib", type=int, default=1024)
    ap.add_argument("--od-factor", type=float, default=4.0)
    ap.add_argument("--patch-budget", type=int, default=60000)
    ap.add_argument("--gride-points", type=int, default=2500)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--stage-a-mults", nargs="*", type=float, default=None)
    ap.add_argument("--tourney-mults", nargs="*", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--bisect-rounds", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    found = discover()
    if a.models:
        sel = []
        for pat in a.models:
            m = [(p, n) for p, n in found if p.stem == pat or p.stem.startswith(pat + "_")]
            if not m:
                raise SystemExit(f"--models {pat!r} matched nothing")
            sel += m
        found = sorted(set(sel), key=lambda t: t[0].stat().st_size)
    elif not a.all:
        found = found[-1:]
    else:
        found = found[::-1]        # smallest first: results land early, ImageNet32 last

    for path, mname in found:
        try:
            ck = load_checkpoint(path, mname, a.device)
            base, meta = None, None
            if a.source == "hidden":
                sp = HIDDEN_RESULTS / f"{path.stem}_{a.hidden_kind}.arch.json"
                if not sp.exists():
                    log(f"!! {path.stem}: no hidden-compressed artefact at {sp.name}; "
                        f"run hidden_layer_compression first or pass --source dense")
                    continue
                base, meta = load_compressed(sp, a.device)
            Session(ck, a, base, meta).run()
        except Exception as exc:
            log(f"!! {path.stem} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
