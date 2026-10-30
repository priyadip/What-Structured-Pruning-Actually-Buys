#!/usr/bin/env python3
"""Autonomous zero-shot channel compression for VGG and ResNet.

AIM
---
Take a converged VGG/ResNet checkpoint from `trainer/checkpoint/` and shrink its
convolutional layers with **no fine-tuning at all** -- zero backward passes, zero
`optimizer.step()` -- landing within +/- TOL points of the dense test accuracy.

GOLDEN CONSTRAINT (non-negotiable)
----------------------------------
Every compressed layer's channel count comes **only** from that layer's GRIDE
intrinsic dimension:

        M_l = round(multiplier * d_l)

`d_l` is measured at runtime by `intrinsic_rank.id_estimator.gride` on the
layer's *effective neuron cloud* -- the filters flattened to `R^(C_in*k*k)` and
projected onto the column space of the unfolded input, i.e. the part of each
filter the data can actually see. No width comes from a PCA variance threshold,
a sparsity ratio, or a hand-set number, and `Strategy.apply` is asserted to
return exactly `target_m` channels.

This is the paper's thesis carried over unchanged: Zador's quantisation bound
`E_m ~ m^(-1/d)` makes the achievable width a function of the *intrinsic*
dimension, so `d` is what sets the budget.

WHAT CHANGES FOR A CNN, AND WHAT DOES NOT
-----------------------------------------
Unchanged: the neuron cloud, the GRIDE rule, the select-and-correct compression,
the dense-target least-squares refit with a free intercept, the sequential
error-correcting sweep.

Changed: a conv layer is a linear map on *patches*, so the design matrix is the
unfolded input and its columns come in groups of `k*k`; and a ResNet's residual
stream is shared by a whole stage, so only the **internal** convs of a block are
compressible unit-by-unit (see `common/surgery.py`, and `--stream` for the
stage-level extension).

PHASES
------
0  discovery, dense accuracy, calibration sizing
1  GRIDE profile of every unit (three candidate clouds, all logged)
2  strategy tournament at a few budgets -- which family wins on CNNs
3  full GRIDE-multiplier ladder with the top strategies
4  frontier bisection between the last miss and the first hit
5  adaptive research round(s) from `novel_strategies/*.py` if the aim is unmet
6  save the artefact + per-model report

    python3 run_cnn_compression.py --models cifar10_vgg11_bn
    python3 run_cnn_compression.py --all
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent
PROJECT_ROOT = EXT_ROOT.parent
for _p in (str(PROJECT_ROOT), str(EXT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import gride_conv as GC          # noqa: E402
from common import surgery as SG             # noqa: E402
from common.discovery import (               # noqa: E402
    calibration_batch, discover, evaluate, load_checkpoint, loaders_for, split_stem)
from common.foldbn import fold_bn            # noqa: E402
from common.rebuild import (                 # noqa: E402
    count_macs, count_params, save_compressed)

from common.jacobian_metric import pullback_metrics  # noqa: E402

import strategies as ST                      # noqa: E402

RESULTS = HERE / "results"
CSV_PATH = RESULTS / "compression_results_cnn.csv"
PROGRESS = HERE / "PROGRESS.md"
NOVEL_DIR = HERE / "novel_strategies"

CSV_FIELDS = [
    "timestamp", "model_name", "dataset", "family", "strategy", "multiplier",
    "compressed_acc", "dense_acc", "acc_drop", "target_hit",
    "params", "param_reduction_pct", "macs", "mac_reduction_pct",
    "widths", "orig_widths", "gride_dims", "id_space", "phase", "stage",
    "n_units", "calib", "patch_budget", "ridge", "seconds", "note",
]


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def csv_append(row: dict):
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


def progress(text: str):
    with PROGRESS.open("a") as f:
        f.write(text + "\n")


@dataclass
class Run:
    strategy: str
    multiplier: float
    widths: list[int]
    acc: float
    drop: float
    params: int
    macs: int
    hit: bool
    phase: int
    seconds: float
    stage: str = "search"
    targets: list[int] | None = None      # explicit widths, when not from a scalar mult
    model: object = field(default=None, repr=False)


# ===========================================================================
class Session:
    def __init__(self, ck, args):
        self.ck, self.args = ck, args
        self.dev = args.device
        self.tol = args.tol
        self.runs: list[Run] = []
        self.novel: list[ST.Strategy] = []
        self.best: Run | None = None
        self.frontier: Run | None = None

    # -- phase 0 ---------------------------------------------------------
    def setup(self):
        a = self.args
        if a.include_novel:            # make novel strategies nameable in --strategies
            self.load_novel()
        self.train_loader, self.test_loader = loaders_for(self.ck, a.batch_size, a.workers)
        self.dense_acc = evaluate(self.ck.model, self.test_loader, self.dev)
        self.folded = fold_bn(self.ck.model, self.ck.family).to(self.dev)
        self.folded_acc = evaluate(self.folded, self.test_loader, self.dev)
        # Folding is exact in exact arithmetic; in float32 it can still flip the
        # argmax on a near-tie, which on ImageNet32's 1000 classes is worth about
        # a tenth of a point. Anything larger means the fold is wrong, not noisy.
        if abs(self.folded_acc - self.dense_acc) > 0.15:
            raise RuntimeError(f"BN folding changed accuracy: {self.dense_acc:.2f} -> "
                               f"{self.folded_acc:.2f}; refusing to continue.")

        probe = calibration_batch(self.train_loader, 8, self.dev, seed=a.seed)
        self.in_shape = tuple(probe.shape[1:])
        need, info = SG.plan_calibration(self.folded, self.ck.family, probe, a.od_factor)
        self.calib_n = max(a.calib, need)
        self.unit_info = info
        self.X = calibration_batch(self.train_loader, self.calib_n, self.dev, seed=a.seed)
        self.orig_widths = [i["C"] for i in info]

        self.dense_params = self.ck.n_params
        self.folded_params = count_params(self.folded)
        self.dense_macs = count_macs(self.folded, self.in_shape, self.dev)

        log(f"AIM: {self.ck.tag} {self.dense_acc:.2f}% -> within "
            f"[{self.dense_acc - self.tol:.2f}, {self.dense_acc + self.tol:.2f}] "
            f"with zero fine-tuning")
        log(f"  {len(info)} compressible units, widths {self.orig_widths}")
        log(f"  params {self.dense_params:,} (folded {self.folded_params:,}), "
            f"MACs {self.dense_macs/1e6:.1f}M, calib {self.calib_n} images")

    # -- phase 1: GRIDE --------------------------------------------------
    def gride_profile(self):
        """One walk, no compression: measure each unit's intrinsic dimension."""
        t0 = time.time()
        dims: dict[str, dict[str, float]] = {}
        order: list[str] = []

        def probe(ctx):
            U, G = ctx.producer_patches(self.args.gride_patches)
            clouds = GC.layer_clouds(ctx.filters_flat(), G, U, seed=self.args.seed)
            d = {}
            for k, cloud in clouds.items():
                try:
                    d[k] = GC.gride_id(cloud, seed=self.args.seed)
                except Exception as exc:
                    raise RuntimeError(
                        f"GRIDE failed on unit {ctx.name} cloud '{k}': {exc}. "
                        "Widths may not be set by any other means.") from exc
            dims[ctx.name] = d
            order.append(ctx.name)
            del U, G
            return ctx.A

        walker = SG.Walker(self.folded, SG.clone_eval(self.folded), self.ck.family,
                           self.X, self.args.patch_budget, self.args.ridge, self.args.seed)
        walker.walk(probe)
        self.dims_all = dims
        self.unit_order = order
        self.dims = [dims[n][self.args.id_space] for n in order]
        log(f"  GRIDE ({self.args.id_space}) in {time.time()-t0:.0f}s: "
            + ", ".join(f"{n.split('.')[-1] if self.ck.family=='vgg' else n}={d:.1f}"
                        for n, d in zip(order, self.dims)))
        self.mult_grid = GC.multiplier_grid(self.dims, self.orig_widths)
        log(f"  multiplier ladder: {self.mult_grid}")

    # -- the compression itself ------------------------------------------
    def compress(self, strategy: ST.Strategy, multiplier: float,
                 targets: list[int] | None = None):
        if targets is None:
            targets = GC.widths_from_gride(self.dims, multiplier, self.orig_widths)
        comp = SG.clone_eval(self.folded)
        walker = SG.Walker(self.folded, comp, self.ck.family, self.X,
                           self.args.patch_budget, self.args.ridge, self.args.seed)
        widths: list[int] = []
        it = iter(range(len(targets)))

        met = self.unit_metrics() if getattr(strategy, "needs_metric", False) else None

        def decide(ctx):
            i = next(it)
            m = int(targets[i])
            if m >= ctx.C:                                   # nothing to do at this budget
                widths.append(ctx.C)
                return ctx.A
            ctx.require_overdetermined(2.0)
            if met is not None:
                ctx.metric = met.get(ctx.cons_path)
            W, b, Vf, c = strategy.apply(ctx, m)
            assert W.shape[0] == m, f"{strategy.name} returned {W.shape[0]} != {m}"
            assert Vf.shape[1] == m * ctx.gs, f"{strategy.name} consumer width mismatch"
            SG.set_module(comp, ctx.name, SG.new_producer(ctx.prod, W, b))
            SG.set_module(comp, ctx.cons_path, SG.new_consumer(ctx.consumer, Vf, c, m))
            widths.append(m)
            return ctx.acts_for(W, b)

        walker.walk(decide)
        return comp, widths

    def evaluate_config(self, strategy, multiplier, phase, stage="search", keep=False,
                        targets=None):
        t0 = time.time()
        try:
            comp, widths = self.compress(strategy, multiplier, targets)
        except Exception as exc:
            log(f"    {strategy.name} x{multiplier}: FAILED {type(exc).__name__}: {exc}")
            if self.args.debug:
                traceback.print_exc()
            csv_append(self._row(strategy.name, multiplier, [], float("nan"), phase,
                                 stage, 0, 0, time.time() - t0,
                                 note=f"{type(exc).__name__}: {exc}"))
            return None
        acc = evaluate(comp, self.test_loader, self.dev)
        p, mm = count_params(comp), count_macs(comp, self.in_shape, self.dev)
        r = Run(strategy.name, multiplier, widths, acc, acc - self.dense_acc, p, mm,
                abs(acc - self.dense_acc) <= self.tol, phase, time.time() - t0,
                stage, list(targets) if targets is not None else None,
                comp if keep else None)
        self.runs.append(r)
        csv_append(self._row(strategy.name, multiplier, widths, acc, phase, stage,
                             p, mm, r.seconds))
        log(f"    {strategy.name:22s} x{multiplier:<6.2f} {acc:6.2f}%  "
            f"({r.drop:+.2f})  params {100*(p/self.dense_params-1):+5.1f}%  "
            f"MACs {100*(mm/self.dense_macs-1):+5.1f}%  {'HIT' if r.hit else ''}")
        if self.best is None or acc > self.best.acc:
            self.best = r
        if r.hit and (self.frontier is None or p < self.frontier.params):
            self.frontier = r
        if not keep:
            del comp
            torch.cuda.empty_cache()
        return r

    def _row(self, sname, mult, widths, acc, phase, stage, params, macs, secs, note=""):
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.ck.tag, "dataset": self.ck.dataset,
            "family": self.ck.family, "strategy": sname, "multiplier": mult,
            "compressed_acc": f"{acc:.4f}", "dense_acc": f"{self.dense_acc:.4f}",
            "acc_drop": f"{acc - self.dense_acc:.4f}",
            "target_hit": int(abs(acc - self.dense_acc) <= self.tol) if acc == acc else 0,
            "params": params, "macs": macs,
            "param_reduction_pct": f"{100*(1-params/self.dense_params):.2f}" if params else "",
            "mac_reduction_pct": f"{100*(1-macs/self.dense_macs):.2f}" if macs else "",
            "widths": json.dumps(widths), "orig_widths": json.dumps(self.orig_widths),
            "gride_dims": json.dumps([round(d, 3) for d in self.dims]),
            "id_space": self.args.id_space, "phase": phase, "stage": stage,
            "n_units": len(self.orig_widths), "calib": self.calib_n,
            "patch_budget": self.args.patch_budget, "ridge": self.args.ridge,
            "seconds": f"{secs:.1f}", "note": note,
        }

    # -- phase 2: which family wins on CNNs ------------------------------
    def tournament(self):
        grid = self.mult_grid
        picks = sorted({grid[len(grid) // 3], grid[2 * len(grid) // 3], grid[-1]})
        if self.args.tourney_mults:
            picks = self.args.tourney_mults
        log(f"  phase 2: strategy tournament at multipliers {picks}")
        pool = ST.ALL if not self.args.strategies else [ST.BY_NAME[s] for s in self.args.strategies]
        for mlt in picks:
            for s in pool:
                self.evaluate_config(s, mlt, phase=2)
        rank = _rank(self.runs, [s.name for s in pool])
        log("  ranking (mean gap to the best strategy at equal budget):")
        for name, gap, wins, n in rank:
            log(f"    {name:22s} gap {gap:+.2f}  best-in-cell {wins}/{n}")
        self.ranking = rank
        return [ST.BY_NAME[r[0]] for r in rank[: self.args.top_k]]

    # -- phase 3/4: ladder + bisection -----------------------------------
    def ladder(self, pool):
        """Full ladder for the top strategy, targeted probes for the rest.

        The whole accuracy-vs-budget curve is worth having for one strategy --
        it is the evidence for the `m ~ d` scaling and it costs one sweep. For
        the runners-up only the question "does it clear the window sooner?"
        matters, so they are probed at the rungs around the leader's hit point.
        """
        lead, rest = pool[0], pool[1:]
        log(f"  phase 3: full GRIDE ladder with {lead.name}")
        hit_idx = None
        for i, mlt in enumerate(self.mult_grid):
            r = self.evaluate_config(lead, mlt, phase=3)
            if r and r.hit:
                hit_idx = i
                break
        if hit_idx is None:
            return None
        for s in rest:
            for j in range(max(0, hit_idx - 2), hit_idx + 1):
                r = self.evaluate_config(s, self.mult_grid[j], phase=3)
                if r and r.hit:
                    hit_idx = min(hit_idx, j)
                    break
        return self.mult_grid[hit_idx]

    # -- phase 6: Zador-derived per-layer allocation ----------------------
    def amplification_weights(self):
        """`w_l = tr(Omega_l Sigma_l)` -- what distortion at layer `l` costs at
        the logits.

        `Omega_l` is the metric the frozen tail induces on layer `l`'s output
        (closed-form backward recursion over forward quantities -- no autograd,
        no backward pass); `Sigma_l` is that layer's own output covariance. The
        product is the expected squared logit error from replacing the layer's
        output with noise of matching scale, which is exactly the "how much does
        this layer matter" term the plain Zador allocation is missing.
        """
        if getattr(self, "_w", None) is not None:
            return self._w
        Om = pullback_metrics(self.folded, self.ck.family, self.X, 128)
        sig: dict[str, torch.Tensor] = {}

        def probe(ctx):
            z = ctx.prod(ctx.h[:512]) if ctx.h.shape[0] > 512 else ctx.prod(ctx.h)
            zf = (z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]) if z.dim() == 4
                  else z).double()
            sig[ctx.name] = zf.t() @ zf / zf.shape[0]
            del z, zf
            return ctx.A

        SG.Walker(self.folded, SG.clone_eval(self.folded), self.ck.family, self.X,
                  self.args.patch_budget, self.args.ridge, self.args.seed).walk(probe)
        self._w = [float((Om[n] * sig[n]).sum()) if n in Om else 1.0
                   for n in self.unit_order]
        return self._w

    def unit_metrics(self):
        """`Omega` at each consumer's output, computed once per model."""
        if getattr(self, "_om", None) is None:
            log("  computing logit-pullback metrics (forward-only recursion)")
            self._om = pullback_metrics(self.folded, self.ck.family, self.X, 128)
        return self._om

    def zador_sweep(self, pool, cost_key: str):
        """Same GRIDE dimensions, same bound -- but the budget split across
        layers is solved for instead of assumed uniform."""
        cost = [i[cost_key] for i in self.unit_info]
        ref = self.frontier or self.best
        if ref is None:
            return
        ref_cost = sum(c * w for c, w in zip(cost, ref.widths))
        variants = [("flat", None)]
        try:
            variants.append(("amp", self.amplification_weights()))
        except Exception as exc:
            log(f"    amplification weights unavailable: {type(exc).__name__}: {exc}")
        log(f"  phase 6: Zador allocation on {cost_key} "
            f"(matching the x{ref.multiplier} frontier's budget)")

        for wname, wts in variants:
            lo, hi = 1e-30, 1e30                             # bracket K by bisection
            for _ in range(80):
                K = (lo * hi) ** 0.5
                w, _m = GC.zador_widths(self.dims, K, self.orig_widths, cost, wts)
                if sum(c * x for c, x in zip(cost, w)) > ref_cost:
                    hi = K
                else:
                    lo = K
            base_K = (lo * hi) ** 0.5
            for scale in (1.0, 4.0, 16.0):
                widths, mults = GC.zador_widths(self.dims, base_K * scale,
                                                self.orig_widths, cost, wts)
                if widths == self.orig_widths:
                    continue
                # Label with the *realised* multiplier -- mean(width/d) -- not the
                # unclipped one. A layer pinned at its full width reports a nominal
                # multiplier in the thousands, which makes the column meaningless.
                eff = float(np.mean([w / max(d, 1e-9)
                                     for w, d in zip(widths, self.dims)]))
                self.evaluate_config(pool[0], round(eff, 3), phase=6,
                                     stage=f"zador_{wname}_{cost_key}", targets=widths)

    def bisect(self, pool, lo_miss: float, hi_hit: float, rounds: int = 3):
        """The ladder is coarse (12 -> 16 -> 20); the true frontier sits between
        its rungs. Bisect on the multiplier with the strategies that were closest
        at the failing rung, not only those that hit -- at `lo_miss` none hit."""
        log(f"  phase 4: bisecting the frontier in ({lo_miss}, {hi_hit})")
        for _ in range(rounds):
            mid = round((lo_miss + hi_hit) / 2, 2)
            if mid <= lo_miss or mid >= hi_hit:
                break
            got = [self.evaluate_config(s, mid, phase=4) for s in pool]
            if any(r and r.hit for r in got):
                hi_hit = mid
            else:
                lo_miss = mid
        return hi_hit

    # -- phase 5: adaptive ------------------------------------------------
    def load_novel(self) -> list[ST.Strategy]:
        out = []
        if not NOVEL_DIR.exists():
            return out
        for p in sorted(NOVEL_DIR.glob("*.py")):
            if p.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"novel_{p.stem}", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:
                log(f"  could not load {p.name}: {exc}")
                continue
            got = getattr(mod, "STRATEGIES", [])
            for s in got:
                ST.BY_NAME[s.name] = s
            out += got
            log(f"  loaded {len(got)} strategies from {p.name}")
        return out

    def write_request(self):
        p = RESULTS / f"adaptive_request_{self.ck.tag}.txt"
        b = self.best
        worst = sorted(zip(self.unit_order, self.dims, self.orig_widths),
                       key=lambda t: t[2] / max(t[1], 1e-9), reverse=True)[:5]
        p.write_text(
            f"ADAPTIVE RESEARCH REQUEST -- {self.ck.tag}\n"
            f"{'='*70}\n"
            f"dense accuracy      : {self.dense_acc:.2f}%\n"
            f"target window       : +/- {self.tol}\n"
            f"best so far         : {b.acc:.2f}% ({b.drop:+.2f}) via {b.strategy} "
            f"x{b.multiplier}\n"
            f"params at best      : {b.params:,} of {self.dense_params:,}\n"
            f"GRIDE dims          : "
            + ", ".join(f"{n}={d:.1f}/{w}" for n, d, w in
                        zip(self.unit_order, self.dims, self.orig_widths)) + "\n"
            f"most over-parameterised units (N/d): "
            + ", ".join(f"{n}({w}/{d:.1f})" for n, d, w in worst) + "\n\n"
            f"Strategy ranking:\n"
            + "\n".join(f"  {n:24s} gap {g:+.2f}  wins {w}/{c}"
                        for n, g, w, c in getattr(self, "ranking", [])) + "\n\n"
            "Drop a module into novel_strategies/ exporting STRATEGIES = [...]\n"
            "of common.surgery-compatible Strategy objects. Constraints: exactly\n"
            "target_m output channels, closed form only, no backward passes.\n")
        return p

    # -- driver ------------------------------------------------------------
    def run(self):
        self.setup()
        self.gride_profile()
        pool = self.tournament()
        hit_at = self.ladder(pool)

        if hit_at is not None:
            idx = self.mult_grid.index(hit_at)
            lo = self.mult_grid[idx - 1] if idx else max(1.0, hit_at / 2)
            if idx:
                near = sorted([r for r in self.runs if r.multiplier == lo],
                              key=lambda r: -r.acc)[: self.args.top_k]
                self.bisect([ST.BY_NAME[r.strategy] for r in near] or pool, lo, hit_at)
        else:
            log("  aim unmet on the seed ladder -- entering adaptive research")
            req = self.write_request()
            log(f"  wrote {req}")
            novel = self.load_novel()
            for rnd in range(self.args.adaptive_rounds):
                if not novel:
                    break
                log(f"  adaptive round {rnd+1}: {[s.name for s in novel]}")
                for mlt in self.mult_grid:
                    got = [self.evaluate_config(s, mlt, phase=5) for s in novel]
                    if any(r and r.hit for r in got):
                        hit_at = mlt
                        break
                if hit_at is not None:
                    break
                novel = []

        if self.args.allocate:
            for key in self.args.allocate:
                self.zador_sweep(pool, key)

        self.finalize()

    # -- artefacts ---------------------------------------------------------
    def finalize(self):
        """Re-run the two configurations worth keeping and save them from that
        run's own tensors. In the MLP study writing the globally-best weights
        under the frontier run's spec was a real bug that only a from-disk
        rebuild caught, so each artefact is produced by its own compression."""
        keep = []
        if self.frontier:
            keep.append(("frontier", self.frontier))
        if self.best and (not self.frontier or self.best is not self.frontier):
            keep.append(("best", self.best))
        for kind, r in keep:
            comp, widths = self.compress(ST.BY_NAME[r.strategy], r.multiplier, r.targets)
            acc = evaluate(comp, self.test_loader, self.dev)
            p, mm = count_params(comp), count_macs(comp, self.in_shape, self.dev)
            meta = {
                "source_checkpoint": self.ck.path.name, "model_name": self.ck.model_name,
                "dataset": self.ck.dataset, "family": self.ck.family, "bn_folded": True,
                "kind": kind, "strategy": r.strategy, "multiplier": r.multiplier,
                "allocation": r.stage, "explicit_targets": r.targets,
                "unit_order": self.unit_order, "widths": widths,
                "original_widths": self.orig_widths,
                "gride_dims": {n: self.dims_all[n] for n in self.unit_order},
                "id_space": self.args.id_space,
                "dense_acc": self.dense_acc, "compressed_acc": acc,
                "dense_params": self.dense_params, "params": p,
                "dense_macs": self.dense_macs, "macs": mm,
                "tol": self.tol, "calib": self.calib_n, "seed": self.args.seed,
                "input_shape": list(self.in_shape),
            }
            tag = f"{self.ck.tag}_{kind}"
            save_compressed(comp, meta, RESULTS, tag)
            csv_append(self._row(r.strategy, r.multiplier, widths, acc, r.phase,
                                 kind, p, mm, 0.0, note="saved artefact"))
            log(f"  saved {tag}: {acc:.2f}% ({acc-self.dense_acc:+.2f}), "
                f"params -{100*(1-p/self.dense_params):.1f}%, "
                f"MACs -{100*(1-mm/self.dense_macs):.1f}%")
            del comp
            torch.cuda.empty_cache()

        b = self.frontier or self.best
        progress(f"\n### {datetime.now():%Y-%m-%d %H:%M} — {self.ck.tag}\n\n"
                 f"- dense {self.dense_acc:.2f}%, {len(self.orig_widths)} units, "
                 f"GRIDE({self.args.id_space}) "
                 f"{[round(d,1) for d in self.dims]}\n"
                 f"- {len(self.runs)} configurations evaluated\n"
                 f"- best in window: {b.strategy} x{b.multiplier} -> {b.acc:.2f}% "
                 f"({b.drop:+.2f}), params -{100*(1-b.params/self.dense_params):.1f}%, "
                 f"MACs -{100*(1-b.macs/self.dense_macs):.1f}%"
                 f"{'' if b.hit else '  **AIM NOT MET**'}\n")


def _rank(runs, names):
    """Mean accuracy gap to the best strategy at the same multiplier."""
    cells: dict[float, dict[str, float]] = {}
    for r in runs:
        if r.phase != 2 or r.acc != r.acc:
            continue
        cells.setdefault(r.multiplier, {})
        cells[r.multiplier][r.strategy] = max(
            cells[r.multiplier].get(r.strategy, -1e9), r.acc)
    gaps, wins, seen = {n: [] for n in names}, {n: 0 for n in names}, {n: 0 for n in names}
    for mlt, d in cells.items():
        top = max(d.values())
        for n, a in d.items():
            gaps[n].append(a - top)
            seen[n] += 1
            wins[n] += int(a >= top - 1e-9)
    out = [(n, float(np.mean(g)) if g else -1e9, wins[n], seen[n])
           for n, g in gaps.items() if seen[n]]
    return sorted(out, key=lambda t: -t[1])


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="checkpoint stems; matched on '_' boundaries, not substrings")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--calib", type=int, default=1024,
                    help="floor; raised automatically to keep every unit over-determined")
    ap.add_argument("--od-factor", type=float, default=4.0)
    ap.add_argument("--patch-budget", type=int, default=60000)
    ap.add_argument("--gride-patches", type=int, default=20000)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id-space", default="filter_proj",
                    choices=["filter_proj", "neuron", "patch"])
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--tourney-mults", nargs="*", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--adaptive-rounds", type=int, default=3)
    ap.add_argument("--allocate", nargs="*", default=["macs_per_ch"],
                    choices=["macs_per_ch", "params_per_ch"],
                    help="Zador-derived per-layer budget allocation; '' to disable")
    ap.add_argument("--results-dir", default=None,
                    help="write CSV and artefacts here instead of ./results "
                         "(keeps a side experiment from overwriting a sweep)")
    ap.add_argument("--include-novel", action="store_true",
                    help="load novel_strategies/ up front so they can enter the "
                         "tournament, instead of only after a missed target")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if a.results_dir:
        global RESULTS, CSV_PATH
        RESULTS = Path(a.results_dir)
        CSV_PATH = RESULTS / CSV_PATH.name

    found = discover()
    if a.models:
        sel = []
        for pat in a.models:
            m = [(p, n) for p, n in found if p.stem == pat or p.stem.startswith(pat + "_")]
            if not m:
                raise SystemExit(f"--models {pat!r} matched nothing. Available: "
                                 f"{[p.stem for p, _ in found]}")
            sel += m
        found = sorted(set(sel), key=lambda t: t[0].stat().st_size)
    elif not a.all:
        found = found[-1:]                                   # smallest, for a quick run
    else:
        found = found[::-1]        # smallest first: results land early, ImageNet32 last

    log(f"{len(found)} checkpoint(s): {[p.stem for p, _ in found]}")
    for path, mname in found:
        try:
            ck = load_checkpoint(path, mname, a.device)
            Session(ck, a).run()
        except Exception as exc:
            log(f"!! {path.stem} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
