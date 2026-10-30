#!/usr/bin/env python3
"""One session, many configurations.

The dense targets and the evaluation set are the expensive part and neither
depends on the budget, so a `Session` builds them once and every configuration
reuses them. Rows land in a CSV whose header is checked on open, because a
header that has drifted from the writer's field order shifts every column past
the difference and turns `hit` into somebody else's value.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import core as C

FIELDS = [
    "timestamp", "tag", "strategy", "alpha", "seed", "cloud", "target",
    "sequential", "ridge", "calib", "rows", "layers",
    "acc", "dense_acc", "delta", "hit",
    "params_total", "params_encoder", "params_ffn",
    "red_total_pct", "red_encoder_pct", "red_ffn_pct",
    "widths", "dims", "seconds", "note",
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def csv_append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    if not fresh:
        with path.open() as fh:
            head = next(csv.reader(fh), [])
        if head and head != FIELDS:
            bak = path.with_suffix(f".csv.schema{len(head)}")
            path.rename(bak)
            print(f"[csv] header mismatch; previous file moved to {bak.name}")
            fresh = True
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


class Session:
    """Model, evaluation set, and one dense-target cache per (seed, calib, rows)."""

    TOL = 2.5

    def __init__(self, calib: int = 2048, rows_budget: int = 65536,
                 cloud: str = "neuron", max_len: int = 128):
        self.dev = C.device()
        self.model = C.load_model(self.dev)
        self.calib, self.rows_budget, self.cloud = calib, rows_budget, cloud
        self.max_len = max_len
        self.eval_ids, self.eval_mask, self.eval_lab = C.eval_set(max_len)
        self.dense_acc = C.accuracy(self.model, self.eval_ids, self.eval_mask,
                                    self.eval_lab)
        self.base = C.param_counts(self.model)
        self._cache: dict[int, tuple] = {}
        g = json.loads((C.RESULTS / "phase2_gride.json").read_text())
        self.gride = g
        self.dims_by_cloud = {
            c: [g["clouds"][c][str(i)]["0"] for i in range(C.N_LAYERS)]
            for c in g["clouds"]
        }
        log(f"dense {self.dense_acc:.2f}%  window >= {self.dense_acc - self.TOL:.2f}%")

    def dims(self, cloud=None, seed=0):
        cloud = cloud or self.cloud
        g = self.gride["clouds"][cloud]
        return [g[str(i)].get(str(seed), g[str(i)]["0"]) for i in range(C.N_LAYERS)]

    def cache(self, seed: int):
        if seed not in self._cache:
            if len(self._cache) >= 2:                 # keep GPU memory bounded
                self._cache.pop(next(iter(self._cache)))
                torch.cuda.empty_cache()
            rows = C.Rows(*C.calibration_set(self.calib, seed, self.max_len),
                          budget=self.rows_budget, seed=seed)
            self._cache[seed] = (rows, C.dense_targets(self.model, rows))
        return self._cache[seed]

    def widths(self, alpha: float, cloud=None, seed=0):
        return C.widths_from_gride(self.dims(cloud, seed), alpha,
                                   [C.HIDDEN_DIM] * C.N_LAYERS)

    def run(self, strategy: str, alpha: float, *, out: Path, tag: str = "",
            seed: int = 0, cloud=None, target: str = "dense",
            sequential: bool = True, ridge: float = 1e-4, layers=None,
            note: str = "", save: Path | None = None) -> dict:
        t0 = time.time()
        cloud = cloud or self.cloud
        rows, cache = self.cache(seed)
        w = self.widths(alpha, cloud, seed)
        if layers is not None:
            w = [w[i] if i in set(layers) else C.HIDDEN_DIM
                 for i in range(C.N_LAYERS)]
        comp, _ = C.compress(self.model, rows, w, strategy, cache, ridge=ridge,
                             seed=seed, target=target, sequential=sequential,
                             layers=layers)
        acc = C.accuracy(comp, self.eval_ids, self.eval_mask, self.eval_lab)
        pc = C.param_counts(comp)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tag": tag, "strategy": strategy, "alpha": alpha, "seed": seed,
            "cloud": cloud, "target": target, "sequential": int(sequential),
            "ridge": ridge, "calib": self.calib, "rows": rows.n,
            "layers": "all" if layers is None else json.dumps(sorted(layers)),
            "acc": round(acc, 4), "dense_acc": round(self.dense_acc, 4),
            "delta": round(acc - self.dense_acc, 4),
            "hit": int(abs(acc - self.dense_acc) <= self.TOL),
            "params_total": pc["total"], "params_encoder": pc["encoder"],
            "params_ffn": pc["ffn"],
            "red_total_pct": round(100 * (1 - pc["total"] / self.base["total"]), 4),
            "red_encoder_pct": round(100 * (1 - pc["encoder"] / self.base["encoder"]), 4),
            "red_ffn_pct": round(100 * (1 - pc["ffn"] / self.base["ffn"]), 4),
            "widths": json.dumps(w),
            "dims": json.dumps([round(d, 3) for d in self.dims(cloud, seed)]),
            "seconds": round(time.time() - t0, 1), "note": note,
        }
        csv_append(out, row)
        log(f"  {strategy:18s} a={alpha:<7g} seed={seed} "
            f"acc={acc:6.2f} ({row['delta']:+6.2f}) "
            f"red tot/enc/ffn = {row['red_total_pct']:5.1f}/"
            f"{row['red_encoder_pct']:5.1f}/{row['red_ffn_pct']:5.1f}%  "
            f"{row['seconds']:.0f}s")
        if save is not None:
            C.save_artifact(comp, save, row)
        del comp
        torch.cuda.empty_cache()
        return row


def bisect(session: Session, strategy: str, lo: float, hi: float, *, out: Path,
           tag: str, depth: int = 5, **kw) -> dict | None:
    """`lo` misses the window, `hi` makes it. Narrow until the widths stop
    changing, which is the real resolution limit: two multipliers that round to
    the same six widths are the same configuration."""
    best = None
    for _ in range(depth):
        mid = 0.5 * (lo + hi)
        if session.widths(mid, kw.get("cloud")) in (session.widths(lo, kw.get("cloud")),
                                                    session.widths(hi, kw.get("cloud"))):
            break
        r = session.run(strategy, mid, out=out, tag=tag, **kw)
        if r["hit"]:
            hi, best = mid, r
        else:
            lo = mid
    return best
