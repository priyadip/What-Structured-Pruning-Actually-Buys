#!/usr/bin/env python3
"""Autonomous input reduction + first-layer refit for already-compressed MLPs.

AIM
---
The hidden layers were already compressed at a GRIDE budget (`../compression_mlp/`).
Now shrink the **input dimension** too: learn an embedding `E: R^D -> R^r` and a
new first layer `A` such that the *fully* compressed model — reduced input plus
compressed hidden layers — stays as close as possible to the **original dense**
accuracy, with the hidden layers **frozen**.

THE RULE
--------
`r` comes only from the GRIDE intrinsic dimension of the **raw input data**,
measured at runtime: `r = round(multiplier * d_input)`.

WHAT IS OPTIMISED
-----------------
Only the landmark coordinates `Z` and the linear map `A` (whose optimum is a
closed-form function of `Z`). **No network weight receives a gradient.** The only
thing read from each compressed checkpoint is its first-layer `(W_1, b_1)`; the
rest is frozen and used solely to score a candidate.

METHOD
------
Regularised srGW-MDS on `K` landmarks,

    L(Z) = stress(Z ; D_geo) + lambda * min_{A,c} || X_L W_1^T - Z A^T - 1 c^T ||_F^2

then an out-of-sample extension to the full dataset. See `core/` for the pieces
and `notes/` for why each one is shaped the way it is.

HONEST ACCOUNTING
-----------------
A *linear* encoder folds — `A(Px) = (AP)x` — so it is exactly a rank-`r`
factorisation of `W_1`, and `baselines.optimal_linear_encoder` is the provable
ceiling for that entire family. A *distance-based* encoder can beat it but costs
`K x D` flops per sample, which is not a parameter saving. Both numbers are
logged for every run; neither is quietly omitted.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.baselines import BASELINES, build_baseline  # noqa: E402
from core.geometry import landmark_dissimilarity  # noqa: E402
from core.gride import gride_id  # noqa: E402
from core.landmarks import STRATEGIES, USES_LABELS, select_landmarks  # noqa: E402
from core.model_io import (  # noqa: E402
    accuracy_on_codes, build_reduced_model, discover_compressed, fit_A_from_codes,
    load_compressed, load_flat_data, save_reduced,
)
from core.oos import OOS_METHODS, build_encoder  # noqa: E402
from core.srgw_mds import fit_embedding  # noqa: E402

RESULTS = HERE / "results"
CSV_PATH = RESULTS / "input_compression_results.csv"
PROGRESS = HERE / "PROGRESS.md"
REQUEST = HERE / "adaptive_research_request.txt"
SUMMARY = HERE / "input_compression_summary.md"
NOVEL_DIR = HERE / "novel_strategies"

MULT_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
LAMBDA_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
K_GRID = [55, 100, 200, 500]

CSV_FIELDS = [
    "timestamp", "model_name", "r", "lambda", "landmark_strategy", "K",
    "out_of_sample_method", "mds_stress", "recon_loss", "alpha",
    "compressed_acc", "target_hit",
    # extras — honest accounting and provenance
    "dataset", "stage", "encoder_family", "multiplier", "r_input_gride",
    "dense_acc", "hidden_only_acc", "drop_vs_dense", "drop_vs_hidden",
    "input_dim", "encoder_params", "encoder_flops", "folds_into_layer1",
    "total_params", "cwf", "n_fit", "seconds",
]

log = logging.getLogger("inputc")


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class Run:
    model: str
    dataset: str
    r: float
    lam: float
    strategy: str
    K: int
    oos: str
    family: str
    acc: float
    stress: float
    recon: float
    stage: str
    params: int
    folds: bool
    hit: bool
    meta: dict = field(default_factory=dict)


def csv_append(row: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def progress_append(line: str):
    with PROGRESS.open("a") as f:
        f.write(line + "\n")


def cwf(total_params: int, r_input: float, acc: float) -> float:
    """Compression Waste Factor: log10(params) / (r * -log10(1-acc)).

    1 = the Zador-ideal model for this input intrinsic dimension, lower is
    better. Uses the *reduced* input dimension, which is the point of this study.
    """
    a = min(max(acc / 100.0, 1e-6), 1 - 1e-6)
    return float(np.log10(max(total_params, 2)) / (r_input * -np.log10(1 - a)))


# ---------------------------------------------------------------------------
# one experiment
# ---------------------------------------------------------------------------

class Session:
    def __init__(self, cm, Xtr, ytr, Xte, yte, args):
        self.cm = cm
        self.Xtr, self.ytr, self.Xte, self.yte = Xtr, ytr, Xte, yte
        self.args = args
        self.dev = args.device
        self.runs: list[Run] = []
        self.best: Run | None = None
        self._novel: list = []
        self._geo_cache: dict = {}
        self._last_run = self._last_enc = self._last_A = self._last_c = None

        self.Y_tr = (Xtr @ cm.W1.t())                       # first-layer pre-acts
        self.lo = cm.dense_acc - args.tol
        self.hi = cm.dense_acc + args.tol

        # fitting subsample (search speed); final refit uses everything
        n = Xtr.shape[0]
        g = np.random.default_rng(args.seed)
        self.fit_idx = (g.choice(n, min(args.n_fit, n), replace=False)
                        if n > args.n_fit else np.arange(n))

    # -- setup ------------------------------------------------------------
    def setup(self):
        cm = self.cm
        print(f"\n{'#'*100}\n# {cm.tag}   dataset={cm.dataset}  D={cm.in_dim}  "
              f"hidden={cm.spec['hidden_dims']}\n{'#'*100}")
        print(f"\U0001F3AF AIM: fully compress {cm.model_name} to within "
              f"[{self.lo:.2f}%, {self.hi:.2f}%] of dense {cm.dense_acc:.2f}% "
              f"(hidden-only already gives {cm.compressed_acc:.2f}%)")
        head = cm.compressed_acc - self.lo
        print(f"   headroom left by the hidden compression: {head:+.2f} pts")
        if head <= 0:
            print("   ⚠️  the hidden stage already spent the whole budget — the ±tol "
                  "target is unreachable here by construction; reporting the *extra* "
                  "cost of input reduction instead.")

        t0 = time.time()
        self.d_input = gride_id(self.Xtr[self.fit_idx].float().cpu().numpy(),
                                seed=self.args.seed)
        self.r_grid = sorted({max(2, int(round(m * self.d_input))) for m in MULT_GRID})
        self.r_grid = [r for r in self.r_grid if r < self.cm.in_dim]
        print(f"\U0001F4D0 GRIDE on RAW INPUT: d_input = {self.d_input:.2f}  "
              f"[{time.time()-t0:.1f}s]")
        print(f"   r sweep = round(mult × d_input) for mult {MULT_GRID}  ->  {self.r_grid}")
        print(f"   fitting subsample: {len(self.fit_idx)}  test: {self.Xte.shape[0]}")

    # -- evaluation of one encoder ----------------------------------------
    def _score(self, enc, r, *, lam, strategy, K, oos, family, stage,
               stress=float("nan"), recon=float("nan"), n_fit=None) -> Run | None:
        cm = self.cm
        t0 = time.time()
        try:
            Xf = self.Xtr[self.fit_idx]
            Zf = enc.encode(Xf)
            A, c = fit_A_from_codes(Zf, self.Y_tr[self.fit_idx], self.args.ridge)
            model = build_reduced_model(cm, A, c, self.dev)
            Zte = enc.encode(self.Xte)
            acc = accuracy_on_codes(model, Zte, self.yte)
        except Exception as exc:
            log.warning("%s r=%s failed: %s", family, r, exc)
            return None
        secs = time.time() - t0

        cost = enc.cost(cm.in_dim)
        # model.parameters() already includes A (= layers[0]) and its bias; the
        # encoder is what sits in front of it, so the two simply add. A linear
        # encoder costs r*D+r and the pair stays factored (that is the saving);
        # a distance-based one costs K*D+K*r of stored landmarks.
        hidden_params = sum(p.numel() for p in model.parameters())
        total = hidden_params + cost["params"]
        hit = self.lo <= acc <= self.hi
        run = Run(cm.tag, cm.dataset, r, lam, strategy, K, oos, family, acc,
                  stress, recon, stage, total, cost["folds_into_first_layer"], hit)
        self.runs.append(run)
        # artefacts of *this* run. `_save` must write the weights belonging to the
        # run it is describing -- reading `self._best_*` instead wrote the globally
        # best run's tensors under the frontier run's metadata (r=193 weights in an
        # r=145 spec), which loads as a shape mismatch rather than a wrong answer,
        # but only because the verifier checks. Keep them paired.
        self._last_run, self._last_enc, self._last_A, self._last_c = run, enc, A, c
        if self.best is None or acc > self.best.acc:
            self.best = run
            self._best_enc = enc
            self._best_A, self._best_c = A, c

        flag = "  ✅ HIT" if hit else ""
        fold = "fold" if cost["folds_into_first_layer"] else "dist"
        print(f"   [{family:<22}] r={r:<4} λ={lam:<6} {strategy[:14]:<14} K={K:<4} "
              f"{oos:<8} {fold}  acc={acc:6.2f}%  Δdense={acc-cm.dense_acc:+6.2f}  "
              f"Δhidden={acc-cm.compressed_acc:+6.2f}  {secs:4.1f}s{flag}")

        csv_append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": cm.tag, "r": r, "lambda": lam,
            "landmark_strategy": strategy, "K": K, "out_of_sample_method": oos,
            "mds_stress": round(stress, 6) if stress == stress else "",
            "recon_loss": round(recon, 6) if recon == recon else "",
            "alpha": round(1 - recon, 6) if recon == recon else "",
            "compressed_acc": round(acc, 4), "target_hit": int(hit),
            "dataset": cm.dataset, "stage": stage, "encoder_family": family,
            "multiplier": round(r / self.d_input, 3), "r_input_gride": round(self.d_input, 4),
            "dense_acc": cm.dense_acc, "hidden_only_acc": cm.compressed_acc,
            "drop_vs_dense": round(cm.dense_acc - acc, 4),
            "drop_vs_hidden": round(cm.compressed_acc - acc, 4),
            "input_dim": cm.in_dim, "encoder_params": cost["params"],
            "encoder_flops": cost["flops_per_sample"],
            "folds_into_layer1": int(cost["folds_into_first_layer"]),
            "total_params": total, "cwf": round(cwf(total, self.d_input, acc), 5),
            "n_fit": n_fit or len(self.fit_idx), "seconds": round(secs, 2),
        })
        progress_append(
            f"- `{datetime.now():%Y-%m-%d %H:%M}` **{cm.tag}** {stage} | {family} | "
            f"r={r} (×{r/self.d_input:.2f} of d_input={self.d_input:.1f}) λ={lam} "
            f"{strategy} K={K} {oos} | acc **{acc:.2f}%** "
            f"(dense {cm.dense_acc:.2f}, hidden-only {cm.compressed_acc:.2f}) | "
            f"{'HIT' if hit else 'miss'}")
        return run

    # -- stage A: the linear ceiling and conventional baselines ------------
    def stage_a(self):
        print(f"\n{'='*100}\nSTAGE A — linear baselines (opt_linear is the provable "
              f"ceiling for every linear encoder)\n{'='*100}")
        for r in self.r_grid:
            for name in BASELINES:
                enc = build_baseline(name, self.Xtr[self.fit_idx], self.cm.W1, r,
                                     self.dev, self.args.seed)
                self._score(enc, r, lam=0.0, strategy="-", K=0, oos="linear",
                            family=name, stage="A")

    # -- srGW-MDS core ----------------------------------------------------
    def _geo(self, strategy, K):
        key = (strategy, K)
        if key not in self._geo_cache:
            Xs = self.Xtr[self.fit_idx]
            ys = self.ytr[self.fit_idx].cpu().numpy()
            idx = select_landmarks(Xs, ys, K, strategy, self.args.seed)
            D = landmark_dissimilarity(Xs, idx, kind=self.args.dissim,
                                       k=self.args.knn, seed=self.args.seed)
            self._geo_cache[key] = (idx, D)
        return self._geo_cache[key]

    def _srgw(self, r, lam, strategy, K, oos, stage):
        idx, D = self._geo(strategy, K)
        Xs = self.Xtr[self.fit_idx]
        Xl = Xs[idx]
        Y_l = Xl @ self.cm.W1.t()
        fit = fit_embedding(D, Y_l, r, lam, epochs=self.args.epochs,
                            lr=self.args.lr, device=self.dev, seed=self.args.seed)
        enc = build_encoder(oos, Xl, fit.Z, X_fit=Xs)
        return self._score(enc, r, lam=lam, strategy=strategy, K=K, oos=oos,
                           family=f"srgw_{oos}", stage=stage,
                           stress=fit.stress, recon=fit.recon)

    # -- stage B: r x lambda ----------------------------------------------
    def stage_b(self):
        print(f"\n{'='*100}\nSTAGE B — srGW-MDS: sweep r × λ "
              f"(landmarks=centroid_boundary, K=100, OOS=nystrom)\n{'='*100}")
        for r in self.r_grid:
            for lam in LAMBDA_GRID:
                self._srgw(r, lam, "centroid_boundary", 100, "nystrom", "B")

    # -- stage C: landmark strategy x K -----------------------------------
    def _anchor(self):
        """(r, lambda) for stages C/D: the best srGW run so far, else the r of the
        best baseline with a mid-grid lambda, so C/D can run standalone."""
        b = self._best_srgw()
        if b is not None:
            return b.r, b.lam
        base = max((x for x in self.runs), key=lambda x: x.acc, default=None)
        r = base.r if base else self.r_grid[len(self.r_grid) // 2]
        return r, self.args.anchor_lambda

    def stage_c(self):
        r, lam = self._anchor()
        print(f"\n{'='*100}\nSTAGE C — landmark strategy × K  at r={r}, λ={lam}\n{'='*100}")
        for strategy in STRATEGIES:
            for K in K_GRID:
                if strategy == "centroid_boundary" and K == 100 and self._best_srgw():
                    continue                       # already done in stage B
                self._srgw(r, lam, strategy, K, "nystrom", "C")

    # -- stage D: out-of-sample method -------------------------------------
    def stage_d(self):
        b = self._best_srgw()
        if b is None:
            r, lam = self._anchor()
            b = Run(self.cm.tag, self.cm.dataset, r, lam, "centroid_boundary", 200,
                    "nystrom", "srgw", -1, 0, 0, "D", 0, False, False)
        print(f"\n{'='*100}\nSTAGE D — out-of-sample extension  at r={b.r}, λ={b.lam}, "
              f"{b.strategy}, K={b.K}\n{'='*100}")
        for oos in OOS_METHODS:
            self._srgw(b.r, b.lam, b.strategy, b.K, oos, "D")

    def _best_srgw(self) -> Run | None:
        cand = [r for r in self.runs if r.family.startswith("srgw")]
        return max(cand, key=lambda r: r.acc) if cand else None

    # -- adaptive research -------------------------------------------------
    def diagnostics(self) -> str:
        cm = self.cm
        ranked = sorted(self.runs, key=lambda r: -r.acc)[:15]
        lin = [r for r in self.runs if r.family == "opt_linear"]
        best_lin = max(lin, key=lambda r: r.acc) if lin else None
        bs = self._best_srgw()
        L = [
            "ADAPTIVE RESEARCH REQUEST — input reduction", "=" * 78,
            f"generated       : {datetime.now().isoformat(timespec='seconds')}",
            f"model           : {cm.tag}  ({cm.dataset}, D={cm.in_dim})",
            f"dense accuracy  : {cm.dense_acc:.2f}%",
            f"hidden-only     : {cm.compressed_acc:.2f}%  (input still full-dimensional)",
            f"target window   : [{self.lo:.2f}%, {self.hi:.2f}%]",
            f"GRIDE d_input   : {self.d_input:.3f}   r grid: {self.r_grid}",
            f"runs so far     : {len(self.runs)}", "",
            "BEST SO FAR", "-" * 78,
        ]
        if self.best:
            L.append(f"{self.best.family} r={self.best.r} λ={self.best.lam} "
                     f"{self.best.strategy} K={self.best.K} {self.best.oos} -> "
                     f"{self.best.acc:.2f}%  (gap to window {self.lo - self.best.acc:+.2f})")
        if best_lin:
            L += ["", "THE LINEAR CEILING (opt_linear = reduced-rank regression)", "-" * 78,
                  f"best linear encoder at any r: {best_lin.acc:.2f}% (r={best_lin.r})",
                  "Any linear encoder — PCA, random projection, or a distilled srGW-MDS",
                  "probe — is mathematically bounded by this. Beating it REQUIRES a",
                  "non-linear encoder, which costs K×D flops per sample."]
        if bs and best_lin:
            L += ["", f"srGW-MDS best {bs.acc:.2f}% vs linear ceiling {best_lin.acc:.2f}% "
                      f"-> {'BEATS' if bs.acc > best_lin.acc else 'does NOT beat'} it"]
        L += ["", "TOP 15 RUNS", "-" * 78]
        L += [f"{r.acc:6.2f}%  {r.family:<20} r={r.r:<4} λ={r.lam:<6} {r.strategy:<18} "
              f"K={r.K:<4} {r.oos}" for r in ranked]

        by_lam, by_strat, by_oos = {}, {}, {}
        for r in self.runs:
            if not r.family.startswith("srgw"):
                continue
            by_lam[r.lam] = max(by_lam.get(r.lam, -1), r.acc)
            by_strat[r.strategy] = max(by_strat.get(r.strategy, -1), r.acc)
            by_oos[r.oos] = max(by_oos.get(r.oos, -1), r.acc)
        for title, d in (("BEST BY LAMBDA", by_lam), ("BEST BY LANDMARK STRATEGY", by_strat),
                         ("BEST BY OOS METHOD", by_oos)):
            L += ["", title, "-" * 78]
            L += [f"  {k}: {v:.2f}%" for k, v in sorted(d.items(), key=lambda kv: -kv[1])]

        L += ["", "WHAT IS NEEDED", "-" * 78,
              "Drop new encoder factories into novel_strategies/*.py exposing",
              "`STRATEGIES = [(name, fn), ...]` where fn(session, r) -> core.oos.Encoder.",
              "They are imported and swept automatically on the next round.",
              "",
              "Directions worth trying, given the diagnostics above:",
              " (a) build D_geo on the COMPRESSED HIDDEN ACTIVATIONS rather than raw",
              "     pixels — the manifold that matters is the one the network sees;",
              " (b) replace the linear reconstruction A with a small non-linear map,",
              "     accepting that it no longer folds;",
              " (c) density-based or coreset landmark selection;",
              " (d) supervised//class-aware D_geo (shrink between-class distances).",
              "",
              "CONSTRAINTS: r must stay round(multiplier × GRIDE d_input). Hidden layers",
              "stay frozen — no gradient may touch any network weight. Only the embedding",
              "and the linear map A are fitted.",
              ]
        return "\n".join(L)

    def load_novel(self, already: set[str]) -> list:
        NOVEL_DIR.mkdir(parents=True, exist_ok=True)
        new = []
        for f in sorted(NOVEL_DIR.glob("*.py")):
            if f.name.startswith("_") or f.name in already:
                continue
            already.add(f.name)
            try:
                spec = importlib.util.spec_from_file_location(f"novel_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                got = getattr(mod, "STRATEGIES", None) or []
                new.extend(got)
                self._novel.extend(got)
                print(f"   loaded {len(got)} encoder strategies from {f.name}")
            except Exception as exc:
                print(f"   !! failed to load {f.name}: {type(exc).__name__}: {exc}")
        return new

    def adaptive(self) -> bool:
        already: set[str] = set()
        for rnd in range(1, self.args.rounds + 1):
            print(f"\n{'='*100}\nADAPTIVE RESEARCH — round {rnd}/{self.args.rounds}\n{'='*100}")
            diag = self.diagnostics()
            REQUEST.write_text(diag)
            (RESULTS / f"adaptive_request_{self.cm.tag}.txt").write_text(diag)
            print(f"   diagnostics -> {REQUEST.relative_to(PROJECT_ROOT)}")
            if self.args.wait_for_operator:
                input(f"   add encoders to {NOVEL_DIR}/ then press Enter... ")
            novel = self.load_novel(already)
            if not novel:
                print("   no new strategies available — ending adaptive phase.")
                return False
            before = self.best.acc if self.best else -1
            for name, fn in novel:
                for r in self.r_grid:
                    try:
                        enc, meta = fn(self, r)
                    except Exception as exc:
                        log.warning("%s r=%s build failed: %s", name, r, exc)
                        continue
                    self._score(enc, r, lam=meta.get("lam", 0.0),
                                strategy=meta.get("strategy", "-"), K=meta.get("K", 0),
                                oos=meta.get("oos", "custom"), family=name,
                                stage=f"adaptive{rnd}", stress=meta.get("stress", float("nan")),
                                recon=meta.get("recon", float("nan")))
            gain = (self.best.acc if self.best else -1) - before
            if any(r.hit for r in self.runs) and not self.args.exhaustive:
                return True
            if gain < 0.5 and not self.args.exhaustive:
                print(f"   round {rnd} improved best by only {gain:+.2f} pts — stopping.")
                return False
        return any(r.hit for r in self.runs)

    # -- rebuilding a configuration on the FULL training set ---------------
    def _rebuild_on_full(self, family, r, *, lam=0.0, strategy="-", K=0, oos="linear",
                         stage="final"):
        """Re-fit one configuration using every training sample.

        The staged search runs on a subsample for speed, so anything that will be
        quoted or saved has to be rebuilt here. Encoder-level caches keyed to the
        subsample (the geodesic distances, the logit-pullback metric) are dropped
        first, otherwise the "full-data" refit would silently reuse them.
        """
        saved_idx = self.fit_idx
        self.fit_idx = np.arange(self.Xtr.shape[0])
        self._geo_cache.clear()
        for attr in ("_Lsqrt",):
            if hasattr(self, attr):
                delattr(self, attr)
        try:
            novel = {n: fn for n, fn in self._novel}
            if family.startswith("srgw"):
                return self._srgw(r, lam, strategy, K, oos, stage)
            if family in novel:
                enc, meta = novel[family](self, r)
                return self._score(enc, r, lam=meta.get("lam", 0.0),
                                   strategy=meta.get("strategy", "-"),
                                   K=meta.get("K", 0), oos=meta.get("oos", "custom"),
                                   family=family, stage=stage)
            if family in BASELINES:
                enc = build_baseline(family, self.Xtr, self.cm.W1, r, self.dev,
                                     self.args.seed)
                return self._score(enc, r, lam=0.0, strategy="-", K=0, oos="linear",
                                   family=family, stage=stage)
            print(f"   cannot rebuild family {family!r} on full data")
            return None
        finally:
            self.fit_idx = saved_idx
            self._geo_cache.clear()
            for attr in ("_Lsqrt",):
                if hasattr(self, attr):
                    delattr(self, attr)

    def _save(self, run, kind: str, selection: str):
        assert run is self._last_run, (
            "_save must be called immediately after the run it describes")
        assert self._last_A.shape[1] == run.r, (
            f"artefact/metadata mismatch: A has r={self._last_A.shape[1]} but the "
            f"run reports r={run.r}")
        meta = {
            "source_compressed": self.cm.tag, "dataset": self.cm.dataset,
            "input_dim": self.cm.in_dim, "r": run.r,
            "r_input_gride": self.d_input, "multiplier": run.r / self.d_input,
            "encoder_family": run.family, "lambda": run.lam,
            "landmark_strategy": run.strategy, "K": run.K,
            "out_of_sample_method": run.oos, "folds_into_first_layer": run.folds,
            "hidden_dims": self.cm.spec["hidden_dims"],
            "num_classes": self.cm.spec["num_classes"],
            "dense_acc": self.cm.dense_acc, "hidden_only_acc": self.cm.compressed_acc,
            "fully_compressed_acc": run.acc, "total_params": run.params,
            "cwf": cwf(run.params, self.d_input, run.acc),
            "target_hit": run.hit, "fitted_on_n": int(self.Xtr.shape[0]),
            "selection": selection,
        }
        path = save_reduced(build_reduced_model(self.cm, self._last_A, self._last_c, self.dev),
                            self._last_enc, self.cm, meta, RESULTS, kind=kind)
        print(f"   saved -> {path.name}")
        return path

    # -- final: refit the most accurate configuration on ALL data ----------
    def finalize(self):
        b = self.best
        if b is None:
            return None
        print(f"\n{'='*100}\nFINAL — refitting the most accurate configuration on ALL "
              f"{self.Xtr.shape[0]} training samples\n{'='*100}")
        run = self._rebuild_on_full(b.family, b.r, lam=b.lam, strategy=b.strategy,
                                    K=b.K, oos=b.oos, stage="final")
        if run is None:
            return None
        self._save(run, "best", "highest accuracy")
        return run

    # -- frontier: smallest r that clears the window ON FULL DATA -----------
    def validate_frontier(self):
        """Find, and verify on all training data, the smallest GRIDE budget that
        still clears the window.

        Frontier selection picks the configuration with the *least* margin by
        construction, so a frontier chosen on the search subsample is
        systematically optimistic — two of eight drifted outside the window when
        first refitted on full data. Rather than trusting the subsample, walk the
        GRIDE grid upward and re-fit on everything at each step, stopping at the
        first budget that genuinely clears. Costs a handful of extra fits and
        removes the whole failure mode.
        """
        hits = [x for x in self.runs if x.hit]
        if not hits:
            print("\n   no configuration cleared the window — no frontier to validate")
            return None
        start = min(x.r for x in hits)
        fam = min((x for x in hits if x.r == start), key=lambda x: -x.acc)
        ladder = [r for r in self.r_grid if r >= start]
        print(f"\n{'='*100}\nFRONTIER — validating on ALL {self.Xtr.shape[0]} training "
              f"samples, walking up from r={start}\n{'='*100}")
        for r in ladder:
            run = self._rebuild_on_full(fam.family, r, lam=fam.lam, strategy=fam.strategy,
                                        K=fam.K, oos=fam.oos, stage="frontier")
            if run is None:
                continue
            if run.hit:
                print(f"   frontier confirmed on full data: r={r} "
                      f"(×{r/self.d_input:.2f} of d_input) -> {run.acc:.2f}%")
                self._save(run, "frontier", "smallest r clearing the window on full data")
                return run
            print(f"   r={r} misses on full data ({run.acc:.2f}%, "
                  f"Δdense {run.acc-self.cm.dense_acc:+.2f}) — trying the next budget")
        print("   no budget on the GRIDE ladder clears the window on full data")
        return None
        return run


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _matches(tag: str, pattern: str) -> bool:
    """Anchored prefix match on underscore boundaries.

    A bare substring test lets `mnist` match `fashionmnist_mlp2_bn_compressed`,
    which silently ran one model under another's configuration and overwrote its
    artefact. Requiring the pattern to end at a `_` boundary makes `mnist`,
    `cifar10` and `mnist_mlp2_bn` all behave as intended.
    """
    return tag == pattern or tag.startswith(pattern + "_")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", default="frontier", choices=["frontier", "compressed"],
                   help="which hidden-compressed variant to build on")
    p.add_argument("--models", default=None, help="comma-separated tag substrings")
    p.add_argument("--tol", type=float, default=2.5)
    p.add_argument("--n-fit", type=int, default=20000, help="training subsample for the search")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--knn", type=int, default=12, help="k for the geodesic graph")
    p.add_argument("--dissim", default="geodesic", choices=["geodesic", "euclidean"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--stages", default="ABCD")
    p.add_argument("--anchor-lambda", type=float, default=10.0,
                   help="lambda used by stages C/D when stage B was not run")
    p.add_argument("--wait-for-operator", action="store_true")
    p.add_argument("--no-final", action="store_true")
    p.add_argument("--only", default=None,
                   help="rebuild exactly one configuration on ALL data and save it, "
                        "as 'FAMILY@r' (e.g. N1_jacobian_linear@30)")
    p.add_argument("--exhaustive", action="store_true",
                   help="run every stage AND the adaptive phase even if a stage already "
                        "cleared the window (otherwise novel encoders are never tested)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    return a


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not PROGRESS.exists():
        PROGRESS.write_text(
            "# compression_input_space — run log\n\n"
            "One line per experiment. `r` is always `round(multiplier × GRIDE d_input)`,\n"
            "with `d_input` measured on the raw data at runtime. Hidden layers frozen.\n\n")

    found = discover_compressed(args.kind)
    if not found:
        raise SystemExit(f"no *_{args.kind}.arch.json found in ../compression_mlp/results/")
    if args.models:
        pats = [s.strip() for s in args.models.split(",")]
        found = [(p, t) for p, t in found if any(_matches(t, s) for s in pats)]
        if not found:
            raise SystemExit(f"--models {args.models!r} matched nothing")
    print(f"discovered {len(found)} compressed models ({args.kind}):")
    for _, t in found:
        print(f"   {t}")

    by_dataset: dict[str, tuple] = {}
    outcomes = []
    for spec_path, tag in found:
        cm = load_compressed(spec_path, args.device)
        if cm.dataset not in by_dataset:
            by_dataset[cm.dataset] = load_flat_data(cm.dataset, args.device,
                                                    seed=args.seed)
        Xtr, ytr, Xte, yte = by_dataset[cm.dataset]

        sess = Session(cm, Xtr, ytr, Xte, yte, args)
        sess.setup()
        if args.only:
            fam, _, rr = args.only.partition("@")
            sess.fit_idx = np.arange(Xtr.shape[0])       # full data, as required
            sess.load_novel(set())
            novel = {n: fn for n, fn in sess._novel}
            r = int(rr)
            if fam in novel:
                enc, meta = novel[fam](sess, r)
                run = sess._score(enc, r, lam=meta.get("lam", 0.0),
                                  strategy=meta.get("strategy", "-"), K=meta.get("K", 0),
                                  oos=meta.get("oos", "custom"), family=fam, stage="frontier")
            else:
                enc = build_baseline(fam, Xtr, cm.W1, r, args.device, args.seed)
                run = sess._score(enc, r, lam=0.0, strategy="-", K=0, oos="linear",
                                  family=fam, stage="frontier")
            if run is not None:
                meta = {"source_compressed": cm.tag, "dataset": cm.dataset,
                        "input_dim": cm.in_dim, "r": r, "r_input_gride": sess.d_input,
                        "multiplier": r / sess.d_input, "encoder_family": fam,
                        "lambda": run.lam, "landmark_strategy": run.strategy, "K": run.K,
                        "out_of_sample_method": run.oos, "folds_into_first_layer": run.folds,
                        "hidden_dims": cm.spec["hidden_dims"],
                        "num_classes": cm.spec["num_classes"], "dense_acc": cm.dense_acc,
                        "hidden_only_acc": cm.compressed_acc,
                        "fully_compressed_acc": run.acc, "total_params": run.params,
                        "cwf": cwf(run.params, sess.d_input, run.acc),
                        "target_hit": run.hit, "fitted_on_n": int(Xtr.shape[0]),
                        "selection": "smallest r inside the window"}
                path = save_reduced(
                    build_reduced_model(cm, sess._last_A, sess._last_c, args.device),
                    sess._last_enc, cm, meta, RESULTS, kind="frontier")
                print(f"   saved -> {path.name}")
            continue
        if "A" in args.stages:
            sess.stage_a()
        if "B" in args.stages:
            sess.stage_b()
        if "C" in args.stages:
            sess.stage_c()
        if "D" in args.stages:
            sess.stage_d()
        hit = any(r.hit for r in sess.runs)
        if not hit or args.exhaustive:
            hit = sess.adaptive() or hit
        if not args.no_final:
            sess.finalize()
            sess.validate_frontier()

        b = sess.best
        outcomes.append((tag, hit, b))
        if hit:
            print(f"\n\U0001F389 SUCCESS for {tag}: {b.acc:.2f}% within "
                  f"[{sess.lo:.2f}, {sess.hi:.2f}]")
        else:
            print(f"\n⚠️  {tag}: best {b.acc:.2f}% "
                  f"(dense {cm.dense_acc:.2f}, hidden-only {cm.compressed_acc:.2f}, "
                  f"need ≥{sess.lo:.2f})")

    print(f"\n{'='*100}\n{sum(1 for _, h, _ in outcomes if h)}/{len(outcomes)} models "
          f"inside ±{args.tol} of dense\n{'='*100}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
