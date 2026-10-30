#!/usr/bin/env python3
"""One loader for every experiment in the repository.

Each study wrote its own CSV schema at its own time, so the column names, the
strategy vocabularies and even the sign convention of the accuracy drop differ.
This normalises all of them into a single record type so that contrasts can be
computed *across* architectures rather than only inside one, which is the whole
point of the analysis this folder exists to do.

Nothing here recomputes an experiment. Every field is read from a CSV that was
written by the original driver, and the file each record came from is carried on
the record so any number can be traced back.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The three studies use three names for the same two operators. Selection is
# either target-blind (span the activation space) or target-aware (group OMP
# against what the consumer reads); the refit is either present or absent.
CANON = {
    # MLP  (compression_mlp/run_autonomous_compression.py)
    "N01_span_ls": "span_ls", "N02_omp_ls": "omp_ls",
    "N02_omp_ls_lowridge": "omp_ls_lowridge",
    "N07_fwdbwd_omp": "fwdbwd_ls", "N10_fisher_span_ls": "fisher_span_ls",
    "N09_fisher_omp": "fisher_omp",
    "02_topm_l2_norm": "magnitude", "04_cosine_ot_intrafusion": "ot_intrafusion",
    "08_ot_then_ls": "ot_then_ls", "03_activation_pattern_kmeans": "kmeans_merge",
    "01_direct_ls_svd": "svd_ls", "10_iterative_refinement": "iterative_ls",
    "N06_augdict_omp": "augdict_omp", "N08_augdict_fwdbwd": "augdict_fwdbwd",
    # CNN  (compression_cnn/hidden_layer_compression/strategies.py)
    "S3_span_ls": "span_ls", "S4_omp_ls": "omp_ls", "S5_fwdbwd_ls": "fwdbwd_ls",
    "S1_magnitude": "magnitude", "S2_magnitude_ls": "magnitude_ls",
    "S6_ot_intrafusion": "ot_intrafusion", "S7_ot_then_ls": "ot_then_ls",
    "S8_kmeans_merge": "kmeans_merge", "S9_augdict_omp": "augdict_omp",
    "S10_augdict_fwdbwd": "augdict_fwdbwd",
    # Transformer  (compression_transformer/scripts/core.py) already canonical
}

REFIT = {"span_ls", "omp_ls", "omp_ls_lowridge", "fwdbwd_ls", "fisher_span_ls",
         "fisher_omp", "ot_then_ls", "svd_ls", "iterative_ls", "magnitude_ls",
         "augdict_omp", "augdict_fwdbwd", "random_ls", "magnitude_prod_ls",
         "thinet_lsfull", "omp_ls_noint"}
MERGE = {"ot_intrafusion", "kmeans_merge", "ot_then_ls"}
TARGET_AWARE = {"omp_ls", "omp_ls_lowridge", "fwdbwd_ls", "fisher_omp",
                "augdict_omp", "augdict_fwdbwd", "omp", "omp_ls_noint"}


@dataclass
class Run:
    arch: str                  # mlp | cnn | transformer
    model: str
    dataset: str
    strategy: str              # canonical
    mult: float
    acc: float
    dense: float
    drop: float                # signed, negative means worse than dense
    params_cut: float          # % of the whole model
    widths: list[int]
    orig_widths: list[int]
    dims: list[float]
    phase: str
    source: str
    extra: dict = field(default_factory=dict)

    @property
    def n_over_d(self) -> float:
        r = [n / d for n, d in zip(self.orig_widths, self.dims) if d > 0]
        return sum(r) / len(r) if r else float("nan")

    @property
    def has_refit(self) -> bool:
        return self.strategy in REFIT

    @property
    def is_merge(self) -> bool:
        return self.strategy in MERGE

    @property
    def target_aware(self) -> bool:
        return self.strategy in TARGET_AWARE


def _jl(s, default=None):
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else default
    except Exception:
        return default


def _f(s, default=float("nan")):
    try:
        return float(s)
    except Exception:
        return default


# Input and output dimensions of the fully connected checkpoints, needed to
# turn recorded widths into multiply-accumulates. A dense layer performs one MAC
# per weight per sample, so a fully connected network's MAC reduction equals its
# weight reduction exactly; the recorded `param_reduction_pct` differs only by
# the bias terms, which are a fraction of a percent.
_MLP_IO = {"MNIST": (784, 10), "FashionMNIST": (784, 10),
           "CIFAR10": (3072, 10), "CIFAR100": (3072, 100)}


def mlp_macs(widths, dataset):
    io = _MLP_IO.get(dataset)
    if not io or not widths:
        return float("nan")
    D, C = io
    dims = [D] + list(widths) + [C]
    return float(sum(a * b for a, b in zip(dims[:-1], dims[1:])))


def load_mlp() -> list[Run]:
    p = ROOT / "compression_mlp/results/compression_results.csv"
    out = []
    for r in csv.DictReader(p.open()):
        stg = CANON.get(r["strategy_name"], r["strategy_name"])
        w = _jl(r["widths"]) or []
        ow = _jl(r["orig_widths"]) or []
        d = _jl(r["gride_dims"]) or []
        acc, dense = _f(r["compressed_acc"]), _f(r["original_acc"])
        # The MLP driver wrote `model_name` without the dataset prefix, so
        # `mlp2_bn` names two different checkpoints. Qualifying it here is not
        # cosmetic: without it a paired contrast can match an MNIST run against
        # a FashionMNIST one at the same nominal model and multiplier.
        ident = f"{r['dataset'].lower()}_{r['model_name']}"
        out.append(Run("mlp", ident, r["dataset"], stg,
                       _f(r["multiplier"]), acc, dense, acc - dense,
                       _f(r["param_reduction_pct"]), w, ow, d, r["phase"],
                       "compression_mlp/results/compression_results.csv",
                       {"id_space": r.get("id_space", ""),
                        "macs_cut": (100.0 * (1 - mlp_macs(w, r["dataset"])
                                              / mlp_macs(ow, r["dataset"]))
                                     if w and ow else float("nan"))}))
    return out


def load_cnn() -> list[Run]:
    p = ROOT / "compression_cnn/hidden_layer_compression/results/compression_results_cnn.csv"
    out = []
    for r in csv.DictReader(p.open()):
        stg = CANON.get(r["strategy"], r["strategy"])
        acc, dense = _f(r["compressed_acc"]), _f(r["dense_acc"])
        out.append(Run("cnn", r["model_name"], r["dataset"], stg,
                       _f(r["multiplier"]), acc, dense, acc - dense,
                       _f(r["param_reduction_pct"]),
                       _jl(r["widths"]) or [], _jl(r["orig_widths"]) or [],
                       _jl(r["gride_dims"]) or [], r["phase"],
                       "compression_cnn/hidden_layer_compression/results/"
                       "compression_results_cnn.csv",
                       {"macs_cut": _f(r["mac_reduction_pct"]),
                        "family": r.get("family", ""), "stage": r.get("stage", "")}))
    return out


def load_transformer() -> list[Run]:
    base = ROOT / "compression_transformer/results"
    out = []
    for name, phase in (("sweep.csv", "ladder"), ("baselines.csv", "matched"),
                        ("ablation.csv", "ablation"), ("stability.csv", "seeds"),
                        ("perlayer.csv", "solo"), ("hyper.csv", "hyper"),
                        ("alloc.csv", "alloc")):
        p = base / name
        if not p.exists():
            continue
        for r in csv.DictReader(p.open()):
            acc, dense = _f(r["acc"]), _f(r["dense_acc"])
            out.append(Run("transformer", "distilbert_mnli", "mnli",
                           r["strategy"], _f(r["alpha"]), acc, dense,
                           acc - dense, _f(r["red_total_pct"]),
                           _jl(r["widths"]) or [], [3072] * 6,
                           _jl(r["dims"]) or [], phase,
                           f"compression_transformer/results/{name}",
                           {"tag": r.get("tag", ""), "seed": r.get("seed", ""),
                            "cloud": r.get("cloud", ""),
                            "target": r.get("target", ""),
                            "sequential": r.get("sequential", ""),
                            "layers": r.get("layers", ""),
                            "red_ffn": _f(r.get("red_ffn_pct", "nan")),
                            "red_enc": _f(r.get("red_encoder_pct", "nan"))}))
    return out


def load_all() -> list[Run]:
    return load_mlp() + load_cnn() + load_transformer()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def summarise(v):
    n = len(v)
    if n == 0:
        return None
    m = sum(v) / n
    pos = sum(1 for x in v if x > 0)
    if n == 1:
        return {"n": 1, "mean": round(m, 3), "sd": 0.0, "se": 0.0, "t": None,
                "pos": pos}
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    se = sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 3), "sd": round(sd, 3), "se": round(se, 3),
            "t": round(m / se, 2) if se > 0 else None, "pos": pos}


def paired(runs, a: str, b: str, key=lambda r: (r.arch, r.model, r.mult, r.phase)):
    """Accuracy of `a` minus accuracy of `b` over cells where both ran at the
    same model and the same budget. Nothing is compared across widths."""
    ia = {key(r): r for r in runs if r.strategy == a}
    ib = {key(r): r for r in runs if r.strategy == b}
    return [(k, ia[k].acc - ib[k].acc) for k in sorted(set(ia) & set(ib), key=str)]


if __name__ == "__main__":
    runs = load_all()
    from collections import Counter
    c = Counter(r.arch for r in runs)
    print("records loaded:", dict(c), "total", len(runs))
    print("distinct models:", len({(r.arch, r.model) for r in runs}))
    print("distinct strategies:", len({r.strategy for r in runs}))
