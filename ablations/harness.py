"""Shared harness for the follow-up experiments in `TO_DO_EXPERIMENTS.md`.

Each experiment here re-uses the production `Session` from
`compression_cnn/hidden_layer_compression/run_cnn_compression.py` rather than
reimplementing the compression, so an experiment cannot accidentally measure a
*different* method than the paper reports. What varies is only what the
experiment is about: the seed, the calibration size, the strategy, or the
per-layer budget.

Two things this module is careful about.

**It does not write into the paper's CSVs.** `run_cnn_compression` appends to a
module-level `CSV_PATH`; every experiment redirects that to its own directory
before constructing a Session, so a side experiment can never contaminate the
sweep the paper quotes.

**It reads the frontier configuration from disk, not from memory.** The
multiplier and strategy each experiment re-runs come from the saved
`*_frontier.arch.json` written by the original sweep, so the configuration under
test is provably the one in Table 1.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HLC = ROOT / "compression_cnn" / "hidden_layer_compression"
for p in (str(ROOT), str(ROOT / "compression_cnn"), str(HLC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                              # noqa: E402

import run_cnn_compression as RC                          # noqa: E402
import strategies as ST                                   # noqa: E402
from common.discovery import discover, load_checkpoint     # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

FRONTIER_DIR = HLC / "results"


# ---------------------------------------------------------------------------
# defaults -- identical to the production driver's argparse defaults, so an
# experiment differs from the paper only in what it deliberately changes
# ---------------------------------------------------------------------------

def base_args(**over) -> SimpleNamespace:
    a = SimpleNamespace(
        models=None, all=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tol=2.5, calib=1024, od_factor=4.0,
        patch_budget=60000, gride_patches=20000, ridge=1e-4,
        batch_size=256, workers=4, seed=0, id_space="filter_proj",
        strategies=None, tourney_mults=None, top_k=3, adaptive_rounds=3,
        allocate=[], results_dir=None, include_novel=False, debug=False,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def redirect_csv(path: Path):
    """Point the driver's module-level CSV at an experiment-local file.

    Without this every experiment would append into the sweep the paper quotes.
    The driver rotates on a header mismatch, so a stale file with a different
    schema is renamed rather than corrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    RC.RESULTS = path.parent
    RC.CSV_PATH = path


# ---------------------------------------------------------------------------
# the saved frontier configurations -- what Table 1 actually reports
# ---------------------------------------------------------------------------

@dataclass
class Frontier:
    model: str
    dataset: str
    family: str
    strategy: str
    multiplier: float
    widths: list[int]
    gride: dict
    dense_acc: float
    compressed_acc: float
    params: int
    macs: int
    dense_params: int
    dense_macs: int
    calib: int
    path: Path = field(repr=False, default=None)


def frontiers() -> dict[str, Frontier]:
    """Every `*_frontier.arch.json` the production sweep wrote."""
    out = {}
    for p in sorted(FRONTIER_DIR.glob("*_frontier.arch.json")):
        m = json.loads(p.read_text())
        tag = p.name[: -len("_frontier.arch.json")]
        out[tag] = Frontier(
            model=tag, dataset=m["dataset"], family=m["family"],
            strategy=m["strategy"], multiplier=float(m["multiplier"]),
            widths=list(m["widths"]), gride=m["gride_dims"],
            dense_acc=float(m["dense_acc"]),
            compressed_acc=float(m["compressed_acc"]),
            params=int(m["params"]), macs=int(m["macs"]),
            dense_params=int(m["dense_params"]), dense_macs=int(m["dense_macs"]),
            calib=int(m.get("calib", 0)), path=p)
    return out


def checkpoint_for(tag: str):
    """The `.safetensors` behind a frontier tag, via the project's own discovery."""
    for path, mname in discover():
        if path.stem == tag:
            return path, mname
    raise KeyError(f"no checkpoint named {tag!r}; have "
                   f"{[p.stem for p, _ in discover()]}")


# ---------------------------------------------------------------------------
# running one configuration
# ---------------------------------------------------------------------------

def make_session(tag: str, args: SimpleNamespace):
    """A Session with phases 0 and 1 done: dense evaluated, BN folded,
    calibration drawn, GRIDE measured. Nothing compressed yet."""
    path, mname = checkpoint_for(tag)
    ck = load_checkpoint(path, mname, args.device)
    sess = RC.Session(ck, args)
    sess.setup()
    sess.gride_profile()
    return sess


def run_config(sess, strategy_name: str, multiplier: float, *,
               targets: list[int] | None = None, phase: int = 9,
               stage: str = "experiment"):
    """Compress + evaluate one configuration. Returns the driver's own `Run`."""
    strat = ST.BY_NAME[strategy_name]
    return sess.evaluate_config(strat, multiplier, phase=phase, stage=stage,
                                targets=targets)


def register(strategy) -> str:
    """Make a locally-defined Strategy visible to the driver by name."""
    ST.BY_NAME[strategy.name] = strategy
    if strategy not in ST.ALL:
        ST.ALL.append(strategy)
    return strategy.name


def free():
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# small stats helpers (no scipy dependency anywhere in this project)
# ---------------------------------------------------------------------------

def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def sd(v, ddof: int = 1):
    v = list(v)
    n = len(v)
    if n <= ddof:
        return float("nan")
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (n - ddof)) ** 0.5


def fmt(v, p=2):
    return "--" if v != v else f"{v:.{p}f}"
