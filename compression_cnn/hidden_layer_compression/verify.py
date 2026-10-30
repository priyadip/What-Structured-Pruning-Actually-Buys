#!/usr/bin/env python3
"""Re-check every saved compressed CNN, from disk, through an independent path.

Nothing is carried over from the run that produced the artefact: the
architecture is reconstructed from the recorded module shapes, the weights are
loaded `strict=True`, and both the compressed and the original dense model are
re-evaluated on the full test split. A claim that does not reproduce here is not
a result.

    python3 verify.py
    python3 verify.py --kind frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXT_ROOT = HERE.parent
PROJECT_ROOT = EXT_ROOT.parent
for p in (str(PROJECT_ROOT), str(EXT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.discovery import evaluate, load_checkpoint, loaders_for  # noqa: E402
from common.rebuild import count_macs, count_params, load_compressed  # noqa: E402
from utils.config import CHECKPOINT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--kind", default=None, choices=["frontier", "best"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=2.5)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    specs = sorted(Path(a.results).glob("*.arch.json"))
    if a.kind:
        specs = [p for p in specs if p.name.endswith(f"_{a.kind}.arch.json")]
    if not specs:
        raise SystemExit(f"no artefacts in {a.results}")

    bad = 0
    for sp in specs:
        model, meta = load_compressed(sp, a.device)
        ck = load_checkpoint(CHECKPOINT_DIR / meta["source_checkpoint"],
                             meta["model_name"], a.device)
        _, test = loaders_for(ck, a.batch_size, 4)

        acc = evaluate(model, test, a.device)
        acc_d = evaluate(ck.model, test, a.device)
        p = count_params(model)
        mm = count_macs(model, tuple(meta["input_shape"]), a.device)

        repro = (abs(acc - meta["compressed_acc"]) < 0.02
                 and abs(acc_d - meta["dense_acc"]) < 0.02
                 and p == meta["params"] and mm == meta["macs"])
        inwin = abs(acc_d - acc) <= a.tol
        bad += 0 if repro else 1

        print(f"\n{sp.name.replace('.arch.json','')}  [{meta['strategy']} "
              f"x{meta['multiplier']}]")
        print(f"  units        : {len(meta['widths'])}  "
              f"{meta['original_widths']} -> {meta['widths']}")
        print(f"  GRIDE ({meta['id_space']}): "
              + ", ".join(f"{d[meta['id_space']]:.1f}"
                          for d in meta["gride_dims"].values()))
        print(f"  params       : {meta['dense_params']:,} -> {p:,}  "
              f"(-{100*(1-p/meta['dense_params']):.1f}%)")
        print(f"  MACs         : {meta['dense_macs']/1e6:.1f}M -> {mm/1e6:.1f}M  "
              f"(-{100*(1-mm/meta['dense_macs']):.1f}%)")
        print(f"  dense acc    : {acc_d:.2f}%   (recorded {meta['dense_acc']:.2f}%)")
        print(f"  compressed   : {acc:.2f}%   (recorded {meta['compressed_acc']:.2f}%)")
        print(f"  drop         : {acc_d - acc:+.2f} pts   within +/-{a.tol}: "
              f"{'YES' if inwin else 'NO'}")
        print(f"  reproduces   : {'YES' if repro else 'NO -- MISMATCH'}")
        del model, ck
        torch.cuda.empty_cache()

    print(f"\n{len(specs) - bad}/{len(specs)} verified.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
