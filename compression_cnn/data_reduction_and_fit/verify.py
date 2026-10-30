#!/usr/bin/env python3
"""Re-check every saved factorised CNN, from disk, through an independent path.

Same contract as `../hidden_layer_compression/verify.py`: the architecture is
rebuilt from the recorded module shapes, the weights are loaded `strict=True`,
and both the factorised model and the original dense checkpoint are re-evaluated
on the full test split. Nothing is carried over from the run that produced the
artefact.

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
        bad += 0 if repro else 1

        print(f"\n{sp.name.replace('.arch.json','')}  [{meta['strategy']} "
              f"x{meta['multiplier']} stage {meta['stage']}]")
        print(f"  pipeline     : {meta['pipeline']}"
              + (f"  (hidden widths {meta['hidden_widths']})"
                 if meta.get('hidden_widths') else ""))
        print(f"  d_image      : {meta['d_image']:.2f} of 3072   "
              f"d_rgb_patch: {meta['d_rgb_patch']:.2f} of 27")
        print(f"  ranks        : {meta['ranks']}")
        print(f"  break-even   : {[round(b) for b in meta['break_even_ranks']]}")
        print(f"  params       : {meta['dense_params']:,} -> {p:,}  "
              f"(-{100*(1-p/meta['dense_params']):.1f}%)")
        print(f"  MACs         : {meta['dense_macs']/1e6:.1f}M -> {mm/1e6:.1f}M  "
              f"(-{100*(1-mm/meta['dense_macs']):.1f}%)")
        print(f"  dense acc    : {acc_d:.2f}%   (recorded {meta['dense_acc']:.2f}%)")
        print(f"  base acc     : {meta['base_acc']:.2f}%")
        print(f"  factorised   : {acc:.2f}%   (recorded {meta['compressed_acc']:.2f}%)")
        print(f"  drop vs dense: {acc_d - acc:+.2f} pts   within +/-{a.tol}: "
              f"{'YES' if abs(acc_d - acc) <= a.tol else 'NO'}")
        print(f"  reproduces   : {'YES' if repro else 'NO -- MISMATCH'}")
        # A file named "frontier" that holds a Stage-A configuration means no
        # Stage-B budget cleared the window while actually shrinking the model.
        # That is a legitimate outcome, but the filename would otherwise imply a
        # compression that does not exist.
        if "frontier" in sp.name and (meta["stage"] != "B" or p >= meta["dense_params"]
                                      or mm >= meta["dense_macs"]):
            print("  NOTE         : this is not a Stage-B compression -- no per-layer "
                  "rank budget cleared\n                 the window while reducing the "
                  "model. Quote the summary, not this file.")
        del model, ck
        torch.cuda.empty_cache()

    print(f"\n{len(specs) - bad}/{len(specs)} verified.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
