#!/usr/bin/env python3
"""Independently re-check every saved fully-compressed model.

Rebuilds encoder + network from `results/final_fully_compressed_*.safetensors`
and its `.arch.json` alone — nothing carried over from the run that produced it —
and re-evaluates on the full test split, alongside the original dense model.

    python3 compression_input_space/verify_fully_compressed.py
    python3 compression_input_space/verify_fully_compressed.py --tag frontier
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for _p in (str(PROJECT_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.mlp import MLP  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from utils.config import CHECKPOINT_DIR, DATASETS  # noqa: E402

from core.model_io import accuracy_on_codes, load_flat_data  # noqa: E402


def dense_params(spec: dict, explore: Path) -> int:
    src = explore / f"{spec['source_compressed']}.arch.json"
    s = json.loads(src.read_text())
    dims = [s["in_dim"]] + list(s["original_widths"]) + [s["num_classes"]]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)), s


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only specs whose name contains this")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=2.5)
    a = ap.parse_args()

    specs = sorted((HERE / "results").glob("final_fully_compressed_*.arch.json"))
    if a.tag:
        specs = [p for p in specs if a.tag in p.name]
    if not specs:
        raise SystemExit("no fully-compressed models found")

    explore = PROJECT_ROOT / "compression_mlp" / "results"
    cache: dict = {}
    bad = 0
    for sp in specs:
        spec = json.loads(sp.read_text())
        sd = load_file(str(sp.with_name(sp.name.replace(".arch.json", ".safetensors"))),
                       device="cpu")
        if "encoder.P" not in sd:
            print(f"\n{sp.name}: non-linear encoder, skipping (needs landmarks at run time)")
            continue
        P, q = sd.pop("encoder.P").to(a.device), sd.pop("encoder.q").to(a.device)

        model = MLP(in_dim=spec["r"], hidden_dims=spec["hidden_dims"],
                    num_classes=spec["num_classes"], activation="relu", batch_norm=False)
        model.load_state_dict(sd, strict=True)
        model = model.to(a.device).eval()

        ds = spec["dataset"]
        if ds not in cache:
            cache[ds] = load_flat_data(ds, a.device)
        _, _, Xte, yte = cache[ds]

        acc = accuracy_on_codes(model, Xte.float() @ P.t() + q, yte)

        dp, src = dense_params(spec, explore)
        dense = build_model(src["source_checkpoint"].replace(".safetensors", "")
                            .split("_", 1)[1], ds)
        dense.load_state_dict(load_file(str(CHECKPOINT_DIR / src["source_checkpoint"]),
                                        device="cpu"), strict=True)
        acc_d = accuracy_on_codes(dense.to(a.device).eval(), Xte, yte)

        enc_p = P.numel() + q.numel()
        net_p = sum(p.numel() for p in model.parameters())
        total = enc_p + net_p
        claim = spec["fully_compressed_acc"]
        ok = abs(acc - claim) < 0.02 and abs(acc_d - spec["dense_acc"]) < 0.02
        inwin = abs(acc_d - acc) <= a.tol
        bad += 0 if ok else 1

        print(f"\n{spec['source_compressed']}  [{spec['encoder_family']}]")
        print(f"  input        : {spec['input_dim']} -> {spec['r']}  "
              f"(×{spec['multiplier']:.2f} of GRIDE d_input={spec['r_input_gride']:.2f})")
        print(f"  hidden       : {spec['hidden_dims']}  (frozen)")
        print(f"  params       : {dp:,} -> {total:,}  (-{100*(1-total/dp):.1f}%)  "
              f"[encoder {enc_p:,} + net {net_p:,}]")
        print(f"  dense acc    : {acc_d:.2f}%   (recorded {spec['dense_acc']:.2f}%)")
        print(f"  hidden-only  : {spec['hidden_only_acc']:.2f}%")
        print(f"  FULLY compr. : {acc:.2f}%   (recorded {claim:.2f}%)")
        print(f"  drop vs dense: {acc_d - acc:+.2f} pts   within ±{a.tol}: "
              f"{'YES' if inwin else 'NO'}")
        print(f"  CWF          : {spec['cwf']:.3f}")
        print(f"  reproduces   : {'YES' if ok else 'NO -- MISMATCH'}")

    n = len([s for s in specs])
    print(f"\n{n - bad}/{n} verified.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
