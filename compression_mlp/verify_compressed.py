#!/usr/bin/env python3
"""Independently re-check a saved compressed model.

Rebuilds the network from `results/*_compressed.safetensors` + its `.arch.json`
alone -- no state carried over from the run that produced it -- and re-evaluates
it on the test split. Also re-evaluates the dense source checkpoint, so the
reported drop is verified end to end rather than trusted.

    python3 compression_mlp/verify_compressed.py            # every saved model
    python3 compression_mlp/verify_compressed.py --tag cifar10_mlp3_quad_bn
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
sys.path.insert(0, str(PROJECT_ROOT))

from models.mlp import MLP  # noqa: E402
from models.model_factory import build_model  # noqa: E402
from utils.config import CHECKPOINT_DIR, DATASETS  # noqa: E402
from utils.data_loader import get_dataloaders  # noqa: E402


@torch.no_grad()
def accuracy(model, loader, device):
    model = model.to(device).eval()
    ok = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return 100.0 * ok / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="verify only this model tag")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    a = ap.parse_args()

    specs = sorted((HERE / "results").glob("*.arch.json"))
    if a.tag:
        specs = [p for p in specs if a.tag in p.name]
    if not specs:
        raise SystemExit("no compressed models found in compression_mlp/results/")

    bad = 0
    for spec_path in specs:
        spec = json.loads(spec_path.read_text())
        weights = spec_path.with_name(spec_path.name.replace(".arch.json", ".safetensors"))
        kind = spec_path.stem.rsplit("_", 1)[-1].replace(".arch", "")
        src = spec["source_checkpoint"]
        dataset = next(d for d in DATASETS if src.lower().startswith(d.lower() + "_"))
        model_name = Path(src).stem[len(dataset) + 1:]

        model = MLP(in_dim=spec["in_dim"], hidden_dims=spec["hidden_dims"],
                    num_classes=spec["num_classes"], activation=spec["activation"],
                    batch_norm=spec["batch_norm"])
        model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)

        dense = build_model(model_name, dataset)
        dense.load_state_dict(load_file(str(CHECKPOINT_DIR / src), device="cpu"), strict=True)

        _, test = get_dataloaders(dataset, batch_size=a.batch_size, num_workers=4, augment=False)
        acc_c = accuracy(model, test, a.device)
        acc_d = accuracy(dense, test, a.device)

        p_c = sum(p.numel() for p in model.parameters())
        p_d = sum(p.numel() for p in dense.parameters())
        claim = spec["compressed_acc"]
        ok = abs(acc_c - claim) < 0.01 and abs(acc_d - spec["original_acc"]) < 0.01
        bad += 0 if ok else 1

        print(f"\n{dataset}/{model_name} ({kind})  [{spec['strategy']} @ x{spec['multiplier']}]")
        print(f"  GRIDE d      : {[round(x, 2) for x in spec['gride_dims']]} "
              f"({spec['id_space']} cloud)")
        print(f"  widths       : {spec['original_widths']} -> {spec['hidden_dims']}")
        print(f"  params       : {p_d:,} -> {p_c:,}  (-{100*(1-p_c/p_d):.1f}%)")
        print(f"  dense acc    : {acc_d:.2f}%   (recorded {spec['original_acc']:.2f}%)")
        print(f"  compressed   : {acc_c:.2f}%   (recorded {claim:.2f}%)")
        print(f"  drop         : {acc_d - acc_c:+.2f} pts")
        print(f"  reproduces   : {'YES' if ok else 'NO -- MISMATCH'}")

    print(f"\n{len(specs) - bad}/{len(specs)} verified.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
