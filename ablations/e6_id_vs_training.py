#!/usr/bin/env python3
"""E6 -- does the intrinsic dimension fall as a network trains?

The paper's ImageNet32 section has one confound it concedes: we cannot separate
"the task is genuinely complex, so the filters are genuinely high-dimensional"
from "a CIFAR-scale architecture is under-fitting 1.28M images at 48-62% top-1,
and an under-fitted network's filters have not organised yet". The two have
different implications -- an under-fitted network has no slack for a reason that
would go away with more training.

The diagnostic is to measure `d` against training time. If `d` **falls** while
accuracy rises, the high `d` is an under-fitting artefact and the paper's
ImageNet32 reading must be reframed. If `d` is flat or rises, the current
reading stands.

Running this on ImageNet32 means re-training (~20 GPU-hours). This script does
the *cheap* version of the same test first, because it is decisive in one
direction: train a CIFAR-100 model -- where we know the method works -- from
scratch with checkpoints, and measure `d` at each. CIFAR-100 sits in the middle
of the `N/d` range, so if `d` moves strongly with training there, the confound
is real everywhere and the expensive ImageNet32 run is justified. If `d` is
stable there, the confound is much less likely to be driving ImageNet32.

    python3 e6_id_vs_training.py --dataset CIFAR100 --model vgg11_bn --epochs 60
    python3 e6_id_vs_training.py --measure-only    # just measure saved epochs

Checkpoints go to `results/e6_ckpt/`, measurements to `results/e6_id_vs_epoch.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

import harness as H
from common import gride_conv as GC        # noqa: E402
from common import surgery as SG           # noqa: E402
from common.foldbn import fold_bn          # noqa: E402
from common.discovery import evaluate      # noqa: E402

sys.path.insert(0, str(H.ROOT))
from models.model_factory import build_model, conv_input_spec   # noqa: E402
from utils.data_loader import get_dataloaders                   # noqa: E402

CKPT = H.RESULTS / "e6_ckpt"
CKPT.mkdir(exist_ok=True)
OUT = H.RESULTS / "e6_id_vs_epoch.csv"
FIELDS = ["dataset", "model", "epoch", "test_acc", "mean_N_over_d", "dims",
          "widths", "seconds"]


def train_with_snapshots(dataset, model_name, epochs, snap_every, device, bs=128):
    """Standard conv recipe from `trainer/train.py` -- SGD 0.1, momentum 0.9,
    nesterov, wd 5e-4, label smoothing 0.1, 5-epoch warmup then cosine -- with a
    state_dict saved every `snap_every` epochs. Deliberately the project's own
    recipe so the trajectory is the one its checkpoints came from."""
    from safetensors.torch import save_file

    force_rgb, image_size = conv_input_spec(model_name, dataset)
    tr, te = get_dataloaders(dataset, batch_size=bs, num_workers=8,
                             force_rgb=force_rgb, image_size=image_size)
    model = build_model(model_name, dataset).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    warm = 5
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else
        0.5 * (1 + torch.cos(torch.tensor(
            (e - warm) / max(1, epochs - warm) * 3.14159265)).item()))

    saved = []
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        if ep % snap_every == 0 or ep == epochs:
            model.eval()
            correct = tot = 0
            with torch.no_grad():
                for x, y in te:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    tot += y.numel()
            acc = 100.0 * correct / tot
            p = CKPT / f"{dataset.lower()}_{model_name}_ep{ep:03d}.safetensors"
            save_file({k: v.detach().cpu().contiguous()
                       for k, v in model.state_dict().items()}, p)
            saved.append((ep, acc, p))
            print(f"  epoch {ep:3d}  test {acc:6.2f}%  -> {p.name}")
    return saved, model_name, dataset


@torch.no_grad()
def measure_id(ckpt_path, model_name, dataset, device, seed=0):
    """GRIDE of the effective neuron cloud at every compressible unit, using the
    same walk the paper uses -- `filter_proj`, filters projected onto the column
    space of the unfolded input."""
    from safetensors.torch import load_file
    from common.discovery import loaders_for, calibration_batch
    from types import SimpleNamespace

    force_rgb, image_size = conv_input_spec(model_name, dataset)
    tr, te = get_dataloaders(dataset, batch_size=256, num_workers=4,
                             force_rgb=force_rgb, image_size=image_size)
    model = build_model(model_name, dataset).to(device)
    model.load_state_dict(load_file(str(ckpt_path)))
    model.eval()

    acc = 0.0
    correct = tot = 0
    for x, y in te:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        tot += y.numel()
    acc = 100.0 * correct / tot

    family = "vgg" if "vgg" in model_name else "resnet"
    folded = fold_bn(model, family).to(device)
    probe = calibration_batch(tr, 8, device, seed=seed)
    need, info = SG.plan_calibration(folded, family, probe, 4.0)
    X = calibration_batch(tr, max(1024, need), device, seed=seed)

    dims, order, widths = [], [], [i["C"] for i in info]

    def visit(ctx):
        U, G = ctx.producer_patches(20000)
        clouds = GC.layer_clouds(ctx.filters_flat(), G, U, seed=seed)
        dims.append(GC.gride_id(clouds["filter_proj"], seed=seed))
        order.append(ctx.name)
        del U, G
        return ctx.A

    SG.Walker(folded, SG.clone_eval(folded), family, X, 60000, 1e-4, seed).walk(visit)
    nd = sum(w / d for w, d in zip(widths, dims) if d > 0) / len(dims)
    del model, folded, X
    H.free()
    return acc, dims, widths, nd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="CIFAR100")
    ap.add_argument("--model", default="vgg11_bn")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if not a.measure_only:
        print(f"training {a.model} on {a.dataset} for {a.epochs} epochs "
              f"(snapshot every {a.snap_every})")
        train_with_snapshots(a.dataset, a.model, a.epochs, a.snap_every, a.device)

    rows = []
    pat = f"{a.dataset.lower()}_{a.model}_ep*.safetensors"
    for p in sorted(CKPT.glob(pat)):
        ep = int(p.stem.split("_ep")[-1])
        t0 = time.time()
        acc, dims, widths, nd = measure_id(p, a.model, a.dataset, a.device, a.seed)
        rows.append({"dataset": a.dataset, "model": a.model, "epoch": ep,
                     "test_acc": round(acc, 4), "mean_N_over_d": round(nd, 4),
                     "dims": json.dumps([round(d, 3) for d in dims]),
                     "widths": json.dumps(widths),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  ep {ep:3d}  acc {acc:6.2f}%  mean N/d {nd:6.2f}  "
              f"deepest d {max(dims):6.1f}")
        _flush(rows)
    report(rows)


def _flush(rows):
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)


def report(rows):
    if len(rows) < 2:
        print("\n(need at least two snapshots to say anything)")
        return
    rows.sort(key=lambda r: r["epoch"])
    print("\n" + "=" * 66)
    print("E6 -- intrinsic dimension against training time")
    print("=" * 66)
    print(f"{'epoch':>6s} {'test acc':>9s} {'mean N/d':>9s} {'deepest d':>10s}")
    for r in rows:
        dm = json.loads(r["dims"])
        print(f"{r['epoch']:6d} {r['test_acc']:9.2f} {r['mean_N_over_d']:9.2f} "
              f"{max(dm):10.1f}")
    xs = [float(r["epoch"]) for r in rows]
    nd = [r["mean_N_over_d"] for r in rows]
    ac = [r["test_acc"] for r in rows]
    print("-" * 66)
    print(f"corr(epoch, mean N/d)     = {_pearson(xs, nd):+.3f}")
    print(f"corr(test acc, mean N/d)  = {_pearson(ac, nd):+.3f}")
    first, last = rows[0], rows[-1]
    print(f"mean N/d  {first['mean_N_over_d']:.2f} (ep {first['epoch']}, "
          f"{first['test_acc']:.1f}%)  ->  {last['mean_N_over_d']:.2f} "
          f"(ep {last['epoch']}, {last['test_acc']:.1f}%)")
    print("\nIf N/d RISES with training, filters organise onto a lower-dimensional")
    print("set as the model fits, and ImageNet32's low N/d is consistent with")
    print("under-fitting. If it is flat, the confound is unlikely to be driving it.")
    print(f"\nwrote {OUT}")


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


if __name__ == "__main__":
    main()
