"""Filesystem discovery of the VGG / ResNet checkpoints. Nothing hardcoded.

The dataset a checkpoint belongs to is *verified*, not assumed: the filename
prefix only orders the candidates, and identification is a strict
`load_state_dict`. A wrong dataset changes `num_classes` and therefore the
classifier shape, so a strict load that succeeds is real evidence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safetensors.torch import load_file  # noqa: E402

from models.model_factory import build_model, conv_input_spec, is_conv_model  # noqa: E402
from utils.config import CHECKPOINT_DIR, DATASETS  # noqa: E402
from utils.data_loader import get_dataloaders  # noqa: E402


@dataclass
class ConvCheckpoint:
    path: Path
    dataset: str
    model_name: str          # e.g. "vgg11_bn", "resnet18"
    family: str              # "vgg" | "resnet"
    model: nn.Module
    n_params: int
    force_rgb: bool
    image_size: int | None

    @property
    def tag(self) -> str:
        return self.path.stem


def split_stem(stem: str) -> tuple[str | None, str | None]:
    """`cifar10_vgg11_bn` -> ('CIFAR10', 'vgg11_bn'), using the project's own
    dataset registry as the vocabulary."""
    for ds in DATASETS:
        prefix = ds.lower() + "_"
        if stem.startswith(prefix):
            return ds, stem[len(prefix):]
    return None, None


def discover(ckpt_dir: Path = CHECKPOINT_DIR) -> list[tuple[Path, str]]:
    """Every `.safetensors` whose model part names a VGG or ResNet."""
    found = []
    for p in sorted(ckpt_dir.glob("*.safetensors")):
        _, model_name = split_stem(p.stem)
        if model_name and is_conv_model(model_name):
            found.append((p, model_name))
    found.sort(key=lambda t: t[0].stat().st_size, reverse=True)
    return found


def load_checkpoint(path: Path, model_name: str, device: str) -> ConvCheckpoint:
    state = load_file(str(path), device="cpu")
    hinted, _ = split_stem(path.stem)
    candidates = ([hinted] if hinted else []) + [d for d in DATASETS if d != hinted]

    errors = {}
    for ds in candidates:
        try:
            model = build_model(model_name, ds)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            errors[ds] = str(exc).splitlines()[0][:110]
            continue
        force_rgb, image_size = conv_input_spec(model_name, ds)
        try:
            get_dataloaders(ds, batch_size=8, num_workers=0, augment=False,
                            force_rgb=force_rgb, image_size=image_size)
        except Exception as exc:
            errors[ds] = f"weights fit but data unavailable: {exc}"
            continue
        return ConvCheckpoint(
            path=path, dataset=ds, model_name=model_name,
            family="vgg" if model_name.startswith("vgg") else "resnet",
            model=model.to(device).eval(),
            n_params=sum(p.numel() for p in model.parameters()),
            force_rgb=force_rgb, image_size=image_size)

    raise RuntimeError(f"Could not match {path.name} to any dataset. Tried: {errors}")


def loaders_for(ck: ConvCheckpoint, batch_size: int, workers: int):
    return get_dataloaders(ck.dataset, batch_size=batch_size, num_workers=workers,
                           augment=False, force_rgb=ck.force_rgb, image_size=ck.image_size)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, max_batches: int | None = None) -> float:
    model = model.to(device).eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def calibration_batch(loader, n: int, device: str, seed: int = 0) -> torch.Tensor:
    """`n` training images, in a *deterministic* order.

    The project's train loader is built with `shuffle=True`; seeding immediately
    before draining pins the permutation. Without this the calibration set --
    and therefore every GRIDE estimate and every width derived from it -- is
    different on each run, which is exactly the failure that made the MLP study's
    `r` grid irreproducible.
    """
    torch.manual_seed(seed)
    xs, got = [], 0
    for x, _ in loader:
        xs.append(x.to(device))
        got += x.shape[0]
        if got >= n:
            break
    return torch.cat(xs)[:n].contiguous()
