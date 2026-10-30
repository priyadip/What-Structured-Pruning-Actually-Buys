"""Standalone accuracy/loss evaluation for a saved checkpoint — used both for
the original (uncompressed) model and for compressed models produced by
ot_compression/, so the two are scored with exactly the same code path."""

import argparse

import torch
import torch.nn as nn
from safetensors.torch import load_file

from models.model_factory import build_model, conv_input_spec
from trainer.utils import make_loaders
from utils.config import CHECKPOINT_DIR, DEVICE
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(model: nn.Module, dataset: str, model_name: str = None, batch_size: int = None,
             device: str = DEVICE):
    # Match the input adaptation used at train time (conv nets on 1-channel
    # datasets need RGB+resize); no-op when model_name is None or non-conv.
    force_rgb, image_size = conv_input_spec(model_name, dataset) if model_name else (False, None)
    _, test_loader = make_loaders(dataset, batch_size=batch_size, force_rgb=force_rgb, image_size=image_size)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device).eval()

    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return {"loss": total_loss / n, "accuracy": total_acc / n}


def load_checkpoint(dataset: str, model_name: str, device: str = DEVICE):
    model = build_model(model_name, dataset)
    ckpt_path = CHECKPOINT_DIR / f"{dataset.lower()}_{model_name}.safetensors"
    model.load_state_dict(load_file(ckpt_path, device=device))
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True,
                   help="Any name accepted by models.model_factory.build_model, "
                        "e.g. mlp, mlp3_doub_bn, cnn, lstm, vgg11_bn, resnet18.")
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = load_checkpoint(args.dataset, args.model)
    metrics = evaluate(model, args.dataset, model_name=args.model, batch_size=args.batch_size)
    print(f"[{args.dataset}/{args.model}] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
