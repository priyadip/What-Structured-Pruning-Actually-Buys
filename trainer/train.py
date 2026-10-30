"""Generic training loop with logging, used for every (dataset, model) pair.

Usage:
    python -m trainer.train --dataset MNIST --model mlp --epochs 30
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file

from models.model_factory import build_model, conv_input_spec, is_conv_model
from trainer.utils import EarlyStopping, make_loaders
from utils.config import CHECKPOINT_DIR, DEVICE, TRAIN_DEFAULTS
from utils.metrics import accuracy


def _make_optimizer(model, optimizer_name, lr, weight_decay, momentum):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=momentum > 0)
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{optimizer_name}'. Options: sgd, adam, adamw")


def _make_scheduler(optimizer, epochs, warmup_epochs=0):
    """Cosine annealing over `epochs`, optionally preceded by a linear LR warmup.
    Warmup stabilizes deep conv nets (esp. VGG) at the high initial SGD LR."""
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    if warmup_epochs <= 0:
        return cosine
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    return optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def train(dataset: str, model_name: str, epochs: int = None, batch_size: int = None,
          lr: float = None, patience: int = None, device: str = DEVICE, use_scheduler: bool = True,
          augment: bool = True, optimizer_name: str = None, momentum: float = 0.9,
          weight_decay: float = None, label_smoothing: float = 0.0, num_workers: int = 2,
          amp: bool = True, **model_kwargs):
    epochs = epochs or TRAIN_DEFAULTS["epochs"]
    patience = patience or TRAIN_DEFAULTS["early_stopping_patience"]

    # Conv nets (VGG/ResNet) train best with SGD+momentum+weight_decay on CIFAR;
    # MLPs default to Adam. Pick sensible per-family defaults unless overridden.
    conv = is_conv_model(model_name)
    optimizer_name = optimizer_name or ("sgd" if conv else "adam")
    lr = lr or (0.1 if optimizer_name == "sgd" else TRAIN_DEFAULTS["lr"])
    weight_decay = weight_decay if weight_decay is not None else (5e-4 if conv else TRAIN_DEFAULTS["weight_decay"])

    # BF16 autocast on CUDA ~halves conv-net step time on the L40S and is
    # numerically safe (same exponent range as FP32, so no GradScaler needed;
    # BatchNorm stays FP32 under autocast automatically).
    use_amp = amp and str(device).startswith("cuda")

    # 1-channel datasets (MNIST/FashionMNIST) get grayscale->RGB + resize to 32
    # so the 3x32x32-only VGG/ResNet stem accepts them; no-op for CIFAR/etc.
    force_rgb, image_size = conv_input_spec(model_name, dataset)
    train_loader, test_loader = make_loaders(dataset, batch_size=batch_size, augment=augment,
                                             force_rgb=force_rgb, image_size=image_size,
                                             num_workers=num_workers)
    model = build_model(model_name, dataset, **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _make_optimizer(model, optimizer_name, lr, weight_decay, momentum)
    scheduler = _make_scheduler(optimizer, epochs, warmup_epochs=5 if conv else 0) if use_scheduler else None
    # Early-stop on validation accuracy plateauing, not loss — cross-entropy
    # loss can start rising from overconfidence while accuracy keeps
    # climbing, which would otherwise cut training short prematurely.
    stopper = EarlyStopping(patience=patience, mode="max")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{dataset.lower()}_{model_name}.safetensors"

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        if scheduler is not None:
            scheduler.step()

        val_loss, val_acc = evaluate_loop(model, test_loader, criterion, device, use_amp=use_amp)
        print(f"[{dataset}/{model_name}] epoch {epoch:03d} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time() - t0:.1f}s)")

        # Best checkpoint tracked by validation accuracy (the target metric),
        # not loss — the two don't always peak on the same epoch.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(state_dict, ckpt_path)

        if stopper.step(val_acc):
            print(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
            break

    print(f"[{dataset}/{model_name}] Saved best checkpoint to {ckpt_path} (best_val_acc={best_val_acc:.4f})")
    return ckpt_path, best_val_acc


@torch.no_grad()
def evaluate_loop(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_acc += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_loss / n, total_acc / n


def parse_args():
    p = argparse.ArgumentParser(description="Train a model on a dataset for OT neuron compression.")
    p.add_argument("--dataset", required=True, choices=[
        "MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "ImageNet32"])
    p.add_argument("--model", required=True,
                   help="Any name accepted by models.model_factory.build_model, "
                        "e.g. mlp, mlp3_doub_bn, cnn, lstm, vgg11_bn, resnet18.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--optimizer", default=None, choices=["sgd", "adam", "adamw"],
                   help="Defaults to sgd for conv nets (VGG/ResNet), adam otherwise.")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=None,
                   help="Dropout probability forwarded to the model constructor (MLP only).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_kwargs = {"dropout": args.dropout} if args.dropout is not None else {}
    train(args.dataset, args.model, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, patience=args.patience, optimizer_name=args.optimizer, momentum=args.momentum,
          weight_decay=args.weight_decay, label_smoothing=args.label_smoothing, **model_kwargs)
