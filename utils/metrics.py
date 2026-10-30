"""Shared evaluation metrics: accuracy, log-loss, and a couple of
distributional-distance helpers used to score compression quality."""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in [0, 1]."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def log_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean cross-entropy (natural log) over the batch."""
    return F.cross_entropy(logits, targets).item()


def quantization_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between original neuron signatures/activations and
    their compressed (quantized) reconstruction. Both arrays: (n_samples, dim)."""
    diff = original - reconstructed
    return float(np.mean(np.sum(diff ** 2, axis=-1)))


def relative_error(original: np.ndarray, reconstructed: np.ndarray, eps: float = 1e-12) -> float:
    """||original - reconstructed|| / ||original||, averaged over samples."""
    num = np.linalg.norm(original - reconstructed, axis=-1)
    den = np.linalg.norm(original, axis=-1) + eps
    return float(np.mean(num / den))
