"""Global paths and hyperparameters shared across the pipeline.

Every other module should read paths/settings from here instead of
hardcoding strings, so the project can be moved or re-rooted by changing
only this file (or the OT_NEURON_COMPRESSION_ROOT env var).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("OT_NEURON_COMPRESSION_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = PROJECT_ROOT / "trainer" / "checkpoint"
ACTIVATION_STORAGE_DIR = PROJECT_ROOT / "activation_weight_matrices" / "storage"

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

DATASETS = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "ImageNet32"]

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if os.environ.get("FORCE_CPU") != "1" else "cpu"

TRAIN_DEFAULTS = {
    "batch_size": 128,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "early_stopping_patience": 5,
}

# Compression ratios (m / original width) swept during scaling analysis.
COMPRESSION_M_GRID = [2, 4, 8, 16, 32, 64, 128, 256]

# Intrinsic-dimension estimator defaults (GRIDE / TwoNN).
ID_DEFAULTS = {
    "n_neighbors": 16,
    "n_iters": 10,
}


def ensure_dirs():
    """Create all directories this config points to, if missing."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, ACTIVATION_STORAGE_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        (RAW_DATA_DIR / name).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DATA_DIR / name).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
