"""Training helpers: early stopping and data loader convenience wrapper."""

from utils.data_loader import get_dataloaders


class EarlyStopping:
    """Stops training when a monitored metric hasn't improved for `patience`
    consecutive epochs. mode="min" (e.g. loss) or "max" (e.g. accuracy) —
    use whichever metric you actually care about maximizing/minimizing, since
    loss and accuracy don't always plateau on the same epoch."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def make_loaders(dataset_name: str, batch_size: int = None, num_workers: int = 2, augment: bool = True,
                 force_rgb: bool = False, image_size: int = None):
    """Thin wrapper kept in trainer/ so training scripts don't reach into utils/ directly."""
    return get_dataloaders(dataset_name, batch_size=batch_size, num_workers=num_workers,
                           augment=augment, force_rgb=force_rgb, image_size=image_size)
