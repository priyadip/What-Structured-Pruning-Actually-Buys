"""Generic dataset loading for the 4-5 benchmark datasets used in this project.

Wraps torchvision datasets behind a single `get_dataloaders(name, ...)` call so
every downstream script (training, activation extraction, ID estimation) can
stay dataset-agnostic.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.config import RAW_DATA_DIR, TRAIN_DEFAULTS
from utils.hf_datasets import HFParquetImageDataset
from utils.imagenet32 import ImageNet32Dataset

_DATASET_CLASSES = {
    "MNIST": datasets.MNIST,
    "FashionMNIST": datasets.FashionMNIST,
    # CIFAR10/CIFAR100 are NOT loaded via torchvision.datasets here — see
    # _HF_BACKED_DATASETS below. torchvision's own classes download from a
    # single, frequently very slow University of Toronto server; we instead
    # pull the same images from Hugging Face's CDN via utils/hf_datasets.py.
    "CIFAR10": datasets.CIFAR10,
    "CIFAR100": datasets.CIFAR100,
    # ImageNet32 (downsampled ImageNet, Chrabaszcz et al.) ships as pickled
    # batches; handled by utils.imagenet32.ImageNet32Dataset (see below), not a
    # torchvision class. Files must be placed under data/raw/ImageNet32/.
    "ImageNet32": None,
}

# Datasets loaded through utils.hf_datasets.HFParquetImageDataset instead of
# their torchvision class (see comment above).
_HF_BACKED_DATASETS = {"CIFAR10", "CIFAR100"}

_NORMALIZE_STATS = {
    "MNIST": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "ImageNet32": ((0.481, 0.457, 0.408), (0.260, 0.253, 0.268)),
}

# Train-only augmentation, applied before ToTensor/Normalize (never applied to
# the test split, so evaluation/activation-extraction/ID-analysis stay
# deterministic). MLPs have no built-in translation/rotation invariance the
# way convolutions do, so this matters even more for them than for CNNs.
# No horizontal flip for MNIST — flipped digits (e.g. a mirrored "2") aren't
# valid digit shapes; flips are fine for clothing (FashionMNIST) and natural
# images (CIFAR).
_TRAIN_AUGMENTATIONS = {
    "MNIST": [transforms.RandomRotation(10), transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))],
    "FashionMNIST": [transforms.RandomHorizontalFlip(),
                     transforms.RandomAffine(degrees=5, translate=(0.1, 0.1))],
    "CIFAR10": [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
    "CIFAR100": [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
    "ImageNet32": [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
}


def get_transform(name: str, train: bool = False, force_rgb: bool = False, image_size: int = None):
    """Build the transform pipeline for a dataset split.

    force_rgb / image_size adapt a 1-channel dataset (MNIST/FashionMNIST) to the
    3-channel, 32x32 input that the ported VGG/ResNet architectures hardcode —
    grayscale is replicated to 3 channels and the image resized. Normalization
    stats are replicated to 3 channels to match.
    """
    mean, std = _NORMALIZE_STATS[name]
    ops = []

    force_rgb = force_rgb and len(mean) == 1  # only meaningful for 1-channel sources
    if force_rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))
        mean, std = mean * 3, std * 3
    if image_size is not None:
        ops.append(transforms.Resize(image_size))

    if train:
        ops += list(_TRAIN_AUGMENTATIONS.get(name, []))
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_dataloaders(name: str, batch_size: int = None, num_workers: int = 2, augment: bool = True,
                    force_rgb: bool = False, image_size: int = None):
    """Return (train_loader, test_loader) for a dataset name in utils.config.DATASETS.
    The train split gets dataset-appropriate augmentation unless augment=False;
    the test split never does. force_rgb/image_size adapt 1-channel datasets to the
    3-channel 32x32 input the VGG/ResNet architectures require (see get_transform)."""
    if name not in _DATASET_CLASSES:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(_DATASET_CLASSES)}")

    batch_size = batch_size or TRAIN_DEFAULTS["batch_size"]
    root = RAW_DATA_DIR / name
    train_transform = get_transform(name, train=augment, force_rgb=force_rgb, image_size=image_size)
    test_transform = get_transform(name, train=False, force_rgb=force_rgb, image_size=image_size)

    if name == "ImageNet32":
        # Val split is used as the test split (ImageNet has no public labeled test set).
        train_set = ImageNet32Dataset(root, split="train", transform=train_transform)
        test_set = ImageNet32Dataset(root, split="val", transform=test_transform)
    elif name in _HF_BACKED_DATASETS:
        train_set = HFParquetImageDataset(name, "train", root, transform=train_transform)
        test_set = HFParquetImageDataset(name, "test", root, transform=test_transform)
    else:
        dataset_cls = _DATASET_CLASSES[name]
        if dataset_cls is None:
            raise NotImplementedError(
                f"{name} needs a custom Dataset (no native torchvision class) — implement and register it here."
            )
        train_set = dataset_cls(root=str(root), train=True, download=True, transform=train_transform)
        test_set = dataset_cls(root=str(root), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
