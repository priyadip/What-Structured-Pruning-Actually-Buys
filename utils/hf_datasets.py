"""CIFAR10/CIFAR100 loader backed by Hugging Face's CDN-hosted parquet mirror
of the official datasets (uoft-cs/cifar10, uoft-cs/cifar100), used instead of
torchvision's built-in downloader.

torchvision's CIFAR10/CIFAR100 classes download from a single University of
Toronto department server (cave.cs.toronto.edu) with no CDN behind it, which
is frequently rate-limited to tens of KB/s under global load. The Hugging
Face mirror serves the same images from a CDN at several MB/s.
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

_HF_PARQUET_URLS = {
    "CIFAR10": {
        "train": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
        "test": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    },
    "CIFAR100": {
        "train": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
        "test": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
    },
}

# CIFAR100 ships both a 100-class ("fine") and 20-class ("coarse") label;
# this project's num_classes=100 default (see model_factory.py) uses fine.
_LABEL_COLUMN = {
    "CIFAR10": "label",
    "CIFAR100": "fine_label",
}


def download_parquet(name: str, split: str, root: Path) -> Path:
    """Downloads root/<split>.parquet from the HF mirror if not already present."""
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{split}.parquet"
    if dest.exists():
        return dest

    url = _HF_PARQUET_URLS[name][split]
    tmp = dest.with_suffix(".parquet.part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


class HFParquetImageDataset(Dataset):
    """Decodes the PNG-encoded 'img' column of a CIFAR10/CIFAR100 parquet
    split into (transformed_image, label) pairs, matching the interface
    torchvision's own image datasets expose."""

    def __init__(self, name: str, split: str, root: Path, transform=None):
        parquet_path = download_parquet(name, split, root)
        df = pd.read_parquet(parquet_path)
        self.images = df["img"].tolist()  # each entry: {'bytes': <PNG bytes>, 'path': None}
        self.labels = df[_LABEL_COLUMN[name]].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
