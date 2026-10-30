"""ImageNet32 — downsampled 32x32 ImageNet (Chrabaszcz et al., 2017).

Not a torchvision-native dataset, so the Dataset is implemented here and wired
into utils/data_loader.py. Expects the official files under
data/raw/ImageNet32/ (place them there manually — the source requires an
image-net.org account, so there is no automatic download):

- Train: 10 pickle files `train_data_batch_1` .. `train_data_batch_10`, each a
  dict with 'data' (uint8, N x 3072) and 'labels' (list, 1-indexed 1..1000).
- Val:   `Imagenet32_val_npz/val_data.npz` with 'data' (uint8, 50000 x 3072)
  and 'labels' (int64, 1-indexed). Used as the test split (ImageNet has no
  public labeled test set).

Format details that are easy to get wrong:
- Each 'data' row is channel-major: 3072 = 3 x 1024, so reshape(3, 32, 32)
  gives (C, H, W); we transpose to (H, W, C) for PIL.
- Labels are 1-indexed (1..1000); we convert to 0-indexed (0..999) so they
  line up with nn.CrossEntropyLoss / a 1000-way classifier head.
"""

import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

_NUM_TRAIN_BATCHES = 10


class ImageNet32Dataset(Dataset):
    def __init__(self, root, split: str = "train", transform=None):
        root = Path(root)
        self.transform = transform

        if split == "train":
            data_parts, label_parts = [], []
            for i in range(1, _NUM_TRAIN_BATCHES + 1):
                path = root / f"train_data_batch_{i}"
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing ImageNet32 train batch {path}. Extract Imagenet32_train.zip "
                        f"into {root} (it should yield train_data_batch_1..10)."
                    )
                with open(path, "rb") as f:
                    d = pickle.load(f)
                data_parts.append(d["data"])
                label_parts.append(np.asarray(d["labels"], dtype=np.int64))
            self.data = np.concatenate(data_parts, axis=0)      # (~1.28M, 3072) uint8
            self.labels = np.concatenate(label_parts, axis=0)
        elif split in ("val", "test"):
            npz_path = root / "Imagenet32_val_npz" / "val_data.npz"
            if not npz_path.exists():
                matches = list(root.rglob("val_data.npz"))
                if not matches:
                    raise FileNotFoundError(
                        f"Missing ImageNet32 val data (expected {npz_path}). Extract "
                        f"Imagenet32_val_npz.zip into {root}."
                    )
                npz_path = matches[0]
            z = np.load(npz_path)
            self.data = z["data"]
            self.labels = np.asarray(z["labels"], dtype=np.int64)
        else:
            raise ValueError(f"split must be 'train' or 'val'/'test', got '{split}'")

        self.labels = self.labels - 1  # 1-indexed (1..1000) -> 0-indexed (0..999)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # (C,H,W) -> (H,W,C) uint8
        img = Image.fromarray(img)  # RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])
