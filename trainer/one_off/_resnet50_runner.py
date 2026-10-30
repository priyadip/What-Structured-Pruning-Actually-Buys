"""One-off runner that trained `imagenet32_resnet50`.

Preserved for provenance: every checkpoint the paper uses should have the code
that produced it somewhere in this supplement. This run was restarted from a
partial state after an interruption, which is why it exists separately from
`trainer/train.py` and why its hyperparameters depart from the defaults. See
`docs/training.md` for the departure table.

Not part of the reproduction path; `trainer/train.py` trains the same model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer.train import train  # noqa: E402

train("ImageNet32", "resnet50", epochs=40, batch_size=256, patience=50,
      label_smoothing=0.1, weight_decay=1e-4, num_workers=12, amp=True,
      device="cuda")
