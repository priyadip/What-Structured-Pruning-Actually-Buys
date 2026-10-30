# Hyperparameters

Two entirely separate sets of hyperparameters matter in this work, and confusing
them is the single easiest mistake to make when reading the paper.

The **training** hyperparameters produced the 24 vision checkpoints. They are
ordinary supervised training settings and they are upstream of everything the
paper claims.

The **compression** hyperparameters govern the operators the paper actually
studies. These operators use no gradients, no labels and no optimiser steps, so
this second set contains no learning rate, no epoch count and no scheduler. That
absence is a property of the method, not an omission from this document.

---

## 1. Training hyperparameters

Source of truth: `utils/config.py` (`TRAIN_DEFAULTS`) and `trainer/train.py`.

| Setting | Fully connected | Convolutional |
| --- | --- | --- |
| Optimiser | Adam | SGD with momentum |
| Learning rate | 1e-3 | 0.1 |
| Momentum | not applicable | 0.9 |
| Weight decay | 0.0 | 0.0 unless overridden per run |
| Scheduler | cosine annealing | 5 epochs linear warmup, then cosine |
| Warmup start factor | not applicable | 0.1 |
| Epochs | 30 | 30 |
| Batch size | 128 | 128 |
| Early stopping patience | 5 | 5 |
| Label smoothing | 0.0 | 0.0 |
| Global seed | 42 | 42 |

The optimiser is chosen by architecture family in `trainer/train.py`: SGD at
lr 0.1 when the model is convolutional, Adam at lr 1e-3 otherwise. Every value
above is overridable from the command line; the checkpoints in the manifest were
trained at the defaults.

### Preprocessing and augmentation

Normalisation statistics, from `utils/data_loader.py`:

| Dataset | Mean | Standard deviation |
| --- | --- | --- |
| MNIST | (0.1307,) | (0.3081,) |
| FashionMNIST | (0.2860,) | (0.3530,) |
| CIFAR-10 | (0.4914, 0.4822, 0.4465) | (0.2470, 0.2435, 0.2616) |
| CIFAR-100 | (0.5071, 0.4865, 0.4409) | (0.2673, 0.2564, 0.2762) |
| ImageNet32 | (0.481, 0.457, 0.408) | (0.260, 0.253, 0.268) |

Train-split augmentation, applied before `ToTensor` and `Normalize`:

| Dataset | Augmentation |
| --- | --- |
| MNIST | RandomRotation(10), RandomAffine(translate 0.1) |
| FashionMNIST | RandomHorizontalFlip, RandomAffine(degrees 5, translate 0.1) |
| CIFAR-10 | RandomCrop(32, padding 4), RandomHorizontalFlip |
| CIFAR-100 | RandomCrop(32, padding 4), RandomHorizontalFlip |
| ImageNet32 | RandomCrop(32, padding 4), RandomHorizontalFlip |

MNIST gets no horizontal flip because a mirrored digit is not a valid digit. The
test split receives no augmentation at all, which keeps evaluation, activation
extraction and intrinsic-dimension estimation deterministic with respect to the
input pipeline.

---

## 2. Compression hyperparameters

These are the settings the paper reports in Section 5 under *Protocol*.

### Common to all three studies

| Setting | Value | Where |
| --- | --- | --- |
| Budget rule | `M_l = round(alpha * d_l)`, clipped to `[4, N_l]` | Algorithm 1, line 5 |
| `d_l` | GRIDE intrinsic dimension of layer `l`'s unit cloud | `intrinsic_rank/id_estimator.py` |
| Ridge | `lambda = 1e-4`, relative to the mean diagonal of `U^T U` | all drivers |
| Intercept | free, never shrunk by the ridge | `ls_consuming` |
| Calibration data | unlabelled training inputs only | all drivers |
| Over-determination | every least-squares system at least 2x over-determined | enforced at runtime |
| Gradients, labels, optimiser steps | none | by construction |

The budget is swept by a single global multiplier `alpha` on a geometric ladder
whose range is derived at runtime from the layer dimensions, so different models
receive different absolute widths at the same `alpha`. Every contrast the paper
reports is paired at identical per-layer widths, which is what makes the
comparison independent of that choice.

### GRIDE estimator

| Setting | Value |
| --- | --- |
| Neighbour orders | 16 |
| Iterations | 8 (MLP driver), 10 (`ID_DEFAULTS`) |
| Estimate | median over neighbour-order scales |
| Seed | 0 by default, varied to 0, 1, 2 in the stability study |

### Per-study calibration

| Study | Calibration size | Notes |
| --- | --- | --- |
| Fully connected | up to 32768 examples | sized so every solve is over-determined |
| Convolutional | 1024 images by default, raised to whatever over-determination requires | patch budget applies on top |
| Transformer | 2048 sequences, 65536 token rows, max length 128 | padding positions excluded from the row pool |

The paper states the calibration range as 1153 to 32768 examples. The lower bound
is not a setting but an outcome: it is the smallest calibration size any run
actually used after the over-determination requirement was applied.

### Transformer specifics

| Setting | Value |
| --- | --- |
| Model | `typeform/distilbert-base-uncased-mnli` |
| Blocks | 6 |
| Feed-forward width | 3072 |
| Hidden size | 768 |
| Attention heads | 12 of 64 coordinates |
| Budget ladder | alpha in 8, 10, 12, 12.125, 13, 14, 16, 20 |
| Seeds | 0, 1, 2 |
| Targets | `dense` and `residual` |
| Sequential mode | on and off |

### BatchNorm

BatchNorm is folded into the preceding linear map before compression, so a unit
is the pair `(w, b)`. This matters because it makes the unit definition uniform
across studies. The Transformer has nothing to fold, since no normalisation sits
between producer and consumer inside the feed-forward block.

---

## 3. Random seeds

| Context | Seed |
| --- | --- |
| Training | 42 (`utils/config.py`, `SEED`) |
| Compression drivers, default | 0 |
| Calibration subsampling | same seed as the driver |
| GRIDE subsampling | same seed as the driver |
| k-means in merge operators | same seed as the driver |
| Stability study | 0, 1, 2 |

Seeds control calibration subsampling, GRIDE neighbour subsampling and k-means
initialisation. They do not make runs bit-identical, because cuDNN autotuning and
reduction order remain non-deterministic. The paper handles this by reporting
seed variance as a noise band in Figure 3 and by pairing every contrast within a
cell so the shared noise cancels.
