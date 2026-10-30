# Training Details

The paper trains nothing. Its compression operators are gradient-free and
label-free, and Algorithm 1 contains no optimiser step. Training enters this work
only because the 24 vision checkpoints had to exist before anything could be
compressed, and because a reviewer needs to know they were trained normally
rather than in some way that favours the result.

## What was trained

| Family | Architectures | Datasets | Count |
| --- | --- | --- | --- |
| Fully connected | mlp2_bn, mlp3_bn, mlp3_quad_bn, mlp3_oct_bn | MNIST, FashionMNIST, CIFAR-10, CIFAR-100 | 8 |
| Convolutional, VGG | vgg11_bn, vgg16_bn, vgg19_bn | FashionMNIST, CIFAR-10, CIFAR-100, ImageNet32 | 8 |
| Convolutional, ResNet | resnet18, resnet34, resnet50 | FashionMNIST, CIFAR-10, CIFAR-100, ImageNet32 | 8 |

The twenty-fifth checkpoint is DistilBERT fine-tuned for MNLI. It was not trained
here; it is the public checkpoint `typeform/distilbert-base-uncased-mnli`,
downloaded at run time. This matters for reproduction: the Transformer results
depend on a third-party artefact whose weights this project does not control.

The exact file list, with SHA-256 for each, is in
`trainer/checkpoint/CHECKPOINT_MANIFEST.csv`.

## How they were trained

Entry point `trainer/train.py`. Settings come from `utils/config.py`
(`TRAIN_DEFAULTS`) unless overridden on the command line.

```
python -m trainer.train --dataset CIFAR10 --model vgg11_bn --epochs 30
```

| Setting | Fully connected | Convolutional |
| --- | --- | --- |
| Optimiser | Adam | SGD, momentum 0.9 |
| Learning rate | 1e-3 | 0.1 |
| Scheduler | cosine annealing over 30 epochs | 5 epochs linear warmup from factor 0.1, then cosine |
| Epochs | 30 | 30 |
| Batch size | 128 | 128 |
| Weight decay | 0.0 | 0.0 |
| Early stopping | patience 5 on validation accuracy | patience 5 |
| Seed | 42 | 42 |

The optimiser is selected by architecture family inside `train.py`, not passed in.
Convolutional models get SGD at 0.1 with warmup; everything else gets Adam at
1e-3.

### One checkpoint departs from these defaults

`imagenet32_resnet50` was trained by a separate one-off runner, preserved at
`trainer/one_off/_resnet50_runner.py`, after the
original run was interrupted. Its settings differ:

| Setting | Default | `imagenet32_resnet50` |
| --- | --- | --- |
| Epochs | 30 | 40 |
| Batch size | 128 | 256 |
| Early stopping patience | 5 | 50, effectively disabled |
| Label smoothing | 0.0 | 0.1 |
| Weight decay | 0.0 | 1e-4 |
| Mixed precision | off | on |
| Dataloader workers | 2 | 12 |

This is recorded because a table of defaults that quietly excludes an exception
is worse than no table. The departure does not affect any claim: every contrast
in the paper is computed within a single checkpoint at a single budget, so the
training recipe cancels. It would matter only to someone comparing dense
accuracies across checkpoints, which the paper never does.

## Data splits

Standard published splits throughout. No custom split, no held-out subset carved
from train, and no test data used at any point before final evaluation.

| Dataset | Train | Test | Classes | Source |
| --- | --- | --- | --- | --- |
| MNIST | 60000 | 10000 | 10 | torchvision |
| FashionMNIST | 60000 | 10000 | 10 | torchvision |
| CIFAR-10 | 50000 | 10000 | 10 | Hugging Face parquet mirror |
| CIFAR-100 | 50000 | 10000 | 100 | Hugging Face parquet mirror |
| ImageNet32 | 1281167 | 50000 | 1000 | downsampled ImageNet, local files |
| MNLI | 392702 | 9815 matched | 3 | Hugging Face `nyu-mll/glue` |

CIFAR-10 and CIFAR-100 are pulled from a Hugging Face parquet mirror rather than
the University of Toronto server, through `utils/hf_datasets.py`. The images are
identical; only the transport differs. ImageNet32 ships as pickled batches and is
handled by `utils/imagenet32.py`; those files must be placed under
`data/raw/ImageNet32/` by hand, as they are not downloadable without an ImageNet
account.

## Preprocessing and augmentation

Full tables are in `hyperparameters.md`. The one point worth
repeating here is that augmentation is applied to the train split only. The test
split gets `ToTensor` and `Normalize` and nothing else, which keeps evaluation,
activation extraction and intrinsic-dimension estimation deterministic with
respect to the input pipeline.

Calibration for compression draws from the **train** split, unlabelled. No
compression operator at any point sees a test image or any label.

## Approximate training time

Measured on the single L40S described in `hardware.md`.

| Model group | Time per checkpoint | Total |
| --- | --- | --- |
| Fully connected, MNIST and FashionMNIST | 3 to 5 minutes | about 20 minutes |
| Fully connected, CIFAR | 10 to 15 minutes | about 50 minutes |
| VGG and ResNet on FashionMNIST and CIFAR | 25 to 60 minutes | about 6 hours |
| VGG and ResNet on ImageNet32 | 4 to 9 hours | about 24 hours |

Retraining everything is roughly 31 GPU hours. Retraining only the eight fully
connected checkpoints that Section 4 needs is roughly 70 minutes, which is the
relevant figure for a reviewer who wants to verify the mechanism result rather
than the whole paper.

## Why the checkpoints do not need to be bit-identical

Nothing in the paper's argument depends on a specific checkpoint. Every claim is
a **paired contrast** computed within a single model at a single budget, so a
reviewer who retrains and lands a fraction of a point away from the recorded
dense accuracy will still recover the same contrast. This is the property that
makes the result robust to the non-determinism described in the hardware note.
