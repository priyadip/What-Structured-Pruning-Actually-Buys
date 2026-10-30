# Hardware and Software Environment

Every number in the paper was produced on the single machine described here.
This is the same statement the paper makes in Section 5 under *Protocol*,
expanded to the full detail the AAAI reproducibility checklist asks for
(question 4.8).

## Compute

| Component | Value |
| --- | --- |
| GPU | NVIDIA L40S, 48 GB  |
| GPUs present | 2 (all work is single GPU; no data or model parallelism) |
| NVIDIA driver | 550.144.03 |
| CPU | Intel(R) Xeon(R) Gold 5418Y |
| CPU cores / threads | 24 cores per socket, 96 threads total |
| System RAM | 251 GiB (263594016 kB) |
| Operating system | Ubuntu 24.04.1 LTS |
| Kernel | 6.8.0-60-generic |

The 48 GB figure is the card specification. `nvidia-smi` reports 46068 MiB of
usable memory; the difference is the ECC reserve and is normal.

## Software

| Package | Version |
| --- | --- |
| Python | 3.13.12 |
| PyTorch | 2.7.1+cu126 |
| CUDA runtime (bundled with PyTorch) | 12.6 |
| cuDNN | 90501 |
| NumPy | 2.2.6 |
| SciPy | 1.17.1 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |
| safetensors | 0.8.0 |
| transformers | 5.14.1 |
| datasets | 4.8.4 |

`transformers` and `datasets` are required only by the Transformer study. The
vision studies and the whole of the Section 4 residual verification run without
them.

## What the hardware actually has to be

Nothing in this work needs an L40S specifically. The requirements are modest and
the constraint is memory rather than throughput.

| Stage | Requirement |
| --- | --- |
| Training the 24 vision checkpoints | Any CUDA GPU with 8 GB or more |
| MLP and CNN compression sweeps | 8 GB GPU; the CNN patch extraction is the peak |
| Transformer sweeps | 16 GB GPU at the default 65536 calibration rows |
| Section 4 residual verification | 8 GB GPU, or CPU if patient; it is double precision |
| Analysis, tables, figures | CPU only, no GPU, under a minute |

The analysis layer that produces every table and figure in the paper reads CSV
and JSON only. A reviewer who wants to check the paper's arithmetic rather than
rerun its experiments needs no GPU at all.

## Determinism

Runs are seeded (see `hyperparameters.md`) but not
bit-deterministic: cuDNN autotuning and non-deterministic reduction order in
`torch` mean two runs of the same seed can differ in the last digits. This is
why the paper reports seed variance as a noise band rather than claiming exact
reproduction, and why the central contrasts are paired within a cell, which
cancels the shared noise.
