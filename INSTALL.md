# Installation

## Requirements

Python 3.10 or later. PyTorch 2.1 or later. A GPU is optional: the tier that
regenerates every table and figure in the paper runs on CPU.

The paper was produced with Python 3.13 and PyTorch 2.7.1 on CUDA 12.6. Exact
versions are pinned in `requirements.txt` and recorded in
`docs/hardware.md`.

## Option 1, pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Option 2, conda

```bash
conda env create -f environment.yml
conda activate structured-pruning-decomposition
```

## Minimal install

To regenerate the paper's tables and figures you need only:

```bash
pip install numpy matplotlib
```

The analysis layer imports the standard library plus `matplotlib`. It
deliberately does not use `pandas`, so it cannot silently reindex or coerce a
column. `torch` is needed only to touch checkpoints, and `transformers` only for
the Transformer study.

## Verify the install

```bash
cd analysis/scripts && python3 load_all.py
```

Expected:

```
records loaded: {'mlp': 2760, 'cnn': 677, 'transformer': 485} total 3922
distinct models: 25
```

If those three numbers appear, the whole CPU-only reproduction path will work.
See `REPRODUCE.md` next.

## Optional components

| Component | Needed for | Install |
| --- | --- | --- |
| `torch`, `safetensors` | anything touching checkpoints | in `requirements.txt` |
| `transformers`, `datasets` | the Transformer study only | in `requirements.txt` |
| `POT` | the optimal-transport merge baselines | in `requirements.txt` |
| `pdflatex` | rebuilding `paper.pdf` | system package, for example TeX Live |

## Datasets

None are required for the CPU-only tier. See section 7 of `README.md` for what
each study needs if you intend to rerun a sweep from scratch.
