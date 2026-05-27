# Environment

The code was developed with Python 3.11 and PyTorch. For a lightweight GitHub
release, the exact local machine environment is intentionally not encoded in
the scripts.

## Recommended Setup

```bash
conda create -n mvcpert python=3.11
conda activate mvcpert

# Install a CUDA-enabled PyTorch build if GPU training is needed.
# See https://pytorch.org/get-started/locally/ for the command matching your CUDA version.

python -m pip install -r requirements.txt
```

## Main Dependencies

See `requirements.txt` for pinned versions used during local packaging. For
replotting existing figure tables, the key packages are `numpy`, `pandas`,
`matplotlib`, `scipy`, `scikit-learn`, `h5py`, `pyarrow`, and `rdkit`. For model
training, `torch` is required.

## Hardware

The main model is intended for GPU training. CPU mode is useful for smoke tests
and CLI validation, but not for reproducing full training time.
