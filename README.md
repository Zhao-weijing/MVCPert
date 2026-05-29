# MVCPert Lightweight Release

This directory contains the core experimental and plotting code for the
`MVCPert_5_27` paper draft. It is prepared as a lightweight GitHub release:
the goal is to make the main model, training entry point, figure scripts, and
small tabular evidence easy to inspect and reuse. The paper source is not
included in this release.


This is not a full artifact mirror. Large datasets, checkpoints, prediction
profiles, and private run directories are intentionally excluded.


## Contents

- `source/baseline/src/`: model definitions, data loading utilities, and the
  main training script.
- `source/baseline/pathway_benchmark/`: core downstream gene/pathway analysis
  helpers used by the figure pipeline.
- `source/figures/paper/`: Figure 2-5 plotting scripts, small CSV/JSON inputs,
  and final PDF outputs.
- `configs/`: main and smoke-test command-line configurations.
- `scripts/`: convenience wrappers for training, plotting, and release hygiene
  checks.

## Quick Start

```bash
cd reproducibility/MVCPert_5_27
python -m pip install -r requirements.txt

# Optional: run a CLI/import smoke check.
python source/baseline/src/train_mvc.py --help

# Check that no private paths or large artifacts were accidentally included.
python scripts/check_release_ready.py
```

To run training or regenerate all figures, provide local data/artifact paths
through environment variables:

```bash
export MVCPERT_BBBC047_H5=/path/to/data/BBBC047/Paired_CP_GE_Data_one_sample_per_molecule.h5
export MVCPERT_OUTPUT_BASE=/path/to/output/runs
export MVCPERT_ARTIFACT_ROOT=/path/to/external/artifacts
export MVCPERT_DATA_ROOT=/path/to/data

bash scripts/run_smoke.sh --dev cpu
bash scripts/run_mvcpert_main.sh --dev cuda:0
bash scripts/regenerate_figures.sh
```

Figure 2 and Figure 3 can mostly be regenerated from the small analysis tables
included in this release. Figure 4 and Figure 5 require external prediction
profiles and metadata files.

## Main Model Entry Points

- Training script: `source/baseline/src/train_mvc.py`
- Main model: `source/baseline/src/MVCModel_HyperGate.py`
- Residual VAE response refiner:
  `source/baseline/src/model_MVC_residual_vae.py`
- Data loader: `source/baseline/src/dataset.py`
- Split lock:
  `source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json`

## Privacy Note

All repository-facing documentation uses placeholder paths such as
`<DATA_ROOT>`, `<ARTIFACT_ROOT>`, and `<PROJECT_ROOT>`. Do not commit local
server paths, user names, checkpoints, HDF5 files, parquet tables, or cached
Python outputs.
