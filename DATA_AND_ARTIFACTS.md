# Data and External Artifacts

Large data and model outputs are not included in this lightweight release.
Publish them separately through an artifact service such as Zenodo, Figshare,
OSF, Hugging Face Hub, or a GitHub release asset.

## Required Data

Minimum files for the main BBBC047 run:

```text
<DATA_ROOT>/BBBC047/Paired_CP_GE_Data_one_sample_per_molecule.h5
```

Additional files used by selected analyses and figures:

```text
<DATA_ROOT>/BBBC047/Paired_CP_GE_Data.h5
<DATA_ROOT>/BBBC047/merged_repurposing.csv
<DATA_ROOT>/BBBC047/Step2_CP_cleaned.parquet
<DATA_ROOT>/BBBC047/Step2_GE_cleaned.parquet
<DATA_ROOT>/BBBC047/gene_map.csv
<DATA_ROOT>/BBBC036/Paired_CP_GE_Data_official.h5
```

## External Model Outputs

Training and figure regeneration can require:

```text
<ARTIFACT_ROOT>/.../best_model.pt
<ARTIFACT_ROOT>/.../run_summary.json
<ARTIFACT_ROOT>/.../predict/test_prediction_profile.h5
<ARTIFACT_ROOT>/.../predict/test_ps_all_views_prediction_profile.h5
<ARTIFACT_ROOT>/.../predict/test_ps_latents.npz
```

Do not commit these files to Git. Use `MVCPERT_ARTIFACT_ROOT` and
figure-specific environment variables to point scripts to local copies.

## HDF5 Field Convention

The main training HDF5 should include:

```text
canonical_smiles
control_CP
target_CP
control_GE
target_GE
```

If dose-aware features are used, include:

```text
pert_dose
```

The training code applies the split lock by canonical SMILES and estimates
normalization statistics on the training split.
