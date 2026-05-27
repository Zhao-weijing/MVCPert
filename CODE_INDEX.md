# Code Index

## Core Model and Training

| Component | Path | Notes |
| --- | --- | --- |
| Training entry point | `source/baseline/src/train_mvc.py` | Main CLI for training, validation, prediction export, and metric summaries. |
| HyperGate fusion model | `source/baseline/src/MVCModel_HyperGate.py` | Modality encoders, fusion layers, hypergraph refinement, and task heads. |
| Residual VAE refiner | `source/baseline/src/model_MVC_residual_vae.py` | Shared/private residual response module and missing-modality branches. |
| Base MVC model | `source/baseline/src/model_MVC.py` | Baseline multimodal model used by ablations. |
| Data loading | `source/baseline/src/dataset.py` | HDF5 loading, molecule features, normalization, and split utilities. |
| Utility functions | `source/baseline/src/utils.py` | Metrics, HDF5 helpers, and training utilities. |
| Split lock | `source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json` | Molecule-held-out split metadata. |

## Analysis

| Component | Path |
| --- | --- |
| Gene-level downstream evaluation | `source/baseline/pathway_benchmark/evaluate_bbbc047_gene_level_downstream.py` |
| Fit-vs-pathway diagnostic | `source/baseline/pathway_benchmark/diagnose_bbbc047_model_fit_vs_pathway_gap.py` |
| Shared pathway helpers | `source/baseline/pathway_benchmark/bbbc047_pathway_benchmark.py` |
| Pathway knowledge tables | `source/baseline/pathway_benchmark/knowledge/` |

## Figure Scripts

| Figure | Script | Included inputs |
| --- | --- | --- |
| Figure 2 | `source/figures/paper/figure2_multimodal/scripts/generate_figure2_multimodal.py` | `source/figures/paper/figure2_multimodal/analysis/` |
| Figure 3 | `source/figures/paper/figure3_architecture_ablation/scripts/generate_figure3_architecture_ablation.py` | `source/figures/paper/figure3_architecture_ablation/analysis/` |
| Figure 4 | `source/figures/paper/figure4_missing_modality_virtual_profiling/scripts/generate_figure4_missing_modality_virtual_profiling.py` | Small summary JSON plus external prediction profiles |
| Figure 5 | `source/figures/paper/figure5_case_studies/scripts/generate_figure5_case_studies.py` | Small selected-case CSVs plus external metadata/profiles |
