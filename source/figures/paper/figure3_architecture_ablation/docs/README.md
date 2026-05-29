# Figure 3 Architecture Ablation

This directory contains the Figure 3 plotting assets for the BBBC047 architecture and augmentation ablation figure.

## Files

- `scripts/generate_figure3_architecture_ablation.py`
  - canonical generator for Figure 3 outputs in this directory
- `outputs/figure3_bbbc047_architecture_ablation.pdf`
  - current Figure 3 export
- `analysis/`
  - intermediate CSV tables consumed by the generator

## Dependency layout

- Shared plotting style and common data-loading helpers live in `../shared/`.
- This directory is the canonical home for Figure 3 plotting assets under `figures/`.
- Historical report artifacts under `baseline/docs/reports/figures/2026-04-21-paper-positive-results/` are preserved as legacy backups.
