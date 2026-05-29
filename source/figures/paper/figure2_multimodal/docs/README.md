# Figure 2 Multimodal

This directory contains the Figure 2 plotting assets for the multimodal BBBC047/BBBC036 result set.

## Files

- `scripts/generate_figure2_multimodal.py`
  - canonical generator for Figure 2 outputs in this directory
- `outputs/figure2_multimodal_formal_panels.pdf`
  - formal multi-panel Figure 2 export
- `analysis/`
  - intermediate CSV tables consumed by the generator

## Dependency layout

- Shared plotting style and common data-loading helpers live in `../shared/`.
- New paper figures should follow the same structure:
  - `figureX_name/scripts`
  - `figureX_name/outputs`
  - `figureX_name/analysis`
