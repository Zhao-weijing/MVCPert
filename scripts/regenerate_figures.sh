#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export MVCPERT_AIDD_ROOT="${MVCPERT_AIDD_ROOT:-${ROOT_DIR}/source}"
export MVCPERT_ARTIFACT_ROOT="${MVCPERT_ARTIFACT_ROOT:-<ARTIFACT_ROOT>}"
export MVCPERT_DATA_ROOT="${MVCPERT_DATA_ROOT:-<DATA_ROOT>}"

run_figure() {
  local script_path="$1"
  echo "[MVCPert] running ${script_path}"
  (cd "$(dirname "${script_path}")" && "${PYTHON_BIN}" "$(basename "${script_path}")")
}

run_figure "${ROOT_DIR}/source/figures/paper/figure2_multimodal/scripts/generate_figure2_multimodal.py"
run_figure "${ROOT_DIR}/source/figures/paper/figure3_architecture_ablation/scripts/generate_figure3_architecture_ablation.py"
run_figure "${ROOT_DIR}/source/figures/paper/figure4_missing_modality_virtual_profiling/scripts/generate_figure4_missing_modality_virtual_profiling.py"
run_figure "${ROOT_DIR}/source/figures/paper/figure5_case_studies/scripts/generate_figure5_case_studies.py"
