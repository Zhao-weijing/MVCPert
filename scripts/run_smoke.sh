#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${MVCPERT_BBBC047_H5:?Set MVCPERT_BBBC047_H5 to <DATA_ROOT>/BBBC047/Paired_CP_GE_Data_one_sample_per_molecule.h5}"
DATA_PATH="${MVCPERT_BBBC047_H5}"
OUTPUT_BASE="${MVCPERT_OUTPUT_BASE:-${ROOT_DIR}/runs_smoke}"
SPLIT_LOCK="${MVCPERT_SPLIT_LOCK:-${ROOT_DIR}/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json}"
CONFIG_FILE="${MVCPERT_SMOKE_CONFIG_FILE:-${ROOT_DIR}/configs/smoke_args.txt}"

cd "${ROOT_DIR}/source/baseline/src"

"${PYTHON_BIN}" train_mvc.py \
  $(tr '\n' ' ' < "${CONFIG_FILE}") \
  --data_path_override "${DATA_PATH}" \
  --output_base_dir "${OUTPUT_BASE}" \
  --split_lock_path "${SPLIT_LOCK}" \
  "$@"
