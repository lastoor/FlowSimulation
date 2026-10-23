#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${HOME}/.conda/envs/taichi/bin/python}"
output_dir="${OUTPUT_DIR:-${repo_root}/outputs/ergodicity-1}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin} (set PYTHON_BIN)" >&2
  exit 1
fi

cd "${repo_root}"
"${python_bin}" scripts/ergodicity_simulation.py \
  --output-dir "${output_dir}" \
  --num-realizations "${NUM_REALIZATIONS:-100}" \
  --num-particles "${NUM_PARTICLES:-1000000}" \
  --t-star "${T_STAR:-60}" \
  --ly "${LY:-20}" \
  --logk-variance "${LOGK_VARIANCE:-2}" \
  --source-width "${SOURCE_WIDTH:-1.5}" \
  --source-height "${SOURCE_HEIGHT:-6}" \
  --dx "${DX:-0.25}" \
  --dt "${DT:-0.05}" \
  --seed "${SEED:-0}" \
  --taichi-arch gpu \
  "$@"

"${python_bin}" scripts/ergodicity_analysis.py \
  --input-dir "${output_dir}" \
  --seed "${ANALYSIS_SEED:-0}"
