#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$PROJECT_ROOT"
export G1_SUPERMIMIC_HOME="$PROJECT_ROOT"
unset PYTHONHOME
export PYTHONPATH="$PROJECT_ROOT/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${G1_CPU_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/cuda"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR"
export WANDB_MODE=disabled
if [ -n "${G1_PYTHON:-}" ]; then
  PYTHON_BIN="$G1_PYTHON"
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN=python3
fi
exec "$PYTHON_BIN" -m g1_supermimic "$@"
