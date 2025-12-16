#!/usr/bin/env bash
set -euo pipefail
set -x

usage() {
  cat >&2 <<'EOF'
Usage:
  bash fid-eval.sh <REF_NPZ> <EVAL_NPZ>

Optional env:
  OMP_NUM_THREADS=8          (default: 8)
  PYTHON_BIN=python          (default: python)
  CONDA_SH=/path/to/conda.sh (optional; if set, will be sourced)
  CONDA_ENV=your_env_name    (optional; if set, will conda activate)
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

REF_NPZ="$1"
EVAL_NPZ="$2"

if [[ ! -f "$REF_NPZ" ]]; then
  echo "ERROR: REF_NPZ not found: $REF_NPZ" >&2
  exit 1
fi
if [[ ! -f "$EVAL_NPZ" ]]; then
  echo "ERROR: EVAL_NPZ not found: $EVAL_NPZ" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Always run from the LlamaGen project root so relative paths work.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional conda activation (pure bash; no Slurm/srun dependency).
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}"
  fi
fi

exec "${PYTHON_BIN}" evaluations/c2i/evaluator.py "${REF_NPZ}" "${EVAL_NPZ}"