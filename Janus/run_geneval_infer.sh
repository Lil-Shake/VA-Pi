#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Run Janus GenEval inference (pure bash, no Slurm), following the provided Slurm script logic.

Usage:
  bash Janus/run_geneval_infer.sh \
    --prompts-dir /path/to/evaluation_metadata_geneval.jsonl \
    --model-path /path/to/janus_ckpt_dir_or_hf_repo \
    --base-model-path /path/to/Janus-Pro-1B_or_hf_repo \
    --reason-prompt /path/to/reasoning_prompt.txt \
    --save-root /path/to/output_geneval_samples

Required:
  --prompts-dir PATH      JSONL file (each line is a JSON obj containing prompt/text/Prompt)
  --model-path PATH       Janus checkpoint dir or HF repo/path (can be empty to use base model only)

Optional:
  --base-model-path PATH  Base HF model repo/path when --model-path is a DS checkpoint dir
                          (default: deepseek-ai/Janus-Pro-1B)
  --reason-prompt PATH    reasoning_prompt.txt path (default: required by python; pass it)
  --save-root PATH        Output root directory. If omitted, auto-generate under ./outputs/
  --semantic-cot          Enable semantic CoT (passes --semantic_cot True)
  --image-size INT        (default: 384)
  --temperature FLOAT     (default: 1.0)
  --nproc INT             GPUs per node / processes (default: 8)

Env (optional conda activation; you can also activate manually before running):
  CONDA_SH=/path/to/conda.sh
  CONDA_ENV=ar-grpo

Note:
  reason_inference_geneval.py hard-codes MASTER_ADDR=127.0.0.1 and MASTER_PORT=29500, so we
  use torchrun --master_addr=127.0.0.1 --master_port=29500 to match.
EOF
}

PROMPTS_DIR=""
MODEL_PATH=""
BASE_MODEL_PATH="deepseek-ai/Janus-Pro-1B"
REASON_PROMPT_PATH=""
SAVE_ROOT=""
SEMANTIC_COT="false"

IMAGE_SIZE="384"
TEMPERATURE="1.0"
NPROC="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --prompts-dir|--prompts_dir) PROMPTS_DIR="${2:-}"; shift 2 ;;
    --model-path) MODEL_PATH="${2:-}"; shift 2 ;;
    --base-model-path) BASE_MODEL_PATH="${2:-}"; shift 2 ;;
    --reason-prompt|--reasoning-prompt|--reasoning-prompt-path|--reasoning_prompt_path) REASON_PROMPT_PATH="${2:-}"; shift 2 ;;
    --save-root|--save_root) SAVE_ROOT="${2:-}"; shift 2 ;;
    --semantic-cot) SEMANTIC_COT="true"; shift 1 ;;
    --image-size) IMAGE_SIZE="${2:-}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:-}"; shift 2 ;;
    --nproc|--nproc-per-node) NPROC="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PROMPTS_DIR" ]]; then
  echo "ERROR: --prompts-dir is required" >&2
  usage
  exit 2
fi

# Resolve repo root and script path
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JANUS_ROOT="$SELF_DIR"
SCRIPT_PATH="$JANUS_ROOT/evaluation/reason_inference_geneval.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: Script not found: $SCRIPT_PATH" >&2
  exit 1
fi

# Optional conda activation
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}"
  fi
fi

if [[ ! -f "$PROMPTS_DIR" ]]; then
  echo "ERROR: prompts jsonl not found: $PROMPTS_DIR" >&2
  exit 1
fi

if [[ -n "$MODEL_PATH" && ! -e "$MODEL_PATH" ]]; then
  echo "ERROR: --model-path does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$REASON_PROMPT_PATH" ]]; then
  echo "ERROR: --reason-prompt is required (reason_inference_geneval.py expects it)" >&2
  exit 2
fi
if [[ ! -f "$REASON_PROMPT_PATH" ]]; then
  echo "ERROR: --reason-prompt not found: $REASON_PROMPT_PATH" >&2
  exit 1
fi

if [[ -z "$SAVE_ROOT" ]]; then
  # mimic the original naming style
  if [[ -n "$MODEL_PATH" ]]; then
    MODEL_DIR_NAME="$(basename -- "$MODEL_PATH")"
  else
    MODEL_DIR_NAME="no-model-path"
  fi
  SAVE_ROOT="$JANUS_ROOT/outputs/geneval-${MODEL_DIR_NAME}-$(date +%Y%m%d-%H%M%S)"
fi

mkdir -p "$(dirname -- "$SAVE_ROOT")" "$SAVE_ROOT" "$JANUS_ROOT/logs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export NCCL_DEBUG="${NCCL_DEBUG:-warn}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

cd "$JANUS_ROOT"

PY_ARGS=(
  --prompts_dir "$PROMPTS_DIR"
  --base_model_path "$BASE_MODEL_PATH"
  --reasoning_prompt_path "$REASON_PROMPT_PATH"
  --image-size "$IMAGE_SIZE"
  --temperature "$TEMPERATURE"
  --save_root "$SAVE_ROOT"
)

if [[ -n "$MODEL_PATH" ]]; then
  PY_ARGS+=(--model_path "$MODEL_PATH")
fi
if [[ "$SEMANTIC_COT" == "true" ]]; then
  PY_ARGS+=(--semantic_cot True)
fi

# IMPORTANT: python overrides MASTER_ADDR/PORT internally; keep torchrun consistent.
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29500"

set -x
torchrun \
  --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  evaluation/reason_inference_geneval.py \
  "${PY_ARGS[@]}" \
  "$@"

