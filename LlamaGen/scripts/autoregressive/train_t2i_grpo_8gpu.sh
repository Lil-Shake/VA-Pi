#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:" >&2
  echo "  bash scripts/autoregressive/train_t2i_grpo_8gpu.sh <VQ_CKPT> <GPT_INIT_CKPT> <TAR_DATA_SPEC> [T5_FEAT_PATH] [SHORT_T5_FEAT_PATH] [extra python args...]" >&2
  echo "" >&2
  echo "Notes:" >&2
  echo "  - train_t2i_grpo.py requires --t5-feat-path; for --dataset t2i_tar it is ignored, so you can omit it (defaults to /dev/null)." >&2
  echo "  - If you pass extra python args, keep them AFTER optional T5 paths (or omit T5 paths entirely)." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/t2i

VQ_CKPT="$1"
GPT_INIT_CKPT="$2"
TAR_DATA_SPEC="$3"
shift 3

# Optional positional paths:
# - T5_FEAT_PATH (required by CLI, ignored for t2i_tar; default /dev/null)
# - SHORT_T5_FEAT_PATH (optional; pass "-" to disable)
T5_FEAT_PATH="/dev/null"
SHORT_T5_FEAT_PATH=""
if [[ $# -ge 1 && "$1" != --* ]]; then
  T5_FEAT_PATH="$1"
  shift 1
fi
if [[ $# -ge 1 && "$1" != --* ]]; then
  SHORT_T5_FEAT_PATH="$1"
  shift 1
fi
if [[ "${SHORT_T5_FEAT_PATH}" == "-" ]]; then
  SHORT_T5_FEAT_PATH=""
fi

if [[ ! -f "${VQ_CKPT}" ]]; then
  echo "ERROR: VQ_CKPT not found: ${VQ_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${GPT_INIT_CKPT}" ]]; then
  echo "ERROR: GPT_INIT_CKPT not found: ${GPT_INIT_CKPT}" >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12348}"

RUN_NAME="${RUN_NAME:-t2i-grpo}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/t2i/${RUN_NAME}-${TS}.log"

CKPT_DIR="${CKPT_DIR:-checkpoints}"

SHORT_T5_ARGS=()
if [[ -n "${SHORT_T5_FEAT_PATH}" ]]; then
  SHORT_T5_ARGS+=(--short-t5-feat-path "${SHORT_T5_FEAT_PATH}")
fi

set -x

torchrun \
  --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  autoregressive/train/train_t2i_grpo.py \
  --dataset t2i_tar \
  --vq-ckpt "${VQ_CKPT}" \
  --data-path "${TAR_DATA_SPEC}" \
  --t5-feat-path "${T5_FEAT_PATH}" \
  "${SHORT_T5_ARGS[@]}" \
  --image-size 256 --downsample-size 16 \
  --gpt-ckpt "${GPT_INIT_CKPT}" --from-fsdp \
  --gpt-model GPT-XL \
  --sample-batch-size 16 \
  --train-batch-size 8 \
  --gradient-accumulation-steps 16 \
  --num-generations 8 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --epochs 200 \
  --ckpt-every 100 \
  --ckpt-dir "${CKPT_DIR}" \
  --lr 1e-6 --weight-decay 1e-4 --beta1 0.9 --beta2 0.999 \
  --max-grad-norm 1.0 --clip-range 0.2 --adv-clip-max 5.0 \
  --mixed-precision bf16 --data-parallel fsdp \
  --reward-rec-weight 1.0 \
  --reward-perceptual-weight 1.0 \
  --aux-ce-weight 0.1 \
  --data-start 0 --data-end 73 \
  --use-wandb \
  --kl-coef 0.01 \
  --sample-model-mode train \
  --use-token-noise \
  --token-noise-prob 0.5 \
  "$@" 2>&1 | tee "${LOG_FILE}"


