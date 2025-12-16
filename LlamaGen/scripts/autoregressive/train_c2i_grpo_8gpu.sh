#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage:" >&2
  echo "  bash scripts/autoregressive/train_c2i_grpo_8gpu.sh <VQ_MODEL_PATH> <GPT_INIT_CKPT> <DATA_PATH> [extra python args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

VQ_MODEL_PATH="$1"
GPT_INIT_CKPT="$2"
DATA_PATH="$3"
shift 3

if [[ ! -f "${VQ_MODEL_PATH}" ]]; then
  echo "ERROR: VQ_MODEL_PATH not found: ${VQ_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${GPT_INIT_CKPT}" ]]; then
  echo "ERROR: GPT_INIT_CKPT not found: ${GPT_INIT_CKPT}" >&2
  exit 1
fi


MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12347}"

RUN_NAME="${RUN_NAME:-c2i-grpo}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${RUN_NAME}-${TS}.log"

set -x

torchrun \
  --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  autoregressive/train/train_c2i_grpo.py \
  --vq-ckpt "${VQ_MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --gpt-ckpt "${GPT_INIT_CKPT}" --from-fsdp \
  --image-size 384 --downsample-size 16 \
  --gpt-model GPT-XXL \
  --sample-batch-size 16 \
  --train-batch-size 8 \
  --gradient-accumulation-steps 16 \
  --num-generations 8 \
  --temperature 1.0 --top-k 0 --top-p 1.0 \
  --epochs 100 \
  --ckpt-every 100 \
  --lr 1e-6 --weight-decay 1e-4 --beta1 0.9 --beta2 0.999 \
  --max-grad-norm 1.0 --clip-range 0.2 --adv-clip-max 5.0 \
  --mixed-precision bf16 --data-parallel fsdp \
  --reward-rec-weight 1.0 \
  --reward-perceptual-weight 1.0 \
  --aux-ce-weight 0.1 \
  --sample-model-mode train \
  --use-wandb --wandb-project autoreg-grpo --wandb-entity "${WANDB_ENTITY:-none}" --wandb-run-name "${RUN_NAME}" \
  "$@" 2>&1 | tee "${LOG_FILE}"


