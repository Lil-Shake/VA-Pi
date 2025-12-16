#!/usr/bin/env bash
set -euo pipefail

# Pure bash launcher (NO Slurm). Single node (8 GPUs) DeepSpeed.
# Dataset is streamed/downloaded from HuggingFace and optionally saved to disk shards.
#
# Usage:
#   cd /path/to/VA-Pi/Janus
#   bash train/run_t2i_grpo_janus_deepspeed_16g_hf.sh \
#     <MODEL_PATH_OR_ID> <HF_STREAM_DATASET> <HF_STREAM_OUTPUT_DIR> \
#     [HF_STREAM_SPLIT] [HF_STREAM_N_SAMPLES] [HF_STREAM_CHUNK_SIZE] [HF_IMAGE_KEY] [HF_CAPTION_KEY] \
#     [extra python args...]
#
# Example (FLUX-Reason-6M):
#   bash train/run_t2i_grpo_janus_deepspeed_16g_hf.sh \
#     deepseek-ai/Janus-Pro-1B LucasFang/FLUX-Reason-6M ./hf_stream/FLUX-Reason-6M-random \
#     train 50000 2000 image caption_original

if [[ $# -lt 3 ]]; then
  echo "Usage:" >&2
  echo "  bash train/run_t2i_grpo_janus_deepspeed_16g_hf.sh <MODEL_PATH_OR_ID> <HF_STREAM_DATASET> <HF_STREAM_OUTPUT_DIR> [split] [n_samples] [chunk_size] [image_key] [caption_key] [extra python args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

MODEL_PATH="$1"
HF_STREAM_DATASET="$2"
HF_STREAM_OUTPUT_DIR="$3"
shift 3

HF_STREAM_SPLIT="${1:-train}"
HF_STREAM_N_SAMPLES="${2:-50000}"
HF_STREAM_CHUNK_SIZE="${3:-2000}"
HF_IMAGE_KEY="${4:-image}"
HF_CAPTION_KEY="${5:-caption_original}"

if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi

# -----------------------------
# Fixed knobs (match the provided Slurm script requirements)
# -----------------------------
EPOCHS=100
SAMPLE_BS=8
TRAIN_BS=2
GA_STEPS=32
MIXED_PRECISION="bf16"   # none|fp16|bf16
IMAGE_SIZE=384
PATCH_SIZE=16
NUM_GENERATIONS=8
TEMPERATURE=1.0
TOP_K=0
TOP_P=1.0
CFG_SCALE=1.0
DECODE_CHUNK=8
ADV_CLIP_MAX=5.0
CLIP_RANGE=0.2
SAMPLE_MODEL_MODE="train"  # train|eval|twice
REWARD_REC_TYPE="l2"       # l1|l2
REWARD_REC_WEIGHT=1.0
REWARD_PERCEPTUAL_WEIGHT=1.0
AUX_CE_WEIGHT=0.1
USE_KL_LOSS=false
KL_COEF=0.01
KL_BASE_PATH=""
USE_TOKEN_NOISE=true
TOKEN_NOISE_PROB=1.0
LR=1e-7
BETA1=0.9
BETA2=0.999
WEIGHT_DECAY=1e-4

CKPT_EVERY=100
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/checkpoints/$(date +%Y%m%d-%H%M%S)}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/train/configs/zero3.json}"

if [[ -z "${HF_STREAM_DATASET}" ]]; then
  echo "[FATAL] HF_STREAM_DATASET is empty." >&2
  exit 1
fi
if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "[FATAL] DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi
mkdir -p "${CKPT_DIR}"
mkdir -p "${HF_STREAM_OUTPUT_DIR}"

# -----------------------------
# Environment (match the provided script)
# -----------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_DIST_TIMEOUT="${TORCH_DIST_TIMEOUT:-36000}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker,virbr,vmnet,vboxnet}"
export CUDA_LAUNCH_BLOCKING=0

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-10}"

NNODES=1
GPUS_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="${MASTER_PORT:-29500}"

RUN_NAME="${RUN_NAME:-train-t2i-grpo-janus-deepspeed-16g-hf}"
TS="$(date +%Y%m%d_%H%M%S)"
HOST_TAG="$(hostname | tr '/:' '__')"
LOG_FILE="logs/${RUN_NAME}-${TS}-${HOST_TAG}.log"

SCRIPT_PATH="${PROJECT_ROOT}/train/train_t2i_grpo_janus.py"

PY_ARGS=(
  --model-path "${MODEL_PATH}"
  --image-size "${IMAGE_SIZE}"
  --patch-size "${PATCH_SIZE}"
  --epochs "${EPOCHS}"
  --sample-batch-size "${SAMPLE_BS}"
  --train-batch-size "${TRAIN_BS}"
  --gradient-accumulation-steps "${GA_STEPS}"
  --mixed-precision "${MIXED_PRECISION}"
  --num-workers 0
  --num-generations "${NUM_GENERATIONS}"
  --temperature "${TEMPERATURE}"
  --top-k "${TOP_K}"
  --top-p "${TOP_P}"
  --cfg-scale "${CFG_SCALE}"
  --decode-chunk "${DECODE_CHUNK}"
  --adv-clip-max "${ADV_CLIP_MAX}"
  --clip-range "${CLIP_RANGE}"
  --sample-model-mode "${SAMPLE_MODEL_MODE}"
  --reward-rec-type "${REWARD_REC_TYPE}"
  --reward-rec-weight "${REWARD_REC_WEIGHT}"
  --reward-perceptual-weight "${REWARD_PERCEPTUAL_WEIGHT}"
  --aux-ce-weight "${AUX_CE_WEIGHT}"
  --ckpt-every "${CKPT_EVERY}"
  --ckpt-dir "${CKPT_DIR}"
  --use-deepspeed
  --deepspeed-config "${DEEPSPEED_CONFIG}"
  --use-wandb
  --lr "${LR}"
  --beta1 "${BETA1}"
  --beta2 "${BETA2}"
  --weight-decay "${WEIGHT_DECAY}"
  # HF streaming args
  --hf-stream-dataset "${HF_STREAM_DATASET}"
  --hf-stream-split "${HF_STREAM_SPLIT}"
  --hf-stream-n-samples "${HF_STREAM_N_SAMPLES}"
  --hf-stream-chunk-size "${HF_STREAM_CHUNK_SIZE}"
  --hf-stream-output-dir "${HF_STREAM_OUTPUT_DIR}"
  --hf-image-key "${HF_IMAGE_KEY}"
  --hf-caption-key "${HF_CAPTION_KEY}"
  # Optional: also allow training to resume from saved shards
  --hf-from-disk-dir "${HF_STREAM_OUTPUT_DIR}"
)

if [[ "${USE_KL_LOSS}" == "true" ]]; then
  PY_ARGS+=( --use-kl-loss --kl-coef "${KL_COEF}" )
  if [[ -n "${KL_BASE_PATH}" ]]; then
    PY_ARGS+=( --kl-base-path "${KL_BASE_PATH}" )
  fi
fi

if [[ "${USE_TOKEN_NOISE}" == "true" ]]; then
  PY_ARGS+=( --use-token-noise --token-noise-prob "${TOKEN_NOISE_PROB}" )
fi

set -x

deepspeed \
  --num_nodes "${NNODES}" \
  --num_gpus "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${SCRIPT_PATH}" \
  "${PY_ARGS[@]}" \
  "$@" 2>&1 | tee "${LOG_FILE}"


