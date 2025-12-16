#!/usr/bin/env bash
set -euo pipefail
set -x

# Always run from the LlamaGen project root so relative paths work.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# -------- usage --------
usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/autoregressive/sample_c2i.sh <VQ_CKPT> <GPT_CKPT> [SAMPLE_DIR] [-- extra args for sample_c2i_ddp.py]

Examples:
  bash scripts/autoregressive/sample_c2i.sh ./pretrained_models/vq_ds16_c2i.pt ./pretrained_models/c2i_XXL_384.pt
  bash scripts/autoregressive/sample_c2i.sh /abs/vq.pt /abs/gpt.pt /abs/out_dir --from-fsdp --gpt-model GPT-XXL --image-size 384
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

# required inputs
VQ_CKPT="$1"
GPT_CKPT="$2"
shift 2

# optional positional sample dir (only if the next arg is not a flag)
SAMPLE_DIR="${SAMPLE_DIR:-samples}"
if [[ $# -ge 1 && "${1:-}" != -* ]]; then
  SAMPLE_DIR="$1"
  shift 1
fi

# -------- minimal defaults (override via env) --------
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"

GPT_MODEL="${GPT_MODEL:-GPT-XXL}"
FROM_FSDP="${FROM_FSDP:-1}" # set to 0 if your ckpt is not FSDP-style

# sampling defaults (override via env; CLI args at the end can also override)
IMAGE_SIZE="${IMAGE_SIZE:-384}"
IMAGE_SIZE_EVAL="${IMAGE_SIZE_EVAL:-256}"
CFG_SCALE="${CFG_SCALE:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
PER_PROC_BATCH_SIZE="${PER_PROC_BATCH_SIZE:-64}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"

if [[ ! -f "$VQ_CKPT" ]]; then
  echo "ERROR: VQ ckpt not found: $VQ_CKPT" >&2
  echo "Hint: download it and place it under ./pretrained_models/, or set VQ_CKPT=/abs/path/to/vq_ds16_c2i.pt" >&2
  exit 1
fi

if [[ ! -f "$GPT_CKPT" ]]; then
  echo "ERROR: GPT ckpt not found: $GPT_CKPT" >&2
  echo "Hint: download it and place it under ./pretrained_models/, or set GPT_CKPT=/abs/path/to/c2i_XXL_384.pt" >&2
  exit 1
fi

EXTRA_FLAGS=()
if [[ "$FROM_FSDP" == "1" || "$FROM_FSDP" == "true" ]]; then
  EXTRA_FLAGS+=(--from-fsdp)
fi

DEFAULT_ARGS=(
  --gpt-model "$GPT_MODEL"
  --image-size "$IMAGE_SIZE"
  --image-size-eval "$IMAGE_SIZE_EVAL"
  --cfg-scale "$CFG_SCALE"
  --temperature "$TEMPERATURE"
  --per-proc-batch-size "$PER_PROC_BATCH_SIZE"
  --num-fid-samples "$NUM_FID_SAMPLES"
)

exec torchrun \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_port="$MASTER_PORT" \
  autoregressive/sample/sample_c2i_ddp.py \
  --vq-ckpt "$VQ_CKPT" \
  --gpt-ckpt "$GPT_CKPT" \
  --sample-dir "$SAMPLE_DIR" \
  "${DEFAULT_ARGS[@]}" \
  "${EXTRA_FLAGS[@]}" \
  "$@"
