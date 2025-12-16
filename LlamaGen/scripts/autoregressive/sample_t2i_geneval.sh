#!/usr/bin/env bash
set -euo pipefail
set -x

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/autoregressive/sample_t2i_geneval.sh \
    <VQ_CKPT_OR_DIR> <GPT_CKPT_OR_DIR> <T5_PATH_DIR> <PROMPTS_JSONL> <OUT_ROOT> \
    [-- extra args for autoregressive/sample/sample_t2i_ddp_geneval.py]

Common env overrides:
  NPROC_PER_NODE=8            (default: 8)
  MASTER_PORT=12345           (default: 12345)
  GPT_MODEL=GPT-XL            (default: GPT-XL)
  VQ_MODEL=VQ-16              (default: VQ-16)
  IMAGE_SIZE=256              (default: 256)
  DOWN_SIZE=16                (default: 16)
  CLS_TOKENS=120              (default: 120)
  PER_PROC_BS=4               (default: 4)
  PROMPT_BS=16                (default: 16)
  NUM_SAMPLES=4               (default: 4; images per prompt)
  TOPK=2000 TOPP=1.0 CFG_SCALE=7.5 TEMP=1.0 PRECISION=bf16 SEED=0
  COMPILE=0                   (default: 0; set 1 to enable torch.compile)
EOF
}

if [[ $# -lt 5 ]]; then
  usage
  exit 2
fi

VQ_CKPT="$1"
GPT_CKPT="$2"
T5_PATH="$3"
PROMPTS_JSONL="$4"
OUT_ROOT="$5"
shift 5

if [[ ! -e "$VQ_CKPT" ]]; then
  echo "ERROR: VQ_CKPT not found (file or dir): $VQ_CKPT" >&2
  exit 1
fi
if [[ ! -e "$GPT_CKPT" ]]; then
  echo "ERROR: GPT_CKPT not found (file or dir): $GPT_CKPT" >&2
  exit 1
fi
if [[ ! -d "$T5_PATH" ]]; then
  echo "ERROR: T5_PATH not found (expect a directory): $T5_PATH" >&2
  exit 1
fi
if [[ ! -f "$PROMPTS_JSONL" ]]; then
  echo "ERROR: prompts jsonl not found: $PROMPTS_JSONL" >&2
  exit 1
fi

# Always run from the LlamaGen project root so relative imports/paths work.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p "$OUT_ROOT"

# --- defaults (override via env) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"

GPT_MODEL="${GPT_MODEL:-GPT-XL}"
VQ_MODEL="${VQ_MODEL:-VQ-16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
DOWN_SIZE="${DOWN_SIZE:-16}"
CLS_TOKENS="${CLS_TOKENS:-120}"

PER_PROC_BS="${PER_PROC_BS:-4}"
PROMPT_BS="${PROMPT_BS:-16}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"

TOPK="${TOPK:-2000}"
TOPP="${TOPP:-1.0}"
CFG_SCALE="${CFG_SCALE:-7.5}"
TEMP="${TEMP:-1.0}"

PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
COMPILE="${COMPILE:-0}"

EXTRA_FLAGS=()
if [[ "$COMPILE" == "1" || "$COMPILE" == "true" ]]; then
  EXTRA_FLAGS+=(--compile)
fi

DEFAULT_ARGS=(
  --jsonl "$PROMPTS_JSONL"
  --num-samples "$NUM_SAMPLES"
  --out-root "$OUT_ROOT"
  --t5-path "$T5_PATH"
  --t5-model-type flan-t5-xl
  --t5-feature-max-len "$CLS_TOKENS"
  --gpt-model "$GPT_MODEL" --gpt-ckpt "$GPT_CKPT"
  --vq-model "$VQ_MODEL" --vq-ckpt "$VQ_CKPT"
  --image-size "$IMAGE_SIZE" --downsample-size "$DOWN_SIZE"
  --cls-token-num "$CLS_TOKENS"
  --prompt-batch-size "$PROMPT_BS"
  --per-proc-batch-size "$PER_PROC_BS"
  --precision "$PRECISION"
  --cfg-scale "$CFG_SCALE" --top-k "$TOPK" --top-p "$TOPP" --temperature "$TEMP"
  --global-seed "$SEED"
)

exec torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  autoregressive/sample/sample_t2i_ddp_geneval.py \
  "${EXTRA_FLAGS[@]}" \
  "${DEFAULT_ARGS[@]}" \
  "$@"
