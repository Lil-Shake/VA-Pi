#!/usr/bin/env bash
set -euo pipefail

# Pure bash launcher (NO Slurm). Extract T5 features for LAION-COCO style jsonl shards.
#
# Usage:
#   cd /path/to/VA-Pi/LlamaGen
#   bash scripts/language/extract_flan_t5_feat_laion_coco_stage1.sh \
#     <DATA_PATH> <T5_MODEL_PATH> <FEAT_OUT> [DATA_START] [DATA_END] [CAPTION_KEY] [extra args...]
#
# Example:
#   bash scripts/language/extract_flan_t5_feat_laion_coco_stage1.sh \
#     /abs/laion-coco-50m /abs/models /abs/models/flan-t5-xl/t5_features 0 73 blip

if [[ $# -lt 3 ]]; then
  echo "Usage:" >&2
  echo "  bash scripts/language/extract_flan_t5_feat_laion_coco_stage1.sh <DATA_PATH> <T5_MODEL_PATH> <FEAT_OUT> [DATA_START] [DATA_END] [CAPTION_KEY] [extra args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs/t2i

DATA_PATH="$1"
T5_MODEL_PATH="$2"
FEAT_OUT="$3"
shift 3

DATA_START="${1:-0}"
DATA_END="${2:-73}"
CAPTION_KEY="${3:-blip}"
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi
if [[ $# -ge 1 && "$1" != --* ]]; then shift 1; fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -d "${T5_MODEL_PATH}" ]]; then
  echo "ERROR: T5_MODEL_PATH not found: ${T5_MODEL_PATH}" >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12337}"
RUN_NAME="${RUN_NAME:-t2i-t5-feat}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/t2i/${RUN_NAME}-${TS}.log"

set -x

torchrun \
  --nnodes=1 --nproc_per_node=8 --node_rank=0 \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  language/extract_t5_feature.py \
  --data-start "${DATA_START}" --data-end "${DATA_END}" \
  --data-path "${DATA_PATH}" \
  --t5-model-path "${T5_MODEL_PATH}" \
  --t5-model-type flan-t5-xl \
  --t5-path "${FEAT_OUT}" \
  --caption-key "${CAPTION_KEY}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
