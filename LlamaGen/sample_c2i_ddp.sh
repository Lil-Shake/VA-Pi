#!/bin/bash
#SBATCH -p ai_arc                              # 分区名
#SBATCH --job-name=unitok-sample-c2i           # 作业名称
#SBATCH --nodes=1                              # 节点数
#SBATCH --gres=gpu:8                            # GPU 数
#SBATCH -c 64                                   # CPU 核心数
#SBATCH --mem=0                                 # 使用整节点内存
#SBATCH --time=48:00:00                         # 最长运行时间
#SBATCH --ntasks-per-node=1                     # 每个节点的任务数
#SBATCH --output=logs/%x-%j.out                 # 日志输出路径

# -------------------------
# 环境变量
# -------------------------
export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export OMP_NUM_THREADS=8

# -------------------------
# 设置工作目录（项目根目录）
# -------------------------
PROJECT_ROOT="/mnt/petrelfs/huangsiyuan2/lab/LlamaGen"
cd $PROJECT_ROOT
mkdir -p logs

# -------------------------
# 打印 GPU 信息
# -------------------------
nvidia-smi

# -------------------------
# 激活 conda 环境
# -------------------------
source /mnt/petrelfs/huangsiyuan2/anaconda3/etc/profile.d/conda.sh
conda activate unitok

# -------------------------
# Bash 调试模式
# -------------------------
set -x

# -------------------------
# 模型路径
# -------------------------
# VQ_MODEL_PATH="/mnt/petrelfs/share_data/quxiaoye/models/UniTok/unitok_tokenizer.pth"
VQ_MODEL_PATH="/mnt/petrelfs/share_data/quxiaoye/models/LlamaGen/vq_ds16_c2i.pt"
GPT_MODEL_PATH="/mnt/petrelfs/share_data/quxiaoye/models/LlamaGen/c2i_XXL_384.pt"
OUTPUT_DIR="/mnt/petrelfs/share_data/quxiaoye/imagenet-eval"

# -------------------------
# Python import 路径（确保绝对 import 生效）
# -------------------------
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export TORCH_USE_CUDA_DSA=1

# -------------------------
# 运行脚本
# -------------------------
srun torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12345 \
    autoregressive/sample/sample_c2i_ddp.py \
    --vq-ckpt $VQ_MODEL_PATH \
    --gpt-ckpt $GPT_MODEL_PATH \
    --gpt-model GPT-XXL \
    --sample-dir $OUTPUT_DIR \
    --from-fsdp \
    --image-size 384 --image-size-eval 256 --cfg-scale 1.75 --temperature 1.0 \
    --per-proc-batch-size 64 \
    "$@"

    # --temperature 0.9 \
    # --num-codebooks 8 --codebook-size 32768 \
    # --num-fid-samples 100 --per-proc-batch-size 1 \
    #--num-output-layer 4 \