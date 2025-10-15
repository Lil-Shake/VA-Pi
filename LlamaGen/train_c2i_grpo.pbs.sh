#!/bin/bash
#PBS -P CFP03-CF-048
#PBS -j oe
#PBS -k oed
#PBS -N train_c2i_grpo
#PBS -q auto
#PBS -l select=1:ngpus=8
#PBS -l walltime=48:00:00

# -------------------------
# 环境变量
# -------------------------
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export HF_HOME=/scratch/t0937090/cache
export XDG_CACHE_HOME=/scratch/t0937090/cache

# -------------------------
# 设置工作目录（项目根目录）
# -------------------------
PROJECT_ROOT="/home/svu/t0937090/lab/AR-RL/LlamaGen"
cd $PROJECT_ROOT
mkdir -p logs

# 将作业的 stdout/stderr 重定向到日志路径
exec > logs/${PBS_JOBNAME:-job}-${PBS_JOBID}.out 2>&1

# -------------------------
# 打印 GPU 信息
# -------------------------
nvidia-smi

# -------------------------
# 容器与环境
# -------------------------
module load singularity
image="/app1/common/singularity-img/hopper/cuda/cuda_12.1.0-cudnn8-devel-u20.04.sif"
singularity exec -e $image bash << 'EOF'

source /hpctmp/t0937090/virtualenvs/unitok/bin/activate

# -------------------------
# Bash 调试模式
# -------------------------
set -x

# -------------------------
# 模型与数据路径（按需修改）
# -------------------------

VQ_MODEL_PATH="/scratch/t0937090/model/LlamaGen/vq_ds16_c2i.pt"
GPT_INIT_CKPT="/scratch/t0937090/model/LlamaGen/c2i_XXL_384.pt"
DATA_PATH="/scratch/t0937090/dataset/imagenet-1k/train"
CKP_PATH="/scratch/t0937090/checkpoint/LlamaGen/c2i_grpo"

# -------------------------
# Python import 路径（确保绝对 import 生效）
# -------------------------
export PYTHONPATH=$PROJECT_ROOT:
export TORCH_USE_CUDA_DSA=1

# -------------------------
# 组装可选 GPT 初始化参数
# -------------------------
EXTRA_GPT_FLAGS=""
if [ -n "$GPT_INIT_CKPT" ]; then
    EXTRA_GPT_FLAGS="--gpt-ckpt $GPT_INIT_CKPT --from-fsdp"
fi

# -------------------------
# 运行训练脚本（8 卡）
# -------------------------
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \
    --master_port=12347 \
    autoregressive/train/train_c2i_grpo.py \
    --vq-ckpt $VQ_MODEL_PATH \
    --data-path $DATA_PATH \
    --image-size 384 --downsample-size 16 \
    --gpt-model GPT-XXL $EXTRA_GPT_FLAGS \
    --sample-batch-size 16 \
    --train-batch-size 8 \
    --gradient-accumulation-steps 16 \
    --num-generations 8 \
    --temperature 1.0 --top-k 0 --top-p 1.0 \
    --epochs 60 \
    --ckpt-every 60 \
    --lr 1e-5 --weight-decay 1e-4 --beta1 0.9 --beta2 0.999 \
    --max-grad-norm 1.0 --clip-range 0.2 --adv-clip-max 5.0 \
    --mixed-precision bf16 --data-parallel fsdp \
    --reward-rec-weight 1.0 \
    --reward-perceptual-weight 1.0 \
    --aux-ce-weight 0.07 \
    --train-mode train \
    --use-wandb --wandb-project autoreg-grpo --wandb-entity ${WANDB_ENTITY:-none} --wandb-run-name "c2i-grpo" \
    "$@"

EOF