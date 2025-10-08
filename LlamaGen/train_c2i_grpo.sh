#!/bin/bash
#SBATCH -p ai_arc               # 分区名
#SBATCH --job-name=dancegrpo-sd-hps   # 作业名称
#SBATCH --nodes=1               # 节点数
#SBATCH --gres=gpu:8            # 需要的 GPU 数
#SBATCH -c 64                   # CPU 核心数
#SBATCH --mem=0                 # 使用整节点内存，或者改成具体值 e.g. 256G
#SBATCH --time=48:00:00         # 最长运行时间 (48 小时)
#SBATCH --ntasks-per-node=1     # 每个节点的任务数
#SBATCH --output=logs/%x-%j.out # 输出日志路径 (会自动用 jobname-jobid 命名)

# -------------------------
# 环境变量
# -------------------------
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
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
# 模型与数据路径（按需修改）
# -------------------------
VQ_MODEL_PATH="/mnt/petrelfs/share_data/quxiaoye/models/LlamaGen/vq_ds16_c2i.pt"
# 可选：GPT 预训练初始化（留空则不加载）
GPT_INIT_CKPT="/mnt/petrelfs/share_data/quxiaoye/models/LlamaGen/c2i_XXL_384.pt"
DATA_PATH="/mnt/petrelfs/share_data/quxiaoye/imagenet1k-ds"
CKP_PATH=""

# -------------------------
# Python import 路径（确保绝对 import 生效）
# -------------------------
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
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
srun torchrun \
    --nnodes=1 --nproc_per_node=8 --node_rank=0 \
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
    --use-wandb --wandb-project autoreg-grpo --wandb-entity ${WANDB_ENTITY:-none} --wandb-run-name "c2i-grpo" \
    "$@"