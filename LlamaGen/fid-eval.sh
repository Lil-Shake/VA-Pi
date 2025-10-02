#!/bin/bash
#SBATCH -p ai_arc                               # 分区名
#SBATCH --job-name=eval-cfg-1.75                # 作业名称
#SBATCH --nodes=1                               # 节点数
#SBATCH --gres=gpu:1                            # GPU 数
#SBATCH -c 8                                    # CPU 核心数
#SBATCH --mem=0                                 # 使用整节点内存
#SBATCH --time=48:00:00                         # 最长运行时间
#SBATCH --ntasks-per-node=1                     # 每个节点的任务数
#SBATCH --output=logs/%x-%j.out                 # 日志输出路径

export OMP_NUM_THREADS=8
REFER_DATA="/mnt/petrelfs/share_data/quxiaoye/imagenet-eval/VIRTUAL_imagenet256_labeled.npz"
EVAL_DATA="/mnt/petrelfs/share_data/quxiaoye/imagenet-eval/GPT-XXL-LlamaGen-sample-size-384-eval-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-1.75-seed-0.npz"

PROJECT_ROOT="/mnt/petrelfs/huangsiyuan2/lab/LlamaGen"
cd $PROJECT_ROOT

source /mnt/petrelfs/huangsiyuan2/anaconda3/etc/profile.d/conda.sh
conda activate unitok

srun python evaluations/c2i/evaluator.py $REFER_DATA $EVAL_DATA