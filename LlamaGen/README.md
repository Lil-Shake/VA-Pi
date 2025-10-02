## LlamaGen Class-to-Image GRPO Trainer

This repository contains a training script for class-conditioned autoregressive image generation using a VQ tokenizer and a GPT backbone, optimized with a GRPO-style objective over multiple sampled generations.

The main entry point is `LlamaGen/autoregressive/train/train_c2i_grpo.py`.

### What it does
- **Teacher-forced sampling**: Computes logits for ground-truth token sequences conditioned on class IDs.
- **Multiple generations per sample**: Samples `--num-generations` sequences per input and decodes them with a frozen VQ model.
- **Reward and advantage**: Uses negative MSE between decoded and ground-truth images as reward; standardizes per-group advantages across the generations of each input.
- **GRPO-style update**: Optimizes a clipped objective using the ratio of new to old token log-probabilities.
- **Distributed + FSDP**: Supports multi-GPU training with `torch.distributed` (NCCL) and FSDP sharding.
- **Logging and checkpoints**: Optional Weights & Biases logging (offline mode) and periodic checkpointing.

### Requirements
- CUDA-capable GPUs and NCCL backend
- Python, PyTorch, torchvision, Hugging Face `datasets`, optional `wandb`
- A trained VQ model checkpoint for encoding/decoding image code sequences

### Dataset format
Provide `--data-path` pointing to a Hugging Face dataset saved with `datasets.save_to_disk`. Expected structure:
- A dataset dict with a `train` split
- Columns: `image` and `label`
  - If `image` is missing, the script falls back to path-like columns: `image_path`, `file`, or `filepath`
  - If `label` is missing, it will try `labels`

Images are center-cropped to `--image-size` and normalized to `[-1, 1]` in `CHW` format.

### Quick start
Run on 1 GPU:

```bash
torchrun --nproc_per_node=1 lab/LlamaGen/autoregressive/train/train_c2i_grpo.py \
  --vq-ckpt /path/to/vq_checkpoint.pth \
  --data-path /path/to/hf_dataset_root \
  --sample-batch-size 8 \
  --train-batch-size 4 \
  --num-generations 8 \
  --ckpt-every 1000 \
  --ckpt-dir /path/to/checkpoints
```

Run on 8 GPUs (recommended):

```bash
torchrun --nproc_per_node=8 lab/LlamaGen/autoregressive/train/train_c2i_grpo.py \
  --vq-ckpt /path/to/vq_checkpoint.pth \
  --data-path /path/to/hf_dataset_root \
  --sample-batch-size 32 \
  --train-batch-size 8 \
  --num-generations 8 \
  --mixed-precision bf16 \
  --data-parallel fsdp \
  --ckpt-every 1000 \
  --ckpt-dir /path/to/checkpoints \
  --use-wandb
```

Notes:
- The script asserts GPU availability and initializes `torch.distributed` with `nccl`; use `torchrun` even for single-GPU runs.
- Weights & Biases runs in offline mode by default; sync later with `wandb sync` if desired.

### Argument reference (key options)

- **Model**
  - `--gpt-model` (str, default: `GPT-B`): GPT backbone name from `GPT_models`.
  - `--gpt-ckpt` (str, optional): Path to a GPT checkpoint to warm-start.
  - `--from-fsdp` (flag): Interpret `--gpt-ckpt` as FSDP-format state dict.
  - `--vq-model` (str, default: `VQ-16`): VQ model name from `VQ_models`.
  - `--vq-ckpt` (str, required): Path to VQ checkpoint.
  - `--codebook-size` (int, default: 16384): VQ vocabulary size.
  - `--codebook-embed-dim` (int, default: 8): VQ code embedding dimension.
  - `--image-size` (int, default: 256; choices: 256, 384, 512): Output image size.
  - `--downsample-size` (int, default: 16; choices: 8,16): VQ downsample factor; sets latent grid size.
  - `--num-classes` (int, default: 1000): Number of class IDs.
  - `--cls-token-num` (int, default: 1): Number of class-conditioning tokens.

- **Precision & parallelism**
  - `--mixed-precision` (str, default: `bf16`; choices: fp32, tf32, fp16, bf16): Parameter and activation dtype.
  - `--grad-precision` (str, optional; choices: fp32, fp16, bf16): All-reduce dtype; defaults to `mixed-precision` if not set.
  - `--data-parallel` (str, default: `fsdp`; choices: sdp, fsdp, hsdp): Sharding strategy.

- **Optimization**
  - `--lr` (float, default: 1e-5): Learning rate.
  - `--weight-decay` (float, default: 1e-2): Weight decay.
  - `--beta1` (float, default: 0.9), `--beta2` (float, default: 0.95): AdamW betas.
  - `--max-grad-norm` (float, default: 1.0): Gradient clipping norm.

- **Training schedule**
  - `--epochs` (int, default: 100): Upper bound on loader iterations; loop stops when index > epochs.
  - `--global-seed` (int, default: 0): Global seed used for `DistributedSampler`.

- **GRPO sampling**
  - `--num-generations` (int, default: 8): Samples per input used to compute group advantages.
  - `--top-k` (int, default: 0), `--top-p` (float, default: 1.0), `--temperature` (float, default: 1.0): Sampling controls over TF logits.

- **Batches & loader**
  - `--sample-batch-size` (int, default: 32): Batch size for sampling and reward computation.
  - `--train-batch-size` (int, default: 8): Batch size for optimization steps.
  - `--gradient-accumulation-steps` (int, default: 1): Accumulate steps before optimizer step.
  - `--num-workers` (int, default: 8): DataLoader workers.

- **Loss clipping**
  - `--adv-clip-max` (float, default: 5.0): Clamp absolute advantages.
  - `--clip-range` (float, default: 1e-4): PPO-style ratio clip range.

- **Data**
  - `--dataset` (str, default: imagenet): Dataset name tag (informational).
  - `--data-path` (str, required): Path to `datasets.save_to_disk` directory.

- **Logging**
  - `--use-wandb` (flag): Enable Weights & Biases logging (offline mode).
  - `--wandb-project` (str, default: `autoreg-grpo`), `--wandb-entity` (str, optional), `--wandb-run-name` (str, optional)
  - `--log-image-every` (int, default: 100), `--log-interval` (int, default: 10): Image logging is currently commented out in the script.

- **Checkpointing**
  - `--ckpt-every` (int, default: 0): Save every N optimizer steps (0 disables).
  - `--ckpt-dir` (str, default: `checkpoints`): Output directory.
  - `--ckpt-sharded` (flag): Save FSDP sharded state per rank. By default (flag not set), the script saves a consolidated full state (rank 0 only).

### Outputs
- Checkpoints saved under `--ckpt-dir/<step>/` including:
  - `model.shard.*.pth` (if `--ckpt-sharded`) or `consolidated.pth` (if not sharded)
  - `optimizer.*.pth` per rank
  - `state_type.txt` indicating `sharded` or `full`
  - `resume_step.txt` (rank 0) with the global optimizer step

### Tips
- If you see NCCL-related hangs, verify `torchrun` launch, environment, and GPU visibility.
- Adjust `--sample-batch-size`, `--train-batch-size`, `--num-generations`, and `--gradient-accumulation-steps` to stay within memory limits.
- Use `--mixed-precision bf16` (default) on Ampere+ GPUs for speed and stability.
- To reduce checkpoint memory/IO, enable `--ckpt-sharded`.

### File
- Trainer: `LlamaGen/autoregressive/train/train_c2i_grpo.py`


