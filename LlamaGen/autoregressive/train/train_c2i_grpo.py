import os
import time
import argparse
import contextlib
import functools

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from torchvision import transforms
from torchvision.utils import make_grid
import wandb  # type: ignore
import datasets


from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
from tqdm import tqdm

def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def create_optimizer(model: nn.Module, weight_decay: float, learning_rate: float, betas: tuple[float, float]):
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        

class HFImageDataset(torch.utils.data.Dataset):
    """HuggingFace arrow-based dataset loader for images and labels.
    Expects a directory saved via datasets.save_to_disk with data-*.arrow shards,
    dataset_info.json, state.json, and an 'image' column and a 'label' column.
    """
    def __init__(self, root: str, transform=None):
        self.ds = datasets.load_from_disk(root)
        self.transform = transform

    def __len__(self):
        # self.ds is a DatasetDict; use the train split length
        return len(self.ds['train'])

    def __getitem__(self, index: int):
        item = self.ds['train'][int(index)]
        image = item.get('image')
        if image is None:
            # fallback to path-based columns if present
            path = item.get('image_path') or item.get('file') or item.get('filepath')
            if path is None:
                raise KeyError("HF dataset item has no 'image' or path column")
            from PIL import Image
            image = Image.open(path).convert('RGB')
        label = item.get('label')
        if label is None:
            label = item.get('labels')
        if label is None:
            raise KeyError("HF dataset item has no 'label' or 'labels' column")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def tf_logits(model: nn.Module, cls_ids: torch.Tensor, gt_tokens: torch.Tensor, cls_token_num: int):
    # teacher forcing input excludes the last target token
    tf_input = gt_tokens[:, :-1]
    input_pos = torch.arange(0, tf_input.shape[1] + cls_token_num, device=gt_tokens.device)
    logits, _ = model(idx=tf_input, cond_idx=cls_ids, input_pos=input_pos)
    aligned_logits = logits[:, cls_token_num - 1:]
    return aligned_logits  # [B, T, V]


def sample_one_sequence_from_tf(logits_bt_vocab: torch.Tensor, temperature: float, top_k: int, top_p: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Given TF logits [B, T, V], sample one token sequence per item and compute
    per-token and per-sequence log-probs under the provided logits.
    Returns (indices_[B,T], per_token_log_probs_[B,T], seq_log_prob_[B])
    """
    B, T, V = logits_bt_vocab.shape
    logits = logits_bt_vocab / max(temperature, 1e-5)
    logits_flat = logits.reshape(B * T, V).contiguous()
    if top_k > 0 or top_p < 1.0:
        from autoregressive.models.generate import top_k_top_p_filtering
        logits_flat = top_k_top_p_filtering(logits_flat, top_k=top_k, top_p=top_p)
    log_probs_flat = torch.log_softmax(logits_flat, dim=-1)
    # sample indices
    probs_flat = torch.softmax(logits_flat, dim=-1)
    idx_flat = torch.multinomial(probs_flat, num_samples=1)  # [B*T, 1]
    idx_T = idx_flat.view(B, T)  # [B, T]
    # gather log-probs
    gather_lp = log_probs_flat.gather(1, idx_flat).view(B, T)  # [B, T]
    seq_log_prob = torch.sum(gather_lp, dim=-1)
    return idx_T, gather_lp, seq_log_prob


def decode_tokens_to_image(vq_model, indices_T: torch.Tensor, codebook_embed_dim: int, latent_size: int):
    # indices_T: [T]
    z = indices_T.unsqueeze(0)  # [1, T]
    qzshape = [1, codebook_embed_dim, latent_size, latent_size]
    img = vq_model.decode_code(z, qzshape)  # [1, 3, H, W] in [-1,1]
    return img


def mse_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)

def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

# No broadcast of samples in the new setting


def main(args):
    assert torch.cuda.is_available(), "Training requires GPUs."
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + global_rank)

    # Initialize Weights & Biases (rank 0 only)
    if args.use_wandb and global_rank == 0:
        if wandb is None:
            raise ImportError("wandb is not installed, please `pip install wandb` or disable --use-wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "gpt_model": args.gpt_model,
                "vq_model": args.vq_model,
                "image_size": args.image_size,
                "downsample_size": args.downsample_size,
                "codebook_size": args.codebook_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "batch_size_sample": args.sample_batch_size,
                "batch_size_train": args.train_batch_size,
                "num_generations": args.num_generations,
                "clip_range": args.clip_range,
                "adv_clip_max": args.adv_clip_max,
                "mixed_precision": args.mixed_precision,
                "data_parallel": args.data_parallel,
            },
        )

    # Load VQ (frozen)
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq_model.eval()
    with torch.no_grad():
        vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(vq_ckpt["model"])
        del vq_ckpt
    vq_model.requires_grad_(False)
    
    # Load GPT (trainable)
    latent_size = args.image_size // args.downsample_size
    gpt = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type='c2i',
    ).to(device)
    if args.gpt_ckpt:
        ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
        if args.from_fsdp:
            model_weight = ckpt
        elif "model" in ckpt:
            model_weight = ckpt["model"]
        elif "module" in ckpt:
            model_weight = ckpt["module"]
        elif "state_dict" in ckpt:
            model_weight = ckpt["state_dict"]
        else:
            raise RuntimeError("Unrecognized checkpoint format for GPT.")
        gpt.load_state_dict(model_weight, strict=False)
        del ckpt

    # FSDP wrap
    gpt = setup_fsdp_sync(gpt, args, device)

    # Optimizer
    optimizer = create_optimizer(gpt, args.weight_decay, args.lr, (args.beta1, args.beta2))

    # Dataset and per-rank loader
    crop_size = int(args.image_size)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: pil_image.convert('RGB')),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    # Respect CLI-provided dataset argument
    # dataset = build_dataset(args, transform=transform)
    dataset = HFImageDataset(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # data_iter = iter(loader)

    # Sampling loop
    global_opt_step = 0
    for epoch, (x, y)in enumerate(loader):
        #################### SAMPLING ####################
        # sample sample_batch_size (image, label) pairs for each epoch
        expanded_prompts = []
        gpt.eval()
        samples = []
        
        if global_rank == 0:
            print(f"Starting epoch {epoch}")
        if epoch > args.epochs:
            break
        
        # batch_size = 32
        # for i in range(0, images_cpu.shape[0], batch_size):
        #     x = images_cpu[i:i+batch_size]
        #     y = labels_cpu[i:i+batch_size]
            
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # repeat the image and label for num_generations times
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)
        # [B, 3, h, w] -> [B * G, 3, h, w]
        gt_x = repeat_tensor(x)
        gt_y = repeat_tensor(y)

        # Encode GT image to token indices z using VQ (no grad)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x)
        T = latent_size ** 2
        indices = indices.reshape(x.shape[0], -1)
        if indices.shape[1] > T:
            z = indices[:, :T]
        elif indices.shape[1] < T:
            pad = torch.full((x.shape[0], T - indices.shape[1]), 0, dtype=indices.dtype, device=indices.device)
            z = torch.cat([indices, pad], dim=1)
        else:
            z = indices
        if global_rank == 0:
            print(f'z shape: {z.shape}')
        z = z.long()  # [B, T]
        z = z.clamp_(0, args.codebook_size - 1)

        # Enable grads for GPT forward
        # gpt.train()
        # optimizer.zero_grad()
        # sample sequence logits with teacher forcing
        with torch.no_grad():
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                logits_bt_vocab = tf_logits(gpt, y, z, args.cls_token_num)  # [B, T, V]

        # sample num_generations sequences from the same TF logits
        all_token_indices = []
        all_token_log_probs = []
        all_seq_log_probs = []
        
        for _gen in range(args.num_generations):
            idx_T, log_probs, seq_log_probs = sample_one_sequence_from_tf(
                logits_bt_vocab, args.temperature, args.top_k, args.top_p
            )
            all_token_indices.append(idx_T)
            all_token_log_probs.append(log_probs)
            all_seq_log_probs.append(seq_log_probs)
            
        all_token_indices = torch.stack(all_token_indices, dim=1).view(-1, T)  # [B, G, T] -> [B*G, T]
        all_token_log_probs = torch.stack(all_token_log_probs, dim=1).view(-1, T)  # [B, G, T] -> [B*G, T]
        all_seq_log_probs = torch.stack(all_seq_log_probs, dim=1).view(-1)  # [B, G] -> [B*G]
        all_token_indices = all_token_indices.clamp(min=0, max=int(args.codebook_size) - 1)
        # decode all generations at once
        qzshape = [all_token_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
        with torch.no_grad():
            pred_imgs = vq_model.decode_code(all_token_indices, qzshape)  # [B*G, 3, H, W]
        # compute rewards: negative MSE vs GT image (broadcast GT across generations)
        # gt_rep = x.detach().expand_as(pred_imgs)
        # reward: -mse loss
        # TODO VAE loss
        rewards = -torch.mean((pred_imgs - gt_x) ** 2, dim=(1, 2, 3))  # [B * G]
        if args.use_wandb and global_rank == 0:
            wandb.log({
                "reward/mean": rewards.mean().item(),
                "reward/std": rewards.std().item(),
                "iter": epoch,
            }, step=epoch)
            
            for i, img in enumerate(x):
                start_idx = i * args.num_generations
                end_idx = (i + 1) * args.num_generations
                pred_image_batch = pred_imgs[start_idx:end_idx]
                grid_pred = make_grid(pred_image_batch.cpu(), nrow=min(4, args.num_generations), normalize=True, value_range=(-1, 1))
                wandb.log({
                    f"images/predicted/{i}": wandb.Image(grid_pred, caption=f"pred_imgs@iter{epoch}-idx{i}"),
                    f"images/gt/{i}": wandb.Image(img.cpu(), caption=f"gt@iter{epoch}-idx{i}"),
                    "iter": epoch,
                }, step=epoch)

        samples = {
            "class_ids": gt_y,
            "token_indices": all_token_indices,
            "token_log_probs": all_token_log_probs,
            "seq_log_probs": all_seq_log_probs,
            "rewards": rewards,
        }
        # free large tensors to mitigate fragmentation before next epoch
        del pred_imgs, all_token_indices, all_token_log_probs, all_seq_log_probs
        
        n = rewards.shape[0] // (args.num_generations)
        advantages = torch.zeros_like(rewards)
        # compute group advantage within this GPU
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
            
        samples["advantages"] = advantages
        
        #################### TRAINING ####################
        # Iterate mini-batches by slicing to avoid divisibility assumptions
        gpt.train()
        optimizer.zero_grad()
        num_samples = samples["token_indices"].shape[0]
        step_idx = 0
        for start in tqdm(
            range(0, num_samples, args.train_batch_size),
            desc=f"Epoch {epoch}: training",
            position=0,
            disable=not dist.is_initialized() or dist.get_rank() != 0,
        ):
            end = min(start + args.train_batch_size, num_samples)
            sample = {k: v[start:end] for k, v in samples.items()}

            # compute the log likelihood of the tokens under the current model
            logits_bt_vocab = tf_logits(gpt, sample["class_ids"], sample["token_indices"], args.cls_token_num)  # [N, T, V]
            log_probs = torch.log_softmax(logits_bt_vocab, dim=-1)  # [N, T, V]
            new_per_token_logps = log_probs.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)  # [N, T]

            advantages = torch.clamp(
                sample["advantages"],
                -args.adv_clip_max,
                args.adv_clip_max,
            )

            ratio = torch.exp(new_per_token_logps - sample["token_log_probs"])  # [N, T]
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - args.clip_range,
                1.0 + args.clip_range,
            )
            print(f"ratio: {ratio.mean().item()}")
            per_token_unclipped_loss = -advantages.unsqueeze(1) * ratio  # [N, T]
            per_token_clipped_loss = -advantages.unsqueeze(1) * clipped_ratio
            per_token_loss = torch.maximum(per_token_unclipped_loss, per_token_clipped_loss)  # [N, T]
            seq_loss = per_token_loss.mean(dim=-1)  # [N]
            loss = seq_loss.mean()
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            step_idx += 1
            if (step_idx) % args.gradient_accumulation_steps == 0:
                grad_norm = gpt.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_opt_step += 1
                if args.use_wandb and global_rank == 0:
                    metrics = {
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                        "train/seq_loss_mean": seq_loss.mean().item(),
                        "train/ratio_mean": ratio.mean().item(),
                        "train/clipped_ratio_mean": clipped_ratio.mean().item(),
                        "train/grad_norm": float(grad_norm),
                        "iter": epoch,
                        "opt_step": global_opt_step,
                    }
                    wandb.log(metrics, step=global_opt_step)

                # Periodic checkpoint saving
                if args.ckpt_every and args.ckpt_every > 0 and (global_opt_step % args.ckpt_every == 0):
                    if 'cuda' in str(device):
                        torch.cuda.empty_cache()
                    ckpt_dir = os.path.join(args.ckpt_dir, f"{global_opt_step:07d}")
                    ensure_dir(ckpt_dir)

                    # Prefer sharded state dict to reduce memory usage
                    if getattr(args, 'ckpt_sharded', True):
                        with FSDP.state_dict_type(
                            gpt,
                            StateDictType.SHARDED_STATE_DICT,
                        ):
                            model_state = gpt.state_dict()
                            shard_fn = (
                                f"model.shard.{dist.get_rank():05d}-of-"
                                f"{dist.get_world_size():05d}.pth"
                            )
                            torch.save(model_state, os.path.join(ckpt_dir, shard_fn))
                        dist.barrier()
                        if global_rank == 0:
                            with open(os.path.join(ckpt_dir, "state_type.txt"), "w") as f:
                                print("sharded", file=f)
                    else:
                        # Save consolidated model (rank0 only)
                        with FSDP.state_dict_type(
                            gpt,
                            StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                        ):
                            model_state = gpt.state_dict()
                            if global_rank == 0:
                                torch.save(model_state, os.path.join(ckpt_dir, "consolidated.pth"))
                        dist.barrier()
                        if global_rank == 0:
                            with open(os.path.join(ckpt_dir, "state_type.txt"), "w") as f:
                                print("full", file=f)
                    del model_state

                    # Save optimizer state per rank
                    opt_state_fn = (
                        f"optimizer.{dist.get_rank():05d}-of-"
                        f"{dist.get_world_size():05d}.pth"
                    )
                    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, opt_state_fn))
                    dist.barrier()

                    # Save resume step (rank0)
                    if global_rank == 0:
                        with open(os.path.join(ckpt_dir, "resume_step.txt"), "w") as f:
                            print(global_opt_step, file=f)
                    dist.barrier()

                    if global_rank == 0:
                        print(f"Saved checkpoint to {ckpt_dir}")
                    if 'cuda' in str(device):
                        torch.cuda.empty_cache()

        # Optionally log image grids
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
           
        if global_rank == 0:
            print(f"Grad norm: {grad_norm}")
            print(f"Loss: {loss.item()}")
            print(f"Seq loss: {seq_loss.mean().item()}")
            print(f"Per token loss: {per_token_loss.mean().item()}")
            print(f"Ratio: {ratio.mean().item()}")
            print(f"Clipped ratio: {clipped_ratio.mean().item()}")
            print(f"Per token unclipped loss: {per_token_unclipped_loss.mean().item()}")
            print(f"Per token clipped loss: {per_token_clipped_loss.mean().item()}")
            print(f"Finished epoch {epoch}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0) and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model 
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default='bf16')
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=8)
    # dataset
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    # sampling
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # training setup
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--clip-range", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # logging
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project", type=str, default="autoreg-grpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-image-every", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    # checkpointing
    parser.add_argument("--ckpt-every", type=int, default=0, help="Save checkpoint every N optimizer steps (0 to disable)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--ckpt-sharded", action='store_true', help="Save sharded model state instead of consolidated full state")
    args = parser.parse_args()
    main(args)