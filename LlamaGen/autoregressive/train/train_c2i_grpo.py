import os
import time
import argparse
import contextlib
import functools

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
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
from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator  # type: ignore

def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

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
            name=f'{args.wandb_run_name}-mse-{args.reward_rec_weight}-lpips-{args.reward_perceptual_weight}-gan-{args.reward_use_gan}-kl-{args.use_kl_loss}-ce-{args.aux_ce_weight}',
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
                "kl_coef": args.kl_coef,
                "reward_use_gan": args.reward_use_gan,
                "reward_gan_weight": args.reward_gan_weight,
                "reward_disc_dim": args.reward_disc_dim,
                "reward_disc_num_layers": args.reward_disc_num_layers,
                "aux_ce_weight": args.aux_ce_weight,
                "reward_use_embedcos": getattr(args, "reward_use_embedcos", False),
                "reward_embedcos_weight": getattr(args, "reward_embedcos_weight", 0.0),
                "train_mode": args.train_mode,
                # "reward_use_mse": getattr(args, "reward_use_mse", False),
            },
            mode="online"
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
    # Perceptual loss model for reward
    lpips_model = LPIPS().to(device)
    lpips_model.eval()
    lpips_model.requires_grad_(False)

    # Optional discriminator for GAN reward
    disc_model = None
    if getattr(args, "reward_use_gan", False):
        disc_model = PatchGANDiscriminator(
            input_nc=3,
            n_layers=args.reward_disc_num_layers,
            ndf=args.reward_disc_dim,
        ).to(device)
        disc_ckpt = getattr(args, "reward_disc_ckpt", None)
        if disc_ckpt:
            ckpt = torch.load(disc_ckpt, map_location="cpu")
            state = ckpt.get("model", ckpt)
            try:
                disc_model.load_state_dict(state, strict=False)
            except Exception:
                disc_model.load_state_dict(state)
            del ckpt
        disc_model.eval()
        disc_model.requires_grad_(False)

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

    # Create frozen base model for KL regularization (before FSDP wrap)
    gpt_base = None
    if getattr(args, "kl_coef", 0.0) and args.kl_coef > 0:
        # Build the base model (same config as gpt)
        gpt_base = GPT_models[args.gpt_model](
            vocab_size=args.codebook_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type='c2i',
        ).to(device)
        base_ckpt_path = getattr(args, "kl_base_ckpt", None) or args.gpt_ckpt
        if base_ckpt_path:
            base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
            if args.from_fsdp:
                base_weight = base_ckpt
            elif "model" in base_ckpt:
                base_weight = base_ckpt["model"]
            elif "module" in base_ckpt:
                base_weight = base_ckpt["module"]
            elif "state_dict" in base_ckpt:
                base_weight = base_ckpt["state_dict"]
            else:
                raise RuntimeError("Unrecognized checkpoint format for base GPT.")
            gpt_base.load_state_dict(base_weight, strict=False)
            del base_ckpt
        else:
            # Fall back to copying current initialization if no checkpoint provided
            gpt_base.load_state_dict(gpt.state_dict(), strict=False)
        gpt_base.eval()
        gpt_base.requires_grad_(False)
        # Match parameter dtype with the FSDP-wrapped model's param dtype to minimize numeric drift
        _param_dtype_map = {
            "fp32": torch.float,
            "tf32": torch.float,  # params remain fp32 under TF32 matmul
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        try:
            base_param_dtype = _param_dtype_map.get(args.mixed_precision, torch.float)
            gpt_base.to(dtype=base_param_dtype)
        except Exception:
            pass

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
    # Respect CLI-provided dataset argument; allow switching between HF Arrow and built-in dataset
    if args.data_loader == "builtin":
        dataset = build_dataset(args, transform=transform)
    else:
        dataset = HFImageDataset(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
    )
    loader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # data_iter = iter(loader)

    # Sampling loop
    global_opt_step = 0
    if global_rank == 0:
        num_seq_per_epoch = world_size * args.sample_batch_size * args.num_generations
        num_grad_updates_per_epoch = (args.sample_batch_size * args.num_generations) / (args.train_batch_size * args.gradient_accumulation_steps)
        print(f"Total training epochs: {args.epochs}")
        print(f"Number of samples in training set: {len(loader)}")
        print(f"World size: {world_size}")
        print(f"Number of samples per epoch: {num_seq_per_epoch}")
        print(f"Number of gradient updates per epoch: {num_grad_updates_per_epoch}")
        print(f"Number of generations per group: {args.num_generations}")
    
    for epoch in range(args.epochs):
        #################### SAMPLING ####################
        # sample sample_batch_size (image, label) pairs for each epoch
        expanded_prompts = []
        gpt.eval()
        samples = []
        
        if global_rank == 0:
            print(f"Starting epoch {epoch}")
        
        # Ensure per-epoch reshuffle and unique shards across ranks
        sampler.set_epoch(epoch)
        data_iter = iter(loader)
        x, y = next(data_iter)
        
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
        # print(f"gt_x shape: {gt_x.shape}")
        # print(f"gt_y shape: {gt_y.shape}")
        # print(f"gt_y: {gt_y}")

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
        # if global_rank == 0:
        #     print(f'z shape: {z.shape}')
        z = z.long()  # [B, T]
        z = z.clamp_(0, args.codebook_size - 1)

        gt_z = repeat_tensor(z)
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
            # if global_rank == 0:
            #     print(f'y: {y[0]}')
            #     print(f'gt_code:{z[0]}')

        # sample num_generations sequences from the same TF logits
        all_token_indices = []
        all_token_log_probs = []
        all_seq_log_probs = []
        
        for _gen in range(args.num_generations):
            idx_T, log_probs, seq_log_probs = sample_one_sequence_from_tf(
                logits_bt_vocab, args.temperature, args.top_k, args.top_p
            )
            # if _gen == 0 and global_rank == 0:
            #     print(f'old_code: {idx_T[0]}')
            #     print(f'log_probs: {log_probs[0]}')
            #     print(f'seq_log_probs: {seq_log_probs[0]}')
                
            all_token_indices.append(idx_T)
            all_token_log_probs.append(log_probs)
            all_seq_log_probs.append(seq_log_probs)
            
        all_token_indices = torch.stack(all_token_indices, dim=1).view(-1, T)  # [B, G, T] -> [B*G, T]
        all_token_log_probs = torch.stack(all_token_log_probs, dim=1).view(-1, T)  # [B, G, T] -> [B*G, T]
        all_seq_log_probs = torch.stack(all_seq_log_probs, dim=1).view(-1)  # [B, G] -> [B*G]
        all_token_indices = all_token_indices.clamp(min=0, max=int(args.codebook_size) - 1)
        # print(f'all_token_log_probs:{all_token_log_probs}')
        # decode all generations at once
        
        qzshape = [all_token_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
        def process_in_chunks(process_fn, *inputs, chunk_size=8, **kwargs):
            """
            Utility function to process inputs in chunks to avoid OOM.
            - process_fn: function to apply to each chunk (should return a tensor or list of tensors)
            - *inputs: tensors to be chunked along the first dimension
            - chunk_size: size of each chunk
            - **kwargs: additional keyword arguments passed to process_fn
            Returns: concatenated output from all chunks
            """
            total_size = inputs[0].shape[0]
            results = []
            for start in range(0, total_size, chunk_size):
                end = min(start + chunk_size, total_size)
                chunk_inputs = [inp[start:end] for inp in inputs]
                result = process_fn(*chunk_inputs, **kwargs)
                results.append(result)
            if isinstance(results[0], torch.Tensor):
                return torch.cat(results, dim=0)
            else:
                # If process_fn returns a tuple/list of tensors, concatenate each
                return [torch.cat([r[i] for r in results], dim=0) for i in range(len(results[0]))]

        with torch.no_grad():
            # Decode all generations at once, in chunks
            def decode_chunk(chunk_indices):
                chunk_qzshape = [chunk_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
                return vq_model.decode_code(chunk_indices, chunk_qzshape)  # [chunk, 3, H, W]
            pred_imgs = process_in_chunks(
                decode_chunk, all_token_indices, chunk_size=8
            )  # [B*G, 3, H, W]

            # Optional embedding cosine-similarity loss between GT pre-quantization z and predicted codebook embedding z_q (before decoder)
            # Compute once per epoch/sample batch; shapes match codebook_embed_dim and latent grid
            embedcos_loss_vec = None
            if getattr(args, "reward_use_embedcos", False):
                # GT pre-quantization embedding z (after encoder + quant_conv), shape [B, C_e, H', W']
                z_gt_pre = vq_model.quant_conv(vq_model.encoder(x))
                gt_vec = z_gt_pre.view(z_gt_pre.shape[0], -1)  # [B, D]
                gt_vec_rep = repeat_tensor(gt_vec)  # [B*G, D]

                def embedcos_chunk(chunk_indices, gt_vec_chunk):
                    # Look up codebook embeddings for predicted indices to get z_q (pre-decoder), then compute 1 - cosine sim
                    chunk_shape = [chunk_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
                    zq_chunk = vq_model.quantize.get_codebook_entry(chunk_indices, shape=chunk_shape, channel_first=True)
                    zq_vec = zq_chunk.view(chunk_indices.shape[0], -1)
                    cos_sim = F.cosine_similarity(zq_vec, gt_vec_chunk, dim=1)
                    return 1.0 - cos_sim

                embedcos_loss_vec = process_in_chunks(
                    embedcos_chunk, all_token_indices, gt_vec_rep, chunk_size=16
                )  # [B*G]

            # Reconstruction loss per-sample
            if args.reward_rec_type == "l1":
                rec_loss_vec = torch.mean(torch.abs(pred_imgs - gt_x), dim=(1, 2, 3))
            else:
                rec_loss_vec = torch.mean((pred_imgs - gt_x) ** 2, dim=(1, 2, 3))

            # Perceptual loss (LPIPS) per-sample, compute in small chunks to avoid OOM
            def lpips_chunk(pred_imgs_chunk, gt_x_chunk):
                perc_loss_chunk = lpips_model(pred_imgs_chunk, gt_x_chunk)
                if perc_loss_chunk.dim() > 1:
                    perc_loss_chunk = perc_loss_chunk.view(perc_loss_chunk.shape[0], -1).mean(dim=1)
                return perc_loss_chunk

            perc_loss_vec = process_in_chunks(
                lpips_chunk, pred_imgs, gt_x, chunk_size=8
            )
            
            # GAN generator adversarial loss (optional, per-sample)
            if disc_model is not None and getattr(args, "reward_use_gan", False):
                def gan_chunk(pred_imgs_chunk):
                    logits_fake = disc_model(pred_imgs_chunk)
                    # Hinge generator loss: -E[logits_fake]
                    return -logits_fake.view(logits_fake.shape[0], -1).mean(dim=1)
                gan_loss_vec = process_in_chunks(gan_chunk, pred_imgs, chunk_size=8)
            else:
                gan_loss_vec = torch.zeros_like(rec_loss_vec)

            # Combined loss -> reward is negative
            combined_loss = (
                args.reward_rec_weight * rec_loss_vec +
                args.reward_perceptual_weight * perc_loss_vec +
                args.reward_gan_weight * gan_loss_vec
            )
            if embedcos_loss_vec is not None:
                combined_loss = combined_loss + args.reward_embedcos_weight * embedcos_loss_vec
            rewards = -combined_loss  # [B * G]
        rewards_world = gather_tensor(rewards)
        dist.barrier()
        if args.use_wandb and global_rank == 0:
            # Use global_opt_step (next step idx) to keep steps monotonic; don't commit in image logs below
            # print(f'rewards_world:{rewards_world}')
            # print(f'rewards_world shape:{rewards_world.shape}')
            # print(f'epoch:{epoch}')
            wandb.log({
                # "reward": rewards_world,
                "reward_mean": rewards_world.mean().item(),
                "reward_std": rewards_world.std().item(),
                "epoch": epoch,
                "rec_loss_mean": rec_loss_vec.mean().item(),
                "perc_loss_mean": perc_loss_vec.mean().item(),
                "gan_loss_mean": (gan_loss_vec.mean().item() if 'gan_loss_vec' in locals() else 0.0),
                "embedcos_loss_mean": (embedcos_loss_vec.mean().item() if embedcos_loss_vec is not None else 0.0),
            }, step=global_opt_step)
            # INSERT_YOUR_CODE
            # Log predicted and ground truth images as grids, with reward as subtitle
            import torchvision
            import numpy as np

            # Only log up to 8 images for brevity
        #     max_log_images = min(8, x.shape[0])
        #     log_images = []
        #     for i in range(max_log_images):
        #         start_idx = i * args.num_generations
        #         end_idx = (i + 1) * args.num_generations
        #         pred_image_batch = pred_imgs[start_idx:end_idx]
        #         reward_batch = rewards[start_idx:end_idx].cpu().numpy()
        #         # Make grid of predicted images for this input
        #         grid_pred = torchvision.utils.make_grid(
        #             pred_image_batch.cpu(),
        #             nrow=min(4, args.num_generations),
        #             normalize=True,
        #             value_range=(-1, 1)
        #         )
        #         # Compose reward string for this group
        #         reward_str = ", ".join([f"{r:.2f}" for r in reward_batch])
        #         caption_pred = f"pred_imgs@epoch{epoch}-idx{i} | rewards: [{reward_str}]"
        #         # Ground truth image (single image)
        #         gt_img = x[i].cpu()
        #         # Normalize GT image to [0,1] for wandb
        #         gt_img_disp = (gt_img * 0.5 + 0.5).clamp(0, 1)
        #         caption_gt = f"gt@epoch{epoch}-idx{i}"
        #         log_images.append(
        #             wandb.Image(grid_pred, caption=caption_pred)
        #         )
        #         log_images.append(
        #             wandb.Image(gt_img_disp, caption=caption_gt)
        #         )
        #     wandb.log(
        #         {
        #             "images": log_images
        #         },
        #         step=epoch,
        #         commit=False
        #     )
        #     # for i, img in enumerate(x):
        #     #     start_idx = i * args.num_generations
        #     #     end_idx = (i + 1) * args.num_generations
        #     #     pred_image_batch = pred_imgs[start_idx:end_idx]
        #     #     grid_pred = make_grid(pred_image_batch.cpu(), nrow=min(4, args.num_generations), normalize=True, value_range=(-1, 1))
        #     #     wandb.log({
        #     #         f"images_predicted_{i}": wandb.Image(grid_pred, caption=f"pred_imgs@iter{epoch}-idx{i}"),
        #     #         f"images_gt_{i}": wandb.Image(img.cpu(), caption=f"gt@iter{epoch}-idx{i}"),
        #     #     }, step=global_opt_step, commit=False)

        # samples = {
        #     "class_ids": gt_y,
        #     "token_indices": all_token_indices,
        #     "gt_token_indices": gt_z,
        #     "token_log_probs": all_token_log_probs.detach(),
        #     "seq_log_probs": all_seq_log_probs.detach() ,
        #     "rewards": rewards.detach(),
        # }
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
            # if args.use_wandb and global_rank == 0:
            #     print('sample["class_ids"]: ', sample["class_ids"][0])
            #     print('sample["gt_token_indices"]: ', sample["gt_token_indices"][0])
            
            prev_training_mode = gpt.training
            if args.train_mode == "eval":
                gpt.eval()  # match sampling mode to avoid dropout-induced drift
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                logits_bt_vocab = tf_logits(gpt, sample["class_ids"], sample["gt_token_indices"], args.cls_token_num)  # [N, T, V]
                
            if args.train_mode == "eval" and prev_training_mode:
                gpt.train()
                
            logits = logits_bt_vocab / max(args.temperature, 1e-5)
            logits_flat = logits.reshape(-1, logits.shape[-1]).contiguous()
            # Do NOT apply top-k/p filtering during training; keep full distribution for stable ratio/KL
            log_probs = torch.log_softmax(logits_flat, dim=-1).view_as(logits_bt_vocab)
            new_per_token_logps = log_probs.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)
            # log_probs = torch.log_softmax(logits_bt_vocab, dim=-1)  # [N, T, V]
            # new_per_token_logps = log_probs.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)  # [N, T]
            # if args.use_wandb and global_rank == 0:
            #     print(f'sample["class_ids"]: {sample["class_ids"][0]}')
            #     print(f'sample["gt_token_indices"]: {sample["gt_token_indices"][0]}')
            #     print(f'sample["token_indices"]: {sample["token_indices"][0]}')
            #     print(f'new_per_token_logs:{new_per_token_logps[0]}')
            #     print(f'old_per_token_logs:{sample["token_log_probs"][0]}')

            # GRPO-style KL term against frozen base model (no top-k/p filtering)
            per_token_kl = None
            if gpt_base is not None and args.kl_coef > 0:
                with torch.no_grad():
                    # Run base forward under the SAME mixed-precision context as current to avoid dtype drift
                    with {
                        "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                        "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                        "fp32": contextlib.nullcontext(),
                        "tf32": contextlib.nullcontext(),
                    }[args.mixed_precision]:
                        base_logits_bt_vocab = tf_logits(
                            gpt_base, sample["class_ids"], sample["gt_token_indices"], args.cls_token_num
                        )  # [N, T, V]
                # Compute KL in fp32 for stability; same temperature, no filtering
                temp = max(args.temperature, 1e-5)
                cur_logits_for_kl = (logits_bt_vocab / temp).to(torch.float32)
                base_logits_for_kl = (base_logits_bt_vocab / temp).to(torch.float32)
                cur_logps_nofilter = torch.log_softmax(cur_logits_for_kl, dim=-1)
                base_logps_nofilter = torch.log_softmax(base_logits_for_kl, dim=-1)
                cur_token_logps_for_kl = cur_logps_nofilter.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                ref_token_logps_for_kl = base_logps_nofilter.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                # per-token KL used in GRPO: exp(ref - cur) - (ref - cur) - 1
                per_token_kl = torch.exp(ref_token_logps_for_kl - cur_token_logps_for_kl) - (ref_token_logps_for_kl - cur_token_logps_for_kl) - 1
                print(f"per_token_kl: {per_token_kl.mean().item()}")
            
            advantages = torch.clamp(
                sample["advantages"],
                -args.adv_clip_max,
                args.adv_clip_max,
            )
            
            # The ratio should be close to 1 if the new and old log-probs are for the same tokens under similar models.
            # Let's check the calculation of log-probs and the ratio.
            # new_per_token_logps: log-prob of sampled tokens under *current* model, shape [N, T]
            # sample["token_log_probs"]: log-prob of sampled tokens under *old* model, shape [N, T]
            # ratio = exp(new_logp - old_logp) = new_prob / old_prob

            # Sanity check: print means and stds to debug
            # if args.use_wandb and global_rank == 0:
            #     print("mean new_per_token_logps:", new_per_token_logps.mean().item())
            #     print("mean old token_log_probs:", sample["token_log_probs"].mean().item())
            #     print("mean exp(new_per_token_logps):", new_per_token_logps.exp().mean().item())
            #     print("mean exp(old token_log_probs):", sample["token_log_probs"].exp().mean().item())

            ratio = torch.exp(new_per_token_logps - sample["token_log_probs"])  # [N, T]

            # KL divergence between old and new policy for the sampled tokens
            # KL(P || Q) = sum P * (log P - log Q)
            # Here, sample["token_log_probs"] is log P (old), new_per_token_logps is log Q (new)
            # So, KL(old || new) = exp(log P) * (log P - log Q)
            # kl_div = (sample["token_log_probs"].exp() * (sample["token_log_probs"] - new_per_token_logps)).mean()

            clipped_ratio = torch.clamp(
                ratio, 
                1.0 - args.clip_range, 
                1.0 + args.clip_range,
            )
            # print(f"ratio: {ratio}")
            
            # PPO loss: negative advantage * ratio (per token)
            # Note: advantages shape [N], need to unsqueeze to [N, 1] to broadcast over T
            per_token_unclipped_loss = -advantages.unsqueeze(1) * ratio  # [N, T]
            per_token_clipped_loss = -advantages.unsqueeze(1) * clipped_ratio
            per_token_loss = torch.maximum(per_token_unclipped_loss, per_token_clipped_loss)  # [N, T]
            
            # add GRPO KL penalty
            if per_token_kl is not None and args.use_kl_loss:
                    print(f"per_token_kl: {per_token_kl.mean().item()}")
                    print(f"per_token_loss: {per_token_loss.mean().item()}")
                    per_token_loss = per_token_loss + args.kl_coef * per_token_kl
            
            seq_loss = per_token_loss.mean(dim=-1)  # [N]
            loss = seq_loss.mean()

            # Auxiliary CE loss (teacher-forced NLL over GT tokens)
            aux_ce_unweighted = torch.tensor(0.0, device=logits_bt_vocab.device)
            if getattr(args, "aux_ce_weight", 0.0) and args.aux_ce_weight > 0:
                ce_val = F.cross_entropy(
                    logits_bt_vocab.reshape(-1, logits_bt_vocab.shape[-1]).to(torch.float32),
                    sample["gt_token_indices"].reshape(-1),
                    reduction='mean'
                )
                aux_ce_unweighted = ce_val
                loss = loss + args.aux_ce_weight * ce_val
            # (GRPO) KL penalty already added at per-token level above
            loss = loss / args.gradient_accumulation_steps

            # If ratio is not close to 1, possible reasons:
            # - The sampled tokens (sample["token_indices"]) are not the same as those used to compute sample["token_log_probs"]
            # - The log-probs are not aligned (e.g., off by one in sequence)
            # - The model weights have changed a lot between sampling and training
            # - Numerical instability if log-probs are very small/large
            # - Check that sample["token_indices"] is the same as used for both log-prob calculations
            
            loss.backward()
            step_idx += 1
            if (step_idx) % args.gradient_accumulation_steps == 0:
                grad_norm = gpt.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                # Calculate the clip ratio: proportion of tokens where |ratio - 1| > clip_range
                clip_mask = (torch.abs(ratio - 1.0) > args.clip_range)
                clip_ratio = clip_mask.float().mean().item()
                if args.use_wandb and global_rank == 0:
                    metrics = {
                        "loss": loss.item() * args.gradient_accumulation_steps,
                        "seq_loss_mean": seq_loss.mean().item(),
                        "ratio_mean": ratio.mean().item(),
                        "clipped_ratio_mean": clipped_ratio.mean().item(),
                        "grad_norm": float(grad_norm),
                        "clip_ratio": clip_ratio,
                        # "kl_div": kl_div.item(),
                        "kl_to_base": float(per_token_kl.mean().detach().item()),
                        "aux_ce_loss": float(aux_ce_unweighted.detach().item()),
                        "aux_ce_weight": float(args.aux_ce_weight),
                        # "opt_step": global_opt_step,
                    }
                    # Commit at this step to flush any pending image logs
                    wandb.log(metrics, step=global_opt_step)

                # Periodic checkpoint saving
                if args.ckpt_every and args.ckpt_every > 0 and ((global_opt_step + 1) % args.ckpt_every == 0):
                    if 'cuda' in str(device):   
                        torch.cuda.empty_cache()
                    ckpt_dir = os.path.join(args.ckpt_dir, f"{args.wandb_run_name}-mse-{args.reward_rec_weight}-lpips-{args.reward_perceptual_weight}-gan-{args.reward_use_gan}-kl-{args.use_kl_loss}-ce-{args.aux_ce_weight}-{global_opt_step:07d}")
                    ensure_dir(ckpt_dir)

                    # Prefer sharded state dict to reduce memory usage
                    # if getattr(args, 'ckpt_sharded', True):
                    #     with FSDP.state_dict_type(
                    #         gpt,
                    #         StateDictType.SHARDED_STATE_DICT,
                    #     ):
                    #         model_state = gpt.state_dict()
                    #         shard_fn = (
                    #             f"model.shard.{dist.get_rank():05d}-of-"
                    #             f"{dist.get_world_size():05d}.pth"
                    #         )
                    #         torch.save(model_state, os.path.join(ckpt_dir, shard_fn))
                    #     dist.barrier()
                    #     if global_rank == 0:
                    #         with open(os.path.join(ckpt_dir, "state_type.txt"), "w") as f:
                    #             print("sharded", file=f)
                    # else:
                    #     # Save consolidated model (rank0 only)
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
                    # opt_state_fn = (
                    #     f"optimizer.{dist.get_rank():05d}-of-"
                    #     f"{dist.get_world_size():05d}.pth"
                    # )
                    # torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, opt_state_fn))
                    # dist.barrier()

                    # # Save resume step (rank0)
                    # if global_rank == 0:
                    #     with open(os.path.join(ckpt_dir, "resume_step.txt"), "w") as f:
                    #         print(global_opt_step, file=f)
                    dist.barrier()

                    if global_rank == 0:
                        print(f"Saved checkpoint to {ckpt_dir}")
                    if 'cuda' in str(device):
                        torch.cuda.empty_cache()
                
                global_opt_step += 1
                # print(f'epoch-{epoch}: step:{global_opt_step}')
                

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
            # print(f"KL divergence: {kl_div.item()}")
            if gpt_base is not None and args.kl_coef > 0:
                print(f"KL to base: {per_token_kl.mean().item()}")
            print(f"Finished epoch {epoch}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if args.use_wandb and (not dist.is_initialized() or dist.get_rank() == 0):
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
    parser.add_argument("--data-loader", type=str, choices=["hf", "builtin"], default="hf", help="Choose 'hf' for Arrow HF loader or 'builtin' for build_dataset path")
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
    parser.add_argument("--train-mode", type=str, choices=["train", "eval"], default="train", help="Training mode")
    # reward config
    parser.add_argument("--reward-rec-type", type=str, choices=["l1", "l2"], default="l2")
    # parser.add_argument("--reward-use-mse", action='store_true', help="Enable using MSE reconstruction as reward term (overrides --reward-rec-type)")
    parser.add_argument("--reward-rec-weight", type=float, default=1.0)
    parser.add_argument("--reward-perceptual-weight", type=float, default=1.0)
    parser.add_argument("--reward-use-gan", action='store_true')
    parser.add_argument("--reward-gan-weight", type=float, default=1.0)
    parser.add_argument("--reward-disc-dim", type=int, default=64)
    parser.add_argument("--reward-disc-num-layers", type=int, default=3)
    parser.add_argument("--reward-disc-ckpt", type=str, default=None)
    parser.add_argument("--reward-use-embedcos", action='store_true', help="Enable cosine-similarity reward between GT pre-quantization z and predicted codebook embedding z_q")
    parser.add_argument("--reward-embedcos-weight", type=float, default=0.0, help="Weight for cosine-similarity embedding reward (1 - cos sim) in combined loss")
    parser.add_argument("--aux-ce-weight", type=float, default=0.0, help="Weight for auxiliary CE(gt) loss in training loop (0 disables)")
    # KL regularization
    parser.add_argument("--kl-coef", type=float, default=0.01, help="Weight for KL(current||base) penalty")
    parser.add_argument("--use-kl-loss", action='store_true', help="Use KL(current||base) penalty")
    parser.add_argument("--kl-base-ckpt", type=str, default=None, help="Checkpoint path for frozen base GPT (defaults to --gpt-ckpt)")
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