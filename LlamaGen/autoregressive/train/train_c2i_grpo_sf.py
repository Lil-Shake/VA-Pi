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


def sample_free_running_mixed_context(
    model: nn.Module,
    class_ids: torch.Tensor,
    gt_tokens: torch.Tensor,
    cls_token_num: int,
    num_generations: int,
    temperature: float,
    top_k: int,
    top_p: float,
    mix_prob: float,
    batch_chunk_size: int = 0,
):
    """Autoregressively sample tokens with scheduled sampling-style mixed context.

    At each position t, the context tokens [0..t-1] are taken from ground truth with
    probability `mix_prob` per position, otherwise from already generated tokens.

    Returns (indices_[N,T], per_token_log_probs_[N,T], seq_log_prob_[N]) where N=B*num_generations.
    """
    device = gt_tokens.device
    B, T = gt_tokens.shape
    N = B * num_generations

    # Repeat inputs for G generations
    class_ids_rep = torch.repeat_interleave(class_ids, repeats=num_generations, dim=0)
    gt_rep = torch.repeat_interleave(gt_tokens, repeats=num_generations, dim=0)

    gen_tokens = torch.zeros_like(gt_rep)
    per_token_logps = torch.zeros(N, T, device=device, dtype=torch.float32)
    # mixed_context_mask[n, t, p] indicates whether GT token (True) or generated token (False)
    # was used at previous position p (< t) when predicting position t
    mixed_context_mask = torch.zeros(N, T, T, device=device, dtype=torch.bool)

    # Utility: get logits for current step t using only prefix context (memory efficient)
    def get_step_logits(prefix_tokens: torch.Tensor) -> torch.Tensor:
        # prefix_tokens shape: [N, t]
        n_total = prefix_tokens.shape[0]
        chunk = batch_chunk_size if batch_chunk_size and batch_chunk_size > 0 else B
        outputs = []
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            prefix_chunk = prefix_tokens[s:e]
            input_pos = torch.arange(0, prefix_chunk.shape[1] + cls_token_num, device=device)
            logits, _ = model(idx=prefix_chunk, cond_idx=class_ids_rep[s:e], input_pos=input_pos)  # [C, cls+t, V]
            aligned = logits[:, cls_token_num - 1:]
            outputs.append(aligned[:, -1, :])  # [C, V]
        return torch.cat(outputs, dim=0)  # [N, V]

    # Step-by-step generation with mixed context
    for t in range(T):
        # Build surrogate sequence for logits computation
        # Start from GT to ensure valid tensor shape; override past tokens with mixed context
        ctx = gt_rep.clone()
        if t > 0:
            # Use the same mixed-context mask across all generations for the same GT image
            # Sample a base mask per GT image [B, t] and repeat across generations to get [N, t]
            base_mask_bt = (torch.rand(B, t, device=device) < float(mix_prob))
            mask = torch.repeat_interleave(base_mask_bt, repeats=num_generations, dim=0)
            mixed_prev = torch.where(mask, gt_rep[:, :t], gen_tokens[:, :t])
            ctx[:, :t] = mixed_prev
            mixed_context_mask[:, t, :t] = mask

        # Compute logits for current step t using only prefix to reduce memory
        prefix = ctx[:, :t]  # [N, t]
        logits_t = get_step_logits(prefix)  # [N, V]

        # Temperature and top-k/top-p filtering
        logits_t = logits_t / max(float(temperature), 1e-5)
        if top_k > 0 or top_p < 1.0:
            from autoregressive.models.generate import top_k_top_p_filtering
            logits_t = top_k_top_p_filtering(logits_t, top_k=int(top_k), top_p=float(top_p))

        log_probs_t = torch.log_softmax(logits_t, dim=-1)
        probs_t = torch.softmax(logits_t, dim=-1)
        idx_t = torch.multinomial(probs_t, num_samples=1).squeeze(1)  # [N]
        gen_tokens[:, t] = idx_t
        per_token_logps[:, t] = log_probs_t.gather(1, idx_t.unsqueeze(1)).squeeze(1)

    seq_log_prob = per_token_logps.sum(dim=1)
    return gen_tokens, per_token_logps, seq_log_prob, mixed_context_mask

def compute_per_token_logps_mixed(
    model: nn.Module,
    class_ids: torch.Tensor,
    gt_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    cls_token_num: int,
    temperature: float,
    top_k: int,
    top_p: float,
    mixed_context_mask: torch.Tensor,
    batch_chunk_size: int = 0,
):
    """Recompute per-token log-probs under the exact mixed context used during sampling.

    For each position t, the context [0..t-1] is composed per the stored mask:
    GT when mask is True, generated tokens otherwise. Matches sampling semantics.
    Returns per_token_logps: [N, T]
    """
    device = gt_tokens.device
    N, T = gen_tokens.shape
    per_token_logps = torch.zeros(N, T, device=device, dtype=torch.float32)
    def get_step_logits(prefix_tokens: torch.Tensor) -> torch.Tensor:
        n_total = prefix_tokens.shape[0]
        chunk = batch_chunk_size if batch_chunk_size and batch_chunk_size > 0 else n_total
        outputs = []
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            prefix_chunk = prefix_tokens[s:e]
            input_pos = torch.arange(0, prefix_chunk.shape[1] + cls_token_num, device=device)
            # print(f"prefix_tokens shape: {prefix_chunk.shape}")
            logits, _ = model(idx=prefix_chunk, cond_idx=class_ids[s:e], input_pos=input_pos)  # [C, cls+t, V]
            aligned = logits[:, cls_token_num - 1:]
            outputs.append(aligned[:, -1, :])
        return torch.cat(outputs, dim=0)

    for t in range(T):
        ctx = gt_tokens.clone()
        if t > 0:
            mask = mixed_context_mask[:, t, :t]
            mixed_prev = torch.where(mask, gt_tokens[:, :t], gen_tokens[:, :t])
            ctx[:, :t] = mixed_prev
        prefix = ctx[:, :t]
        logits_t = get_step_logits(prefix)
        logits_t = logits_t / max(float(temperature), 1e-5)
        if top_k > 0 or top_p < 1.0:
            from autoregressive.models.generate import top_k_top_p_filtering
            logits_t = top_k_top_p_filtering(logits_t, top_k=int(top_k), top_p=float(top_p))
        log_probs_t = torch.log_softmax(logits_t, dim=-1)
        idx_t = gen_tokens[:, t]
        per_token_logps[:, t] = log_probs_t.gather(1, idx_t.unsqueeze(1)).squeeze(1)
    return per_token_logps

def compute_per_token_logps_mixed_window(
    model: nn.Module,
    class_ids: torch.Tensor,
    gt_tokens: torch.Tensor,
    gen_tokens: torch.Tensor,
    cls_token_num: int,
    temperature: float,
    top_k: int,
    top_p: float,
    mixed_context_mask: torch.Tensor,
    t_start: int,
    t_end: int,
    batch_chunk_size: int = 0,
):
    """Compute per-token log-probs under the exact mixed context for a window of time steps [t_start, t_end).

    Returns per_token_logps_slice: [N, t_end - t_start]
    """
    device = gt_tokens.device
    N, _ = gen_tokens.shape
    t_len = int(t_end - t_start)
    out = torch.zeros(N, t_len, device=device, dtype=torch.float32)

    def get_step_logits(prefix_tokens: torch.Tensor, cid: torch.Tensor) -> torch.Tensor:
        n_total = prefix_tokens.shape[0]
        chunk = batch_chunk_size if batch_chunk_size and batch_chunk_size > 0 else n_total
        outputs = []
        for s in range(0, n_total, chunk):
            e = min(s + chunk, n_total)
            prefix_chunk = prefix_tokens[s:e]
            if torch.cuda.is_available():
                mem = torch.cuda.mem_get_info()
                gpu_free_mem_mb = mem[0] // (1024 * 1024)
                print(f"[INFO] GPU Memory Free: {gpu_free_mem_mb} MiB")
            print(f'prefix_chunk shape:{prefix_chunk.shape}')
            input_pos = torch.arange(0, prefix_chunk.shape[1] + cls_token_num, device=device)
            logits, _ = model(idx=prefix_chunk, cond_idx=cid[s:e], input_pos=input_pos)
            aligned = logits[:, cls_token_num - 1:]
            outputs.append(aligned[:, -1, :])
        return torch.cat(outputs, dim=0)

    for local_i, t in enumerate(range(int(t_start), int(t_end))):
        ctx = gt_tokens.clone()
        if t > 0:
            mask = mixed_context_mask[:, t, :t]
            mixed_prev = torch.where(mask, gt_tokens[:, :t], gen_tokens[:, :t])
            ctx[:, :t] = mixed_prev
        prefix = ctx[:, :t]
        logits_t = get_step_logits(prefix, class_ids)
        logits_t = logits_t / max(float(temperature), 1e-5)
        if top_k > 0 or top_p < 1.0:
            from autoregressive.models.generate import top_k_top_p_filtering
            logits_t = top_k_top_p_filtering(logits_t, top_k=int(top_k), top_p=float(top_p))
        log_probs_t = torch.log_softmax(logits_t, dim=-1)
        idx_t = gen_tokens[:, t]
        out[:, local_i] = log_probs_t.gather(1, idx_t.unsqueeze(1)).squeeze(1)
    return out

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
                "kl_coef": args.kl_coef,
                "aux_ce_weight": args.aux_ce_weight,
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
        if global_rank == 0:
            print(f'z shape: {z.shape}')
        z = z.long()  # [B, T]
        z = z.clamp_(0, args.codebook_size - 1)

        gt_z = repeat_tensor(z)
        # Enable grads for GPT forward
        # gpt.train()
        # optimizer.zero_grad()
        # sample sequence logits with teacher forcing

        # Free-running sampling with mixed context (scheduled sampling) without gradients to save memory
        with torch.no_grad():
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                all_token_indices, all_token_log_probs, all_seq_log_probs, mixed_context_mask = sample_free_running_mixed_context(
                    gpt,
                    y,
                    z,
                    args.cls_token_num,
                    args.num_generations,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    getattr(args, "free_run_gt_mix_prob", 0.7),
                    batch_chunk_size=args.sample_batch_size,
                )
                
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
            
            # Combined loss -> reward is negative
            combined_loss = args.reward_rec_weight * rec_loss_vec + args.reward_perceptual_weight * perc_loss_vec
            rewards = -combined_loss  # [B * G]
        rewards_world = gather_tensor(rewards)
        dist.barrier()
        if args.use_wandb and global_rank == 0:
            # Use global_opt_step (next step idx) to keep steps monotonic; don't commit in image logs below
            print(f'rewards_world:{rewards_world}')
            print(f'rewards_world shape:{rewards_world.shape}')
            print(f'epoch:{epoch}')
            wandb.log({
                # "reward": rewards_world,
                "reward_mean": rewards_world.mean().item(),
                "reward_std": rewards_world.std().item(),
                "epoch": epoch,
            }, step=epoch)
            # INSERT_YOUR_CODE
            # Log predicted and ground truth images as grids, with reward as subtitle
            import torchvision
            import numpy as np

            # Only log up to 8 images for brevity
            max_log_images = min(8, x.shape[0])
            log_images = []
            for i in range(max_log_images):
                start_idx = i * args.num_generations
                end_idx = (i + 1) * args.num_generations
                pred_image_batch = pred_imgs[start_idx:end_idx]
                reward_batch = rewards[start_idx:end_idx].cpu().numpy()
                # Make grid of predicted images for this input
                grid_pred = torchvision.utils.make_grid(
                    pred_image_batch.cpu(),
                    nrow=min(4, args.num_generations),
                    normalize=True,
                    value_range=(-1, 1)
                )
                # Compose reward string for this group
                reward_str = ", ".join([f"{r:.2f}" for r in reward_batch])
                caption_pred = f"pred_imgs@epoch{epoch}-idx{i} | rewards: [{reward_str}]"
                # Ground truth image (single image)
                gt_img = x[i].cpu()
                # Normalize GT image to [0,1] for wandb
                gt_img_disp = (gt_img * 0.5 + 0.5).clamp(0, 1)
                caption_gt = f"gt@epoch{epoch}-idx{i}"
                log_images.append(
                    wandb.Image(grid_pred, caption=caption_pred)
                )
                log_images.append(
                    wandb.Image(gt_img_disp, caption=caption_gt)
                )
            wandb.log(
                {
                    "images": log_images
                },
                step=epoch,
                commit=False
            )
            # for i, img in enumerate(x):
            #     start_idx = i * args.num_generations
            #     end_idx = (i + 1) * args.num_generations
            #     pred_image_batch = pred_imgs[start_idx:end_idx]
            #     grid_pred = make_grid(pred_image_batch.cpu(), nrow=min(4, args.num_generations), normalize=True, value_range=(-1, 1))
            #     wandb.log({
            #         f"images_predicted_{i}": wandb.Image(grid_pred, caption=f"pred_imgs@iter{epoch}-idx{i}"),
            #         f"images_gt_{i}": wandb.Image(img.cpu(), caption=f"gt@iter{epoch}-idx{i}"),
            #     }, step=global_opt_step, commit=False)

        samples = {
            "class_ids": gt_y,
            "token_indices": all_token_indices,
            "gt_token_indices": gt_z,
            "token_log_probs": all_token_log_probs.detach(),
            "seq_log_probs": all_seq_log_probs.detach() ,
            "rewards": rewards.detach(),
            "mixed_context_mask": mixed_context_mask.detach(),
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
            if args.use_wandb and global_rank == 0:
                print('sample["class_ids"]: ', sample["class_ids"][0])
                print('sample["gt_token_indices"]: ', sample["gt_token_indices"][0])
            
            prev_training_mode = gpt.training
            gpt.eval()  # match sampling mode to avoid dropout-induced drift
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                
                # Recompute per-token log-probs under the exact mixed context used during sampling
                # Do it in windows over T to reduce peak memory and parameter unshard time
                N_local, T_local = sample["token_indices"].shape
                t_window = max(1, int(getattr(args, "eval_t_window", 64)))
                mixed_lp_parts = []
                batch_chunk_size = (
                            args.eval_chunk_size
                            if getattr(args, "eval_chunk_size", 0) and args.eval_chunk_size > 0
                            else max(1, min(args.train_batch_size, sample["class_ids"].shape[0]))
                        )
                print(f'batch_chunk_size:{batch_chunk_size}')
                for t0 in range(0, T_local, t_window):
                    t1 = min(t0 + t_window, T_local)
                    part = compute_per_token_logps_mixed_window(
                        gpt,
                        sample["class_ids"],
                        sample["gt_token_indices"],
                        sample["token_indices"],
                        args.cls_token_num,
                        args.temperature,
                        args.top_k,
                        args.top_p,
                        sample["mixed_context_mask"],
                        t0,
                        t1,
                        batch_chunk_size=batch_chunk_size,
                    )  # [N, t1-t0]
                    mixed_lp_parts.append(part)
                mixed_lp = torch.cat(mixed_lp_parts, dim=1)  # [N, T]

            if prev_training_mode:
                gpt.train()

            # Align sequence lengths across tensors to ensure a fair ratio comparison
            T_new = mixed_lp.shape[1]
            T_tok = sample["token_indices"].shape[1]
            T_oldlp = sample["token_log_probs"].shape[1]
            L = min(T_new, T_tok, T_oldlp)

            new_per_token_logps = mixed_lp[:, :L]
            old_logps_slice = sample["token_log_probs"][:, :L]
            if args.use_wandb and global_rank == 0:
                print(f'sample["class_ids"]: {sample["class_ids"][0]}')
                print(f'sample["gt_token_indices"]: {sample["gt_token_indices"][0]}')
                print(f'sample["token_indices"]: {sample["token_indices"][0]}')
                print(f'new_per_token_logs:{new_per_token_logps[0]}')
                print(f'old_per_token_logs:{sample["token_log_probs"][0]}')

            # KL(current || base) regularization against frozen base model (no top-k/p filtering)
            kl_to_base = torch.tensor(0.0, device=new_per_token_logps.device)
            if gpt_base is not None and args.kl_coef > 0:
                with torch.no_grad():
                    # Recompute full distributions at each step under mixed context for KL
                    # For efficiency, we approximate KL using teacher-forced distributions on GT tokens
                    base_logits_bt_vocab = tf_logits(
                        gpt_base, sample["class_ids"], sample["gt_token_indices"], args.cls_token_num
                    )  # [N, T, V]
                    cur_logits_bt_vocab = tf_logits(
                        gpt, sample["class_ids"], sample["gt_token_indices"], args.cls_token_num
                    )  # [N, T, V]
                logp_cur_full = torch.log_softmax(cur_logits_bt_vocab[:, :L, :], dim=-1)
                logp_base_full = torch.log_softmax(base_logits_bt_vocab[:, :L, :], dim=-1)
                p_cur_full = logp_cur_full.exp()
                kl_per_token = (p_cur_full * (logp_cur_full - logp_base_full)).sum(dim=-1)  # [N, L]
                kl_to_base = kl_per_token.mean()  # scalar
            
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
            if args.use_wandb and global_rank == 0:
                print("mean new_per_token_logps:", new_per_token_logps.mean().item())
                print("mean old token_log_probs:", sample["token_log_probs"].mean().item())
                print("mean exp(new_per_token_logps):", new_per_token_logps.exp().mean().item())
                print("mean exp(old token_log_probs):", sample["token_log_probs"].exp().mean().item())

            ratio = torch.exp(new_per_token_logps - old_logps_slice)  # [N, L]

            # KL divergence between old and new policy for the sampled tokens
            # KL(P || Q) = sum P * (log P - log Q) over aligned tokens
            kl_div = (old_logps_slice.exp() * (old_logps_slice - new_per_token_logps)).mean()

            clipped_ratio = torch.clamp(
                ratio, 
                1.0 - args.clip_range, 
                1.0 + args.clip_range,
            )
            print(f"ratio: {ratio}")
            
            # PPO loss: negative advantage * ratio (per token)
            # Note: advantages shape [N], need to unsqueeze to [N, 1] to broadcast over T
            per_token_unclipped_loss = -advantages.unsqueeze(1) * ratio  # [N, T]
            per_token_clipped_loss = -advantages.unsqueeze(1) * clipped_ratio
            per_token_loss = torch.maximum(per_token_unclipped_loss, per_token_clipped_loss)  # [N, T]
            seq_loss = per_token_loss.mean(dim=-1)  # [N]
            loss = seq_loss.mean()
            # Auxiliary CE loss (teacher-forced NLL over GT tokens)
            aux_ce_unweighted = torch.tensor(0.0, device=new_per_token_logps.device)
            if getattr(args, "aux_ce_weight", 0.0) and args.aux_ce_weight > 0:
                with {
                    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[args.mixed_precision]:
                    tf_logits_bt_vocab = tf_logits(
                        gpt, sample["class_ids"], sample["gt_token_indices"], args.cls_token_num
                    )  # [N, T, V]
                ce_val = F.cross_entropy(
                    tf_logits_bt_vocab.reshape(-1, tf_logits_bt_vocab.shape[-1]).to(torch.float32),
                    sample["gt_token_indices"].reshape(-1),
                    reduction='mean'
                )
                aux_ce_unweighted = ce_val
                loss = loss + args.aux_ce_weight * ce_val
            # Add KL-to-base penalty
            if gpt_base is not None and args.kl_coef > 0:
                print(f'loss:{loss}')
                print(f'kl:{args.kl_coef * kl_to_base}')
                loss = loss + args.kl_coef * kl_to_base
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
                        "kl_div": kl_div.item(),
                        "kl_to_base": float(kl_to_base.detach().item() if torch.is_tensor(kl_to_base) else kl_to_base),
                        "aux_ce_loss": float(aux_ce_unweighted.detach().item()),
                        "aux_ce_weight": float(getattr(args, "aux_ce_weight", 0.0)),
                        # "opt_step": global_opt_step,
                    }
                    # Commit at this step to flush any pending image logs
                    wandb.log(metrics, step=global_opt_step)

                # Periodic checkpoint saving
                if args.ckpt_every and args.ckpt_every > 0 and ((global_opt_step + 1) % args.ckpt_every == 0):
                    if 'cuda' in str(device):
                        torch.cuda.empty_cache()
                    ckpt_dir = os.path.join(args.ckpt_dir, f"{global_opt_step:07d}")
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
                print(f'epoch-{epoch}: step:{global_opt_step}')
                

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
            print(f"KL divergence: {kl_div.item()}")
            if gpt_base is not None and args.kl_coef > 0:
                print(f"KL to base: {kl_to_base.item()}")
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
    # reward config
    parser.add_argument("--reward-rec-type", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--reward-rec-weight", type=float, default=1.0)
    parser.add_argument("--reward-perceptual-weight", type=float, default=1.0)
    parser.add_argument("--free-run-gt-mix-prob", type=float, default=0.7, help="Probability to use GT tokens in context during free-running sampling")
    parser.add_argument("--aux-ce-weight", type=float, default=0.0, help="Weight for auxiliary CE(gt) loss in training loop (0 disables)")
    # KL regularization
    parser.add_argument("--kl-coef", type=float, default=0.01, help="Weight for KL(current||base) penalty")
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
    # evaluation / recompute chunking
    parser.add_argument("--eval-chunk-size", type=int, default=0, help="Chunk size for eval-time per-step logits recompute; 0 means no chunking")
    parser.add_argument("--eval-t-window", type=int, default=64, help="Time window size when recomputing mixed log-probs over sequence length")
    args = parser.parse_args()
    main(args)