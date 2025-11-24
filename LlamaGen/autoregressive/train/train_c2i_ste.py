import os
import time
import argparse
import contextlib
import functools
import math

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
from tokenizer.tokenizer_image.lpips import LPIPS


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
        return len(self.ds['train'])

    def __getitem__(self, index: int):
        item = self.ds['train'][int(index)]
        image = item.get('image')
        if image is None:
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


def tf_logits(model: nn.Module, cls_ids: torch.Tensor, gt_tokens: torch.Tensor, cls_token_num: int, tf_input_override: torch.Tensor | None = None):
    tf_input = gt_tokens[:, :-1] if tf_input_override is None else tf_input_override
    input_pos = torch.arange(0, tf_input.shape[1] + cls_token_num, device=gt_tokens.device)
    logits, _ = model(idx=tf_input, cond_idx=cls_ids, input_pos=input_pos)
    aligned_logits = logits[:, cls_token_num - 1:]
    return aligned_logits  # [B, T, V]


def get_codebook_weight(vq_model) -> torch.Tensor:
    # Try common attribute names for the codebook embedding
    q = getattr(vq_model, 'quantize', None)
    if q is None:
        raise AttributeError('VQ model has no quantize module')
    emb_mod = getattr(q, 'embedding', None) or getattr(q, 'embed', None)
    if emb_mod is None:
        raise AttributeError('VQ quantizer has no embedding or embed module')
    if hasattr(emb_mod, 'weight'):
        return emb_mod.weight  # [K, D]
    raise AttributeError('VQ embedding module has no weight parameter')


def logits_to_codebook_embeddings_ste(logits_bt_vocab: torch.Tensor, codebook_weight: torch.Tensor, temperature: float, hard: bool = True) -> torch.Tensor:
    """Map logits [B, T, K] to codebook embeddings [B, T, D] via Gumbel-Softmax STE.
    - codebook_weight: [K, D]
    """
    # Use gumbel-softmax to enable straight-through with optional hard one-hot
    one_hot = F.gumbel_softmax(logits_bt_vocab, tau=max(temperature, 1e-5), hard=hard, dim=-1)  # [B, T, K]
    # [B, T, D] = [B, T, K] @ [K, D]
    zq = torch.matmul(one_hot, codebook_weight)
    return zq


def decode_embeddings_to_image(vq_model, zq_btd: torch.Tensor, latent_size: int) -> torch.Tensor:
    """Decode codebook embeddings to image using VQ-VAE decoder.
    zq_btd: [B, T, D]
    returns: [B, 3, H, W] in [-1, 1]
    """
    B, T, D = zq_btd.shape
    zq = zq_btd.view(B, latent_size, latent_size, D).permute(0, 3, 1, 2).contiguous()  # [B, D, H', W']
    # Standard VQ-VAE uses post_quant_conv before decoder
    if hasattr(vq_model, 'post_quant_conv'):
        zq = vq_model.post_quant_conv(zq)
    img = vq_model.decoder(zq)
    return img


def main(args):
    assert torch.cuda.is_available(), "Training requires GPUs."
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + global_rank)

    total_start_time = time.time()

    # Initialize Weights & Biases (rank 0 only)
    if args.use_wandb and global_rank == 0:
        if wandb is None:
            raise ImportError("wandb is not installed, please `pip install wandb` or disable --use-wandb")
        wandb.init(
            project=args.wandb_project,
            name=f'{args.wandb_run_name}-lr-{args.lr}-rec-{args.rec_weight}-lpips-{args.perc_weight}-ce-{args.aux_ce_weight}-tau-{args.ste_tau}-uncond-{args.uncond_prob}',
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
                "batch_size_train": args.train_batch_size,
                "mixed_precision": args.mixed_precision,
                "data_parallel": args.data_parallel,
                "aux_ce_weight": args.aux_ce_weight,
                "ste_tau": args.ste_tau,
                "rec_type": args.rec_type,
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

    # Perceptual model (optional)
    lpips_model = None
    if args.perc_weight > 0:
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
        class_dropout_prob=float(getattr(args, "uncond_prob", 0.1)),
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

    # Wrap with FSDP
    gpt = setup_fsdp_sync(gpt, args, device)

    # Optimizer
    optimizer = create_optimizer(gpt, args.weight_decay, args.lr, (args.beta1, args.beta2))

    # Dataset
    crop_size = int(args.image_size)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: pil_image.convert('RGB')),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
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
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    codebook_weight = get_codebook_weight(vq_model)  # [K, D]

    global_opt_step = 0

    if global_rank == 0:
        print(f"Total training epochs: {args.epochs}")
        print(f"Number of samples in training set (batches): {len(loader)}")
        print(f"World size: {world_size}")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        gpt.train()
        optimizer.zero_grad()

        if global_rank == 0:
            print(f"Starting epoch {epoch}")

        for step, (x, y) in enumerate(loader):
            
            # if global_opt_step > args.epochs:
            #     break
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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
            z = z.long().clamp_(0, args.codebook_size - 1)  # [B, T]

            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                # Teacher-forced next-token logits
                logits_bt_vocab = tf_logits(gpt, y, z, args.cls_token_num)  # [B, T, V]

            # Map logits -> codebook embeddings via STE, then decode to image
            # Do the STE projection in fp32 to keep decoder stable
            logits_for_ste = (logits_bt_vocab / max(args.temperature, 1e-5)).to(torch.float32)
            zq_flat = logits_to_codebook_embeddings_ste(logits_for_ste, codebook_weight.to(logits_for_ste.dtype), temperature=args.ste_tau, hard=True)  # [B, T, D]
            pred_imgs = decode_embeddings_to_image(vq_model, zq_flat, latent_size=latent_size)  # [B, 3, H, W]

            # Reconstruction loss
            if args.rec_type == "l1":
                rec_loss = torch.mean(torch.abs(pred_imgs - x))
            else:
                rec_loss = torch.mean((pred_imgs - x) ** 2)

            # Perceptual loss
            perc_loss = torch.tensor(0.0, device=x.device)
            if lpips_model is not None and args.perc_weight > 0:
                with torch.no_grad():
                    pass  # keep graph only for pred -> logits -> GPT; LPIPS should not backprop into GPT to avoid instability
                lp = lpips_model(pred_imgs, x)
                if lp.dim() > 1:
                    lp = lp.view(lp.shape[0], -1).mean(dim=1)
                perc_loss = lp.mean()

            # Auxiliary CE on GT tokens for stability (teacher-forced NLL)
            aux_ce_unweighted = torch.tensor(0.0, device=x.device)
            if getattr(args, "aux_ce_weight", 0.0) and args.aux_ce_weight > 0:
                ce_val = F.cross_entropy(
                    logits_bt_vocab.reshape(-1, logits_bt_vocab.shape[-1]).to(torch.float32),
                    z.reshape(-1),
                    reduction='mean'
                )
                aux_ce_unweighted = ce_val

            loss = args.rec_weight * rec_loss + args.perc_weight * perc_loss + args.aux_ce_weight * aux_ce_unweighted
            loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                grad_norm = gpt.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                if args.use_wandb and global_rank == 0 and (global_opt_step % args.log_interval == 0):
                    metrics = {
                        "loss": float(loss.item() * args.gradient_accumulation_steps),
                        "rec_loss": float(rec_loss.detach().item()),
                        "perc_loss": float(perc_loss.detach().item()),
                        "aux_ce_loss": float(aux_ce_unweighted.detach().item()),
                        "grad_norm": float(grad_norm),
                        "epoch": epoch,
                        "step": global_opt_step,
                    }
                    wandb.log(metrics, step=global_opt_step)
                global_opt_step += 1

        # Periodic checkpoint saving
        if args.ckpt_every and args.ckpt_every > 0 and ((args.epochs + 1) % args.ckpt_every == 0):
            ckpt_dir = os.path.join(args.ckpt_dir, f"{args.wandb_run_name}-lr-{args.lr}-rec-{args.rec_weight}-lpips-{args.perc_weight}-ce-{args.aux_ce_weight}-tau-{args.ste_tau}-uncond-{args.uncond_prob}-{global_opt_step:07d}")
            ensure_dir(ckpt_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

            if global_rank == 0:
                print(f"Saved checkpoint to {ckpt_dir}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # end gradient accumulation step

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if global_rank == 0:
            print(f"Finished epoch {epoch}")

    if dist.is_initialized():
        dist.barrier()
        total_time_min = (time.time() - total_start_time) / 60
        if global_rank == 0:
            h = int(total_time_min // 60)
            m = int(total_time_min % 60)
            hms = f"{h:02d}:{m:02d}"
            print(f"Total training time: {hms} ({total_time_min:.2f} min)")
            if args.use_wandb:
                try:
                    wandb.log({
                        "total_training_time_min": float(total_time_min),
                        "total_training_time_hm": hms,
                    }, step=global_opt_step)
                except Exception:
                    pass
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
    parser.add_argument("--uncond-prob", type=float, default=0.1, help="Probability to drop class conditioning (unconditional training)")
    # dataset
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data-loader", type=str, choices=["hf", "builtin"], default="hf")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    # training
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="Logit temperature prior to STE softmax")
    parser.add_argument("--ste-tau", type=float, default=1.0, help="Gumbel-Softmax temperature for STE")
    # losses
    parser.add_argument("--rec-type", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--rec-weight", type=float, default=1.0)
    parser.add_argument("--perc-weight", type=float, default=0.0)
    parser.add_argument("--aux-ce-weight", type=float, default=0.0)
    # logging
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project", type=str, default="autoreg-ste-vae")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-image-every", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    # checkpointing
    parser.add_argument("--ckpt-every", type=int, default=0, help="Save checkpoint every N optimizer steps (0 to disable)")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_ste", help="Directory to save checkpoints")

    args = parser.parse_args()
    main(args)


