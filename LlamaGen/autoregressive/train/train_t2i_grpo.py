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
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from tqdm import tqdm

from autoregressive.models.gpt import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator  # type: ignore

import logging

logger = logging.getLogger("train_t2i_grpo")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def setup_ddp(model: nn.Module, device: int) -> DDP:
    ddp_model = DDP(model.to(device), device_ids=[device])
    torch.cuda.synchronize()
    return ddp_model


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


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def perturb_teacher_forcing_inputs(gt_tokens_full: torch.Tensor, args: argparse.Namespace, rng: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    device_local = gt_tokens_full.device
    tf_base = gt_tokens_full[:, :-1]
    if not getattr(args, "use_token_noise", False):
        return tf_base, torch.zeros((gt_tokens_full.shape[0],), device=device_local, dtype=torch.float32)
    prob = max(0.0, min(1.0, float(args.token_noise_prob)))
    N, Tm1 = tf_base.shape
    bern = torch.bernoulli(torch.full((N, Tm1), prob, device=device_local))
    ui = torch.randint(low=0, high=int(args.codebook_size), size=(N, Tm1), device=device_local, generator=rng, dtype=tf_base.dtype)
    tf_noisy = torch.where(bern.bool(), ui, tf_base)
    eps = torch.full((N,), prob, device=device_local, dtype=torch.float32)
    return tf_noisy, eps


def tf_logits_t2i(model: nn.Module,
                  cond_tokens: torch.Tensor,
                  gt_tokens: torch.Tensor,
                  attn_mask_full: torch.Tensor,
                  valid_mask: torch.Tensor | None,
                  tf_input_override: torch.Tensor | None = None) -> torch.Tensor:
    tf_input = gt_tokens[:, :-1] if tf_input_override is None else tf_input_override
    # Slice attention mask to match input (drop the last target step for teacher-forcing)
    # attn_mask_full: [B, n_head, S, S] where S = cls_token_num + T
    # Ensure mask is 4D [B, 1, S, S] as in train_t2i.py
    if attn_mask_full.dim() == 3:
        attn_mask_full = attn_mask_full.reshape(attn_mask_full.shape[0], 1, attn_mask_full.shape[-2], attn_mask_full.shape[-1])
    attn_mask = attn_mask_full[:, :, :-1, :-1]
    # Only pass targets when the model is in training mode to avoid CE in eval
    targets_to_pass = gt_tokens if getattr(model, "training", False) else None
    logits, _ = model(
        cond_idx=cond_tokens,
        idx=tf_input,
        targets=targets_to_pass,
        mask=attn_mask,
        valid=valid_mask,
    )
    # Align logits to target token positions: take the last T positions
    T = gt_tokens.shape[1]
    aligned = logits[:, -T:]
    return aligned  # [B, T, V]

def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

def main(args: argparse.Namespace):
    assert torch.cuda.is_available(), "Training requires GPUs."
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    torch.manual_seed(args.global_seed * world_size + global_rank)

    # VQ (frozen)
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
       
    # Perceptual loss
    lpips_model = LPIPS().to(device)
    lpips_model.eval()
    lpips_model.requires_grad_(False)
    
    # Optional GAN discriminator
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
        
    latent_size = args.image_size // args.downsample_size
    gpt = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type='t2i',
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
    ).to(device)
    
    num_params = sum(p.numel() for p in gpt.parameters())
    num_gb = num_params * 4 / (1024 ** 3)  # 4 bytes per parameter (fp32/bf16)
    logger.info(f"GPT Parameters: {num_params:,} ({num_gb:.2f} GB)")

    # GPT (trainable)
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        print(f"checkpoint: {checkpoint.keys()}")
        if args.from_fsdp:
            model_weight = checkpoint
        elif "model" in checkpoint:
            model_weight = checkpoint["model"]
        elif "module" in checkpoint:
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise RuntimeError("Unrecognized checkpoint format for GPT.")
        missing, unexpected = gpt.load_state_dict(model_weight, strict=False)
        if missing or unexpected:
            logger.warning(f"While loading GPT, missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        del checkpoint
        logger.info(f"Loaded model weights from: {args.gpt_ckpt}")
    train_steps = 0
    start_epoch = 0

    # Optional KL base model for regularization
    gpt_base = None
    if getattr(args, "use_kl_loss", False) and getattr(args, "kl_base_ckpt", None):
        gpt_base = GPT_models[args.gpt_model](
            vocab_size=args.codebook_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type='t2i',
            resid_dropout_p=args.dropout_p,
            ffn_dropout_p=args.dropout_p,
            token_dropout_p=args.token_dropout_p,
        ).to(device)
        base_ckpt = torch.load(args.kl_base_ckpt, map_location="cpu")
        if "model" in base_ckpt:
            base_state = base_ckpt["model"]
        elif "module" in base_ckpt:
            base_state = base_ckpt["module"]
        elif "state_dict" in base_ckpt:
            base_state = base_ckpt["state_dict"]
        else:
            base_state = base_ckpt
        gpt_base.load_state_dict(base_state, strict=False)
        gpt_base.eval()
        gpt_base.requires_grad_(False)
        del base_ckpt
        logger.info(f"Loaded KL base model from: {args.kl_base_ckpt}")

    # DDP wrap
    gpt = setup_ddp(gpt, device)

    # Optimizer
    optimizer = create_optimizer(gpt, args.weight_decay, args.lr, (args.beta1, args.beta2))

    # Dataset (builtin t2i)
    crop_size = int(args.image_size)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: pil_image.convert('RGB')),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
    )
    loader = DataLoader(dataset, batch_size=args.sample_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    global_opt_step = 0
    latent_T = latent_size ** 2
    
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
        gpt.eval()
        sampler.set_epoch(epoch)
        data_iter = iter(loader)
        x, y, attn_mask, valid = next(data_iter)  # builtin t2i yields 4-tuple
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True) if valid is not None else None

        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(x)
        indices = indices.reshape(x.shape[0], -1)
        if indices.shape[1] > latent_T:
            z = indices[:, :latent_T]
        elif indices.shape[1] < latent_T:
            pad = torch.full((x.shape[0], latent_T - indices.shape[1]), 0, dtype=indices.dtype, device=indices.device)
            z = torch.cat([indices, pad], dim=1)
        else:
            z = indices
        z = z.long().clamp_(0, int(args.codebook_size) - 1)

        # reshape cond tokens if needed (match train_t2i behavior)
        if y.dim() >= 3:
            c_indices = y.reshape(y.shape[0], y.shape[-2], y.shape[-1])
        else:
            c_indices = y

        tf_inputs_for_sampling, eps_vec = perturb_teacher_forcing_inputs(z, args)
        with torch.no_grad():
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                logits_bt_vocab = tf_logits_t2i(gpt, c_indices, z, attn_mask, valid, tf_input_override=tf_inputs_for_sampling)

        # Sample num_generations sequences from same TF logits
        def sample_one_sequence_from_tf(logits_bt_vocab: torch.Tensor, temperature: float, top_k: int, top_p: float):
            B, T, V = logits_bt_vocab.shape
            logits = logits_bt_vocab / max(temperature, 1e-5)
            logits_flat = logits.reshape(B * T, V).contiguous()
            if top_k > 0 or top_p < 1.0:
                from autoregressive.models.generate import top_k_top_p_filtering
                logits_flat = top_k_top_p_filtering(logits_flat, top_k=top_k, top_p=top_p)
            log_probs_flat = torch.log_softmax(logits_flat, dim=-1)
            probs_flat = torch.softmax(logits_flat, dim=-1)
            idx_flat = torch.multinomial(probs_flat, num_samples=1)
            idx_T = idx_flat.view(B, T)
            gather_lp = log_probs_flat.gather(1, idx_flat).view(B, T)
            seq_log_prob = torch.sum(gather_lp, dim=-1)
            return idx_T, gather_lp, seq_log_prob

        all_token_indices = []
        all_token_log_probs = []
        all_seq_log_probs = []
        for _ in range(args.num_generations):
            idx_T, log_probs, seq_log_probs = sample_one_sequence_from_tf(
                logits_bt_vocab, args.temperature, args.top_k, args.top_p
            )
            all_token_indices.append(idx_T)
            all_token_log_probs.append(log_probs)
            all_seq_log_probs.append(seq_log_probs)
        all_token_indices = torch.stack(all_token_indices, dim=1).view(-1, latent_T).clamp(min=0, max=int(args.codebook_size) - 1)
        all_token_log_probs = torch.stack(all_token_log_probs, dim=1).view(-1, latent_T)
        all_seq_log_probs = torch.stack(all_seq_log_probs, dim=1).view(-1)

        # Decode predicted indices
        def process_in_chunks(process_fn, *inputs, chunk_size=8, **kwargs):
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
                return [torch.cat([r[i] for r in results], dim=0) for i in range(len(results[0]))]

        with torch.no_grad():
            def decode_chunk(chunk_indices):
                chunk_qzshape = [chunk_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
                return vq_model.decode_code(chunk_indices, chunk_qzshape)
            pred_imgs = process_in_chunks(decode_chunk, all_token_indices, chunk_size=8)

            # Optional embedding cosine similarity reward
            embedcos_loss_vec = None
            if getattr(args, "reward_use_embedcos", False):
                z_gt_pre = vq_model.quant_conv(vq_model.encoder(x))
                gt_vec = z_gt_pre.view(z_gt_pre.shape[0], -1)
                gt_vec_rep = repeat_tensor(gt_vec)

                def embedcos_chunk(chunk_indices, gt_vec_chunk):
                    chunk_shape = [chunk_indices.shape[0], args.codebook_embed_dim, latent_size, latent_size]
                    zq_chunk = vq_model.quantize.get_codebook_entry(chunk_indices, shape=chunk_shape, channel_first=True)
                    zq_vec = zq_chunk.view(chunk_indices.shape[0], -1)
                    cos_sim = F.cosine_similarity(zq_vec, gt_vec_chunk, dim=1)
                    return 1.0 - cos_sim

                embedcos_loss_vec = process_in_chunks(embedcos_chunk, all_token_indices, gt_vec_rep, chunk_size=16)

            # Reconstruction and perceptual loss
            gt_x = repeat_tensor(x)
            if args.reward_rec_type == "l1":
                rec_loss_vec = torch.mean(torch.abs(pred_imgs - gt_x), dim=(1, 2, 3))
            else:
                rec_loss_vec = torch.mean((pred_imgs - gt_x) ** 2, dim=(1, 2, 3))

            def lpips_chunk(pred_imgs_chunk, gt_x_chunk):
                perc_loss_chunk = lpips_model(pred_imgs_chunk, gt_x_chunk)
                if perc_loss_chunk.dim() > 1:
                    perc_loss_chunk = perc_loss_chunk.view(perc_loss_chunk.shape[0], -1).mean(dim=1)
                return perc_loss_chunk
            perc_loss_vec = process_in_chunks(lpips_chunk, pred_imgs, gt_x, chunk_size=8)

            if disc_model is not None and getattr(args, "reward_use_gan", False):
                def gan_chunk(pred_imgs_chunk):
                    logits_fake = disc_model(pred_imgs_chunk)
                    return -logits_fake.view(logits_fake.shape[0], -1).mean(dim=1)
                gan_loss_vec = process_in_chunks(gan_chunk, pred_imgs, chunk_size=8)
            else:
                gan_loss_vec = torch.zeros_like(rec_loss_vec)

            combined_loss = (
                args.reward_rec_weight * rec_loss_vec +
                args.reward_perceptual_weight * perc_loss_vec +
                args.reward_gan_weight * gan_loss_vec
            )
            if embedcos_loss_vec is not None:
                combined_loss = combined_loss + args.reward_embedcos_weight * embedcos_loss_vec
            rewards = -combined_loss

        # Prepare repeated tensors for training
        gt_z = repeat_tensor(z)
        gt_y = repeat_tensor(c_indices)
        rep_attn_mask = repeat_tensor(attn_mask)
        rep_valid = repeat_tensor(valid)
        tf_inputs_repeated = torch.repeat_interleave(tf_inputs_for_sampling, repeats=args.num_generations, dim=0)
        eps_repeated = torch.repeat_interleave(eps_vec, repeats=args.num_generations, dim=0)

        samples = {
            "class_ids": gt_y,
            "token_indices": all_token_indices,
            "gt_token_indices": gt_z,
            "token_log_probs": all_token_log_probs.detach(),
            "seq_log_probs": all_seq_log_probs.detach(),
            "rewards": rewards.detach(),
            "tf_teacher_inputs": tf_inputs_repeated.detach(),
            "tf_noise_eps": eps_repeated.detach(),
            "attn_mask": rep_attn_mask.detach(),
            "valid": (rep_valid.detach() if rep_valid is not None else None),
        }
        # Group-normalized advantages (per source item)
        n_groups = rewards.shape[0] // (args.num_generations)
        advantages = torch.zeros_like(rewards)
        for i in range(n_groups):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = rewards[start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages

        #################### TRAINING ####################
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
            sample = {k: (v[start:end] if isinstance(v, torch.Tensor) else v) for k, v in samples.items()}

            prev_training_mode = gpt.training
            if args.sample_model_mode == "eval" or args.sample_model_mode == "twice":
                gpt.eval()
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]:
                logits_bt_vocab = tf_logits_t2i(
                    gpt,
                    sample["class_ids"],
                    sample["gt_token_indices"],
                    sample["attn_mask"],
                    sample["valid"],
                    tf_input_override=sample["tf_teacher_inputs"],
                )
            if args.sample_model_mode == "twice" and prev_training_mode:
                gpt.train()

            logits = logits_bt_vocab / max(args.temperature, 1e-5)
            logits_flat = logits.reshape(-1, logits.shape[-1]).contiguous()
            log_probs = torch.log_softmax(logits_flat, dim=-1).view_as(logits_bt_vocab)
            new_per_token_logps = log_probs.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)

            # KL to base model (optional)
            per_token_kl = None
            if gpt_base is not None and args.kl_coef > 0:
                with torch.no_grad():
                    with {
                        "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                        "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                        "fp32": contextlib.nullcontext(),
                        "tf32": contextlib.nullcontext(),
                    }[args.mixed_precision]:
                        base_logits_bt_vocab = tf_logits_t2i(
                            gpt_base,
                            sample["class_ids"],
                            sample["gt_token_indices"],
                            sample["attn_mask"],
                            sample["valid"],
                            tf_input_override=sample["tf_teacher_inputs"],
                        )
                temp = max(args.temperature, 1e-5)
                cur_logits_for_kl = (logits_bt_vocab / temp).to(torch.float32)
                base_logits_for_kl = (base_logits_bt_vocab / temp).to(torch.float32)
                cur_logps_nofilter = torch.log_softmax(cur_logits_for_kl, dim=-1)
                base_logps_nofilter = torch.log_softmax(base_logits_for_kl, dim=-1)
                cur_token_logps_for_kl = cur_logps_nofilter.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                ref_token_logps_for_kl = base_logps_nofilter.gather(-1, sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                per_token_kl = torch.exp(ref_token_logps_for_kl - cur_token_logps_for_kl) - (ref_token_logps_for_kl - cur_token_logps_for_kl) - 1

            advantages = torch.clamp(sample["advantages"], -args.adv_clip_max, args.adv_clip_max)
            ratio = torch.exp(new_per_token_logps - sample["token_log_probs"])  # [N, T]
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)

            per_token_unclipped_loss = -advantages.unsqueeze(1) * ratio
            per_token_clipped_loss = -advantages.unsqueeze(1) * clipped_ratio
            per_token_loss = torch.maximum(per_token_unclipped_loss, per_token_clipped_loss)

            if per_token_kl is not None and args.use_kl_loss:
                per_token_loss = per_token_loss + args.kl_coef * per_token_kl

            token_mask = None
            if getattr(args, "use_token_dropout", False) and getattr(args, "token_dropout_ratio", 0.0) > 0.0:
                drop_ratio = max(0.0, min(1.0, float(args.token_dropout_ratio)))
                keep_prob = 1.0 - drop_ratio
                token_mask = torch.bernoulli(torch.full_like(per_token_loss, keep_prob)).to(per_token_loss.dtype)
                keep_counts = token_mask.sum(dim=1)
                zero_keep = keep_counts == 0
                if zero_keep.any():
                    token_mask[zero_keep, 0] = 1.0
                    keep_counts = token_mask.sum(dim=1)
                seq_loss = (per_token_loss * token_mask).sum(dim=-1) / keep_counts.clamp_min(1)
            else:
                seq_loss = per_token_loss.mean(dim=-1)
            loss = seq_loss.mean()

            if args.sample_model_mode == "twice":
                with {
                    "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                    "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[args.mixed_precision]:
                    logits_bt_vocab_ce = tf_logits_t2i(
                        gpt,
                        sample["class_ids"],
                        sample["gt_token_indices"],
                        sample["attn_mask"],
                        sample["valid"],
                        tf_input_override=sample["tf_teacher_inputs"],
                    )
            else:
                logits_bt_vocab_ce = logits_bt_vocab

            aux_ce_unweighted = torch.tensor(0.0, device=logits_bt_vocab_ce.device)
            if getattr(args, "aux_ce_weight", 0.0) and args.aux_ce_weight > 0:
                ce_val = F.cross_entropy(
                    logits_bt_vocab_ce.reshape(-1, logits_bt_vocab_ce.shape[-1]).to(torch.float32),
                    sample["gt_token_indices"].reshape(-1),
                    reduction='mean'
                )
                aux_ce_unweighted = ce_val
                loss = loss + args.aux_ce_weight * ce_val

            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            step_idx += 1
            if (step_idx) % args.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Optional: compute clip ratio metric
                clip_mask = (torch.abs(ratio - 1.0) > args.clip_range)
                clip_ratio = clip_mask.float().mean().item()

                if global_rank == 0:
                    print({
                        "loss": float(loss.item() * args.gradient_accumulation_steps),
                        "seq_loss_mean": float(seq_loss.mean().item()),
                        "ratio_mean": float(ratio.mean().item()),
                        "clipped_ratio_mean": float(clipped_ratio.mean().item()),
                        "grad_norm": float(grad_norm),
                        "clip_ratio": float(clip_ratio),
                        "kl_base":
                        "aux_ce_loss": float(aux_ce_unweighted.detach().item()),
                        "aux_ce_weight": float(getattr(args, "aux_ce_weight", 0.0)),
                    })

                # Periodic checkpoint saving
                if args.ckpt_every and args.ckpt_every > 0 and ((global_opt_step + 1) % args.ckpt_every == 0):
                    ckpt_dir = os.path.join(args.ckpt_dir, f"t2i-grpo-lr-{args.lr}-mse-{args.reward_rec_weight}-lpips-{args.reward_perceptual_weight}-gan-{args.reward_use_gan}-kl-{args.use_kl_loss}-ce-{args.aux_ce_weight}-noisy-{args.use_token_noise}-{args.token_noise_prob}-clip-{args.clip_range}-{global_opt_step:07d}")
                    ensure_dir(ckpt_dir)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Save on rank 0 only
                    if global_rank == 0:
                        model_state = gpt.module.state_dict()
                        torch.save(model_state, os.path.join(ckpt_dir, "consolidated.pth"))
                        print(f"Saved checkpoint to {ckpt_dir}")
                    dist.barrier()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                global_opt_step += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if global_rank == 0:
            print(f"Finished epoch {epoch}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


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
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
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

    # dataset (builtin t2i)
    parser.add_argument("--dataset", type=str, default='t2i')
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--t5-feat-path", type=str, required=True)
    parser.add_argument("--short-t5-feat-path", type=str, default=None)
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
    parser.add_argument("--sample-model-mode", type=str, choices=["train", "eval", "twice"], default="eval")

    # reward config
    parser.add_argument("--reward-rec-type", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--reward-rec-weight", type=float, default=1.0)
    parser.add_argument("--reward-perceptual-weight", type=float, default=1.0)
    parser.add_argument("--reward-use-gan", action='store_true')
    parser.add_argument("--reward-gan-weight", type=float, default=1.0)
    parser.add_argument("--reward-disc-dim", type=int, default=64)
    parser.add_argument("--reward-disc-num-layers", type=int, default=3)
    parser.add_argument("--reward-disc-ckpt", type=str, default=None)
    parser.add_argument("--reward-use-embedcos", action='store_true')
    parser.add_argument("--reward-embedcos-weight", type=float, default=0.0)
    parser.add_argument("--aux-ce-weight", type=float, default=0.0, help="Weight for auxiliary CE(gt) loss in training loop (0 disables)")

    # KL regularization
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--use-kl-loss", action='store_true')
    parser.add_argument("--kl-base-ckpt", type=str, default=None)

    # checkpointing
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")

    # noisy TF inputs and token dropout during loss
    parser.add_argument("--use-token-noise", action='store_true')
    parser.add_argument("--token-noise-prob", type=float, default=0.5)
    parser.add_argument("--use-token-dropout", action='store_true')
    parser.add_argument("--token-dropout-ratio", type=float, default=0.5)
    
    # data start and end
    parser.add_argument("--data-start", type=int, default=0)
    parser.add_argument("--data-end", type=int, default=10)

    args = parser.parse_args()
    main(args)