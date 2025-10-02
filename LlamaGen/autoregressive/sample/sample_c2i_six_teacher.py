import os
import time
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import top_k_top_p_filtering
from dataset.build import build_dataset


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sample_from_logits_full_sequence(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, num_samples: int) -> torch.Tensor:
    """
    logits: [B, T, V]
    Return: [num_samples, B, T] sampled token indices for each position.
    """
    B, T, V = logits.shape
    flat = logits.reshape(B * T, V) / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        flat = top_k_top_p_filtering(flat, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(flat, dim=-1)
    # multinomial can draw multiple samples per row
    idx = torch.multinomial(probs, num_samples=num_samples)  # [B*T, num_samples]
    idx = idx.view(B, T, num_samples).permute(2, 0, 1).contiguous()  # [num_samples, B, T]
    return idx


@torch.no_grad()
def main(args):
    # Setup
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    vq_ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(vq_ckpt["model"])
    del vq_ckpt
    print("image tokenizer is loaded")

    # Load GPT model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    gpt_ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:
        model_weight = gpt_ckpt
    elif "model" in gpt_ckpt:  # ddp
        model_weight = gpt_ckpt["model"]
    elif "module" in gpt_ckpt:  # deepspeed
        model_weight = gpt_ckpt["module"]
    elif "state_dict" in gpt_ckpt:
        model_weight = gpt_ckpt["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    gpt_model.load_state_dict(model_weight, strict=False)
    del gpt_ckpt
    print("gpt model is loaded")

    # IMPORTANT: teacher forcing forward path expects training mode for positional embedding handling
    gpt_model.train()

    # Build dataset and dataloader
    ensure_dir(args.output_dir)
    args.dataset = 'imagenet_code' if args.dataset is None else args.dataset
    dataset = build_dataset(args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Prepare class labels list
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    if args.class_labels is not None and len(args.class_labels) > 0:
        class_labels = [int(x) for x in args.class_labels.split(',')]
    remaining = set(class_labels)

    # Collect one sample per requested class
    class_to_sample = {}
    for x, y in loader:
        y = y.reshape(-1)
        label = int(y.item())
        if label in remaining:
            class_to_sample[label] = (x, y)
            remaining.remove(label)
            if len(remaining) == 0:
                break

    if len(class_to_sample) == 0:
        print("No matching classes found in dataset for the requested class labels.")
        return

    for label in class_labels:
        if label not in class_to_sample:
            print(f"Skip class {label}: not found in dataset.")
            continue

        x, y = class_to_sample[label]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Match training shapes as in train_c2i_fsdp.py
        z_indices = x.squeeze(dim=1)  # attempt to drop augmentation dimension
        if z_indices.dim() == 2:
            z_indices = z_indices.unsqueeze(1)  # [B, 1, T]

        # Forward teacher forcing: feed ground-truth prefixes, get per-position logits
        with torch.autocast(device_type='cuda', dtype=precision) if device == 'cuda' and precision != torch.float32 else torch.cuda.amp.autocast(enabled=False):
            logits, _ = gpt_model(cond_idx=y, idx=z_indices[:, :, :-1], targets=None)

        # logits: [B, T, V], T should equal latent_size**2
        B, T, V = logits.shape

        # 1) Greedy (argmax) sequence
        pred_argmax = torch.argmax(logits, dim=-1)  # [B, T]

        # 2) 5 stochastic sequences sampled independently for each position
        sampled_all = sample_from_logits_full_sequence(
            logits,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=5,
        )  # [5, B, T]

        # Stack into 6 sequences: first argmax then 5 samples
        seqs = torch.cat([
            pred_argmax.unsqueeze(0),  # [1, B, T]
            sampled_all,
        ], dim=0)  # [6, B, T]

        # reshape to [6, T]
        index_sample = seqs[:, 0, :].contiguous()

        # Decode tokens to images (value range [-1, 1])
        qzshape = [index_sample.shape[0], args.codebook_embed_dim, latent_size, latent_size]
        images = vq_model.decode_code(index_sample, qzshape)

        # Save a single stitched image per class
        out_path = os.path.join(args.output_dir, f"class_{label}_tf.png")
        save_image(images, out_path, nrow=6, normalize=True, value_range=(-1, 1))
        print(f"Saved TF 6-image grid for class {label} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, required=True)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--output-dir", type=str, default="samples_c2i_six_tf", help="directory to save TF 6-image grids per class")
    parser.add_argument("--class-labels", type=str, default=None, help="comma-separated class ids; default uses 8 predefined labels")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name; default uses imagenet_code")
    parser.add_argument("--code-path", type=str, required=True, help="root path for precomputed codes and labels")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)


