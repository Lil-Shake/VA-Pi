import os
import time
import argparse

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create and load VQ model
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

    # create and load GPT model
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
    gpt_model.eval()
    del gpt_ckpt
    print("gpt model is loaded")

    if args.compile:
        print("compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True,
        )
    else:
        print("no need to compile model in demo")

    # Default class labels to condition the model with
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    if args.class_labels is not None and len(args.class_labels) > 0:
        class_labels = [int(x) for x in args.class_labels.split(',')]

    ensure_dir(args.output_dir)

    # For each class, generate 6 images: 1 argmax + 5 sampled
    for label in class_labels:
        print(f"Generating images for class {label} ...")
        t_start = time.time()

        # shapes and condition tensors
        qzshape = [6, args.codebook_embed_dim, latent_size, latent_size]

        # 1) Greedy (argmax) generation
        cond_argmax = torch.tensor([label], device=device)
        idx_argmax = generate(
            gpt_model,
            cond_argmax,
            latent_size ** 2,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            sample_logits=False,
        )  # [1, T]

        # 2) Stochastic sampling generation (5 samples)
        cond_sample = torch.full((5,), fill_value=label, device=device, dtype=torch.long)
        idx_sample = generate(
            gpt_model,
            cond_sample,
            latent_size ** 2,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=0,
            top_p=1.0,
            sample_logits=True,
        )  # [5, T]

        index_sample = torch.cat([idx_argmax, idx_sample], dim=0)  # [6, T]

        # Decode tokens to images (value range expected to be [-1, 1])
        images = vq_model.decode_code(index_sample, qzshape)

        # Save a single stitched image per class
        out_path = os.path.join(args.output_dir, f"class_{label}.png")
        save_image(images, out_path, nrow=6, normalize=True, value_range=(-1, 1))

        print(f"Saved 6-image grid for class {label} -> {out_path} (time: {time.time() - t_start:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--output-dir", type=str, default="samples_c2i_six", help="directory to save 6-image grids per class")
    parser.add_argument("--class-labels", type=str, default=None, help="comma-separated class ids; default uses 8 predefined labels")
    args = parser.parse_args()
    main(args)