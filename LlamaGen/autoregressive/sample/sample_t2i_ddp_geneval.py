import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import time

import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_prompts_from_jsonl(jsonl_path: str):
    dataset = []  # list of (idx, prompt, meta)
    with open(jsonl_path, 'r') as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get('prompt') or obj.get('text') or obj.get('Prompt')
            if isinstance(prompt, str) and len(prompt) > 0:
                dataset.append((idx, prompt, obj))
                idx += 1
    return dataset
def _load_state_dict_flexible(ckpt_path: str):
    """
    Load a state_dict from various formats:
    - Directory with HuggingFace-style files (model.safetensors / pytorch_model.bin, with or without sharded index)
    - Single .safetensors file
    - Single .bin/.pt/.pth file saved via torch.save
    - torch.save dicts wrapping state under keys like ['model', 'module', 'state_dict']
    Returns a flat {param_name: tensor} state_dict on CPU.
    """
    def _clean_wrapped(obj):
        if isinstance(obj, dict):
            for k in ["model", "module", "state_dict"]:
                if k in obj and isinstance(obj[k], dict):
                    return obj[k]
        return obj

    def _maybe_strip_module_prefix(sd: dict):
        if not isinstance(sd, dict):
            return sd
        new_sd = {}
        for k, v in sd.items():
            if isinstance(k, str) and k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v
        return new_sd

    if os.path.isdir(ckpt_path):
        # Try sharded index first
        index_candidates = [
            "pytorch_model.bin.index.json",
            "pytorch_model.safetensors.index.json",
            "model.safetensors.index.json",
        ]
        for name in index_candidates:
            idx_path = os.path.join(ckpt_path, name)
            if os.path.isfile(idx_path):
                with open(idx_path, "r") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map") or {}
                shard_files = sorted(set(weight_map.values()))
                merged = {}
                for shard in shard_files:
                    shard_path = os.path.join(ckpt_path, shard)
                    if shard_path.endswith(".safetensors"):
                        try:
                            from safetensors.torch import load_file as safe_load_file
                        except Exception as e:
                            raise RuntimeError("safetensors is required to load .safetensors checkpoints. Try: pip install safetensors") from e
                        part = safe_load_file(shard_path, device="cpu")
                    else:
                        part = torch.load(shard_path, map_location="cpu")
                        part = _clean_wrapped(part)
                    merged.update(part)
                return _maybe_strip_module_prefix(merged)

        # Fallback to common single-file names inside directory
        single_candidates = [
            "model.safetensors",
            "pytorch_model.safetensors",
            "pytorch_model.bin",
            "model.bin",
            "model.pt",
            "pytorch_model.pt",
        ]
        for name in single_candidates:
            p = os.path.join(ckpt_path, name)
            if os.path.isfile(p):
                return _load_state_dict_flexible(p)
        raise FileNotFoundError(f"No recognizable checkpoint files found in directory: {ckpt_path}")

    # File path
    lower = ckpt_path.lower()
    if lower.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except Exception as e:
            raise RuntimeError("safetensors is required to load .safetensors checkpoints. Try: pip install safetensors") from e
        sd = safe_load_file(ckpt_path, device="cpu")
        return _maybe_strip_module_prefix(sd)
    else:
        obj = torch.load(ckpt_path, map_location="cpu")
        obj = _clean_wrapped(obj)
        if isinstance(obj, dict):
            return _maybe_strip_module_prefix(obj)
        raise ValueError(f"Unsupported checkpoint format at path: {ckpt_path}")

def _remap_state_dict_keys_for_model(model, sd: dict, verbose: bool = True):
    """
    Try several common prefix variants to maximize matched keys against model.state_dict().
    This mitigates issues where checkpoints save under wrappers like 'module.', 'model.', 'gpt_model.', etc.
    Returns the best-matching state_dict and a small report.
    """
    model_keys = set(model.state_dict().keys())
    def strip_prefix(d: dict, prefix: str):
        if not prefix:
            return d
        plen = len(prefix)
        return { (k[plen:] if k.startswith(prefix) else k): v for k, v in d.items() }
    def strip_first_segment(d: dict):
        out = {}
        for k, v in d.items():
            if '.' in k:
                out[k.split('.', 1)[1]] = v
            else:
                out[k] = v
        return out
    candidates = []
    variants = [
        ("as-is", sd),
        ("no module.", strip_prefix(sd, "module.")),
        ("no model.", strip_prefix(sd, "model.")),
        ("no gpt_model.", strip_prefix(sd, "gpt_model.")),
        ("strip first segment", strip_first_segment(sd)),
    ]
    for name, cand in variants:
        matched = len([k for k in cand.keys() if k in model_keys])
        candidates.append((matched, name, cand))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_matched, best_name, best_sd = candidates[0]
    total = len(model_keys)
    if verbose:
        print(f"[state_dict] best match: '{best_name}' matched {best_matched}/{total} model params")
        missing = total - best_matched
        if missing > 0:
            print(f"[state_dict] Warning: {missing} parameters may remain at default init; results could degrade")
    return best_sd, best_matched, total

def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % max(1, torch.cuda.device_count()))))
    device = local_rank
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    assert args.vq_ckpt is not None and len(str(args.vq_ckpt)) > 0, "Please provide --vq-ckpt (file or directory)."
    vq_state_dict = _load_state_dict_flexible(args.vq_ckpt)
    vq_state_dict, vq_matched, vq_total = _remap_state_dict_keys_for_model(vq_model, vq_state_dict, verbose=(rank==0))
    load_res = vq_model.load_state_dict(vq_state_dict, strict=False)
    if rank == 0:
        print(f"[VQ] loaded with missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}")
    del vq_state_dict
    if rank == 0:
        print(f"image tokenizer is loaded")

    # create and load GPT model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    assert args.gpt_ckpt is not None and len(str(args.gpt_ckpt)) > 0, "Please provide --gpt-ckpt (file or directory)."
    model_weight = _load_state_dict_flexible(args.gpt_ckpt)
    model_weight, gpt_matched, gpt_total = _remap_state_dict_keys_for_model(gpt_model, model_weight, verbose=(rank==0))
    load_res = gpt_model.load_state_dict(model_weight, strict=False)
    if rank == 0:
        print(f"[GPT] loaded with missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}")
    gpt_model.eval()
    del model_weight
    if rank == 0:
        print(f"gpt model is loaded")

    if args.compile:
        if rank == 0:
            print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        )
    else:
        if rank == 0:
            print(f"no need to compile model in demo")

    # create t5 model
    assert os.path.exists(args.t5_path)
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    if rank == 0:
        print(f"t5 model is loaded")

    # output root
    out_root = args.out_root
    if rank == 0:
        ensure_dir(out_root)

    # build prompt dataset and shard across ranks
    dataset = load_prompts_from_jsonl(args.jsonl)
    world_size = dist.get_world_size()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.prompt_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda x: x,
    )

    # Only show a single progress bar for rank == 0 on its shard
    from tqdm import tqdm
    loop_iter = loader
    if rank == 0:
        loop_iter = tqdm(loader, desc="Generating prompts", ncols=88)

    for batch in loop_iter:
        # batch is a list of (idx, prompt, meta) tuples
        for idx, prompt, meta in batch:
            if rank == 0:
                short_prompt = (prompt[:60] + "...") if len(prompt) > 60 else prompt
                print(f"[rank{rank}] Start idx={idx}, prompt='{short_prompt}'", flush=True)
            local_count = args.num_samples  # per-prompt images this rank will generate

            # chunk by per-gpu sample batch size
            per_gpu_bs = args.per_proc_batch_size
            num_iters = (local_count + per_gpu_bs - 1) // per_gpu_bs
            save_dir = os.path.join(out_root, f"{idx:05d}", "samples")
            ensure_dir(save_dir)

            saved_so_far = 0
            for _ in range(num_iters):
                cur_bs = min(per_gpu_bs, local_count - saved_so_far)
                prompt_batch = [prompt] * cur_bs

                # text embeddings
                t0 = time.time()
                caption_embs, emb_masks = t5_model.get_text_embeddings(prompt_batch)
                if rank == 0:
                    print(f"[rank{rank}] T5 embeddings done in {time.time()-t0:.2f}s; shape={tuple(caption_embs.shape)}", flush=True)

                # align embeddings and masks by rotating pads to the left
                if not args.no_left_padding:
                    seq_len = emb_masks.size(1)
                    rolled_embs = []
                    rolled_masks = []
                    for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                        valid_num = int(emb_mask.sum().item())
                        shift = seq_len - valid_num
                        rolled_embs.append(torch.roll(caption_emb, shifts=shift, dims=0))
                        rolled_masks.append(torch.roll(emb_mask, shifts=shift, dims=0))
                    new_caption_embs = torch.stack(rolled_embs, dim=0)
                    new_emb_masks = torch.stack(rolled_masks, dim=0)
                else:
                    new_caption_embs, new_emb_masks = caption_embs, emb_masks

                # enforce max condition length
                max_len = args.cls_token_num
                if new_caption_embs.size(1) > max_len:
                    new_caption_embs = new_caption_embs[:, -max_len:, :]
                    new_emb_masks = new_emb_masks[:, -max_len:]

                c_indices = new_caption_embs * new_emb_masks[:, :, None]
                c_emb_masks = new_emb_masks

                qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
                if rank == 0:
                    print(f"[rank{rank}] Generate start: bs={len(c_indices)}, T={new_caption_embs.size(1)}, steps={latent_size ** 2}", flush=True)
                t1 = time.time()
                index_sample = generate(
                    gpt_model,
                    c_indices,
                    latent_size ** 2,
                    c_emb_masks,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=True,
                )
                if rank == 0:
                    print(f"[rank{rank}] Generate done in {time.time()-t1:.2f}s", flush=True)

                samples = vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                # save samples for this rank
                for i, sample in enumerate(samples):
                    global_idx = saved_so_far + i
                    Image.fromarray(sample).save(os.path.join(save_dir, f"{global_idx:04d}.png"))
                saved_so_far += len(samples)
                if rank == 0:
                    print(f"[rank{rank}] Saved {saved_so_far}/{local_count} images for idx={idx}", flush=True)

            # write metadata.jsonl for this prompt (only this rank owns it)
            meta_path = os.path.join(out_root, f"{idx:05d}", "metadata.jsonl")
            with open(meta_path, "w") as f:
                f.write(json.dumps(meta) + "\n")
            if rank == 0:
                print(f"[rank{rank}] Finished idx={idx}", flush=True)

    if rank == 0:
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input / evaluation
    parser.add_argument("--jsonl", type=str)
    parser.add_argument("--num-samples", type=int, default=4, help="number of images to generate per prompt (per GPU shard)")
    parser.add_argument("--out-root", type=str, default="geneval_samples", help="output root directory; per-prompt folders will be created here")

    # text encoder
    parser.add_argument("--t5-path", type=str, default='pretrained_models/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)

    # gpt
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i", help="class->image or text->image")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)

    # vq
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")

    # image / sampling
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--prompt-batch-size", type=int, default=8, help="number of prompts to process per GPU per step")
    parser.add_argument("--per-proc-batch-size", type=int, default=8, help="number of images to generate per forward pass for one prompt")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")

    args = parser.parse_args()
    main(args)