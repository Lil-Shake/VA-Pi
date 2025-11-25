import os
import datetime
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
import math
import time
import json
import io
import tarfile
import random
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from tqdm import tqdm

from lpips import LPIPS  # pip install lpips
import wandb  # optional

from transformers import AutoModelForCausalLM
import deepspeed
from contextlib import contextmanager

from janus.models import MultiModalityCausalLM, VLChatProcessor


logger = logging.getLogger("train_t2i_grpo_janus")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------- Utilities ----------------------------
def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


@contextmanager
def zero3_gather_params(module):
    """
    Context manager that gathers ZeRO-3 partitioned params for the given module
    so we can safely call submodule forwards outside of DeepSpeedEngine.forward.
    No-op if DeepSpeed/ZeRO-3 are not active.
    """
    try:
        try:
            from deepspeed.zero import GatheredParameters  # older DS versions
        except Exception:
            from deepspeed.runtime.zero.partition_parameters import GatheredParameters  # newer DS versions
        params = list(module.parameters())
        if len(params) == 0:
            yield
            return
        with GatheredParameters(params, modifier_rank=None):
            yield
    except Exception:
        # Not in ZeRO-3 or deepspeed not present
        yield


def init_distributed_mode(args) -> None:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    # Ensure env vars exist for DeepSpeed even when launched by srun
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    timeout_sec = int(os.environ.get('TORCH_DIST_TIMEOUT', '600'))
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=timeout_sec),
    )
    dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def repeat_tensor(tensor: Optional[torch.Tensor], repeats: int) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return torch.repeat_interleave(tensor, repeats, dim=0)


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    top_k = min(max(top_k, 0), logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits


def process_in_chunks(process_fn, *inputs, chunk_size: int = 8, **kwargs):
    total = inputs[0].shape[0]
    outputs = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_inputs = [x[start:end] for x in inputs]
        outputs.append(process_fn(*chunk_inputs, **kwargs))
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    else:
        return [torch.cat([o[i] for o in outputs], dim=0) for i in range(len(outputs[0]))]


# ---------------------------- Dataset ----------------------------
class LaionCocoDataset(Dataset):
    """
    Simple TSV dataset: each line is "<absolute_image_path>\t<caption>".
    This intentionally avoids extra dependencies. Prepare your LAION-COCO list in this format.
    """
    def __init__(self, tsv_path: str, image_size: int = 384):
        super().__init__()
        self.items: List[Tuple[str, str]] = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                img_path, caption = parts[0], '\t'.join(parts[1:])
                self.items.append((img_path, caption))

        # transforms for VQ encoder: center-crop/resize to image_size and map to [-1, 1]
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, caption = self.items[idx]
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            img_tensor = self.transform(im)
        return img_tensor, caption


# ---------------------------- LAION/COCO Tar Dataset (image + caption) ----------------------------
class LaionCocoTarCaptionDataset(Dataset):
    """
    WebDataset-style tar loader that pairs image files with per-image JSON sidecars and extracts a caption.

    Each sample is (image_tensor, caption_str).

    - Accepts the same input spec styles as LlamaGen's Text2ImgTarDataset (comma lists, globs, brace patterns),
      but implemented locally to avoid extra dependencies.
    - Pairs \*.json to image by shared filename stem. If multiple are present, one-to-one by stem.
    - Caption extraction tries common keys in order: 'caption', 'text', 'TEXT', 'alt', 'meta.caption'.
    - If caption missing, falls back to empty string.
    """

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    def __init__(self, data_path: str, image_size: int = 384, data_start: Optional[int] = None, data_end: Optional[int] = None):
        super().__init__()
        self.input_spec = data_path
        self.image_size = image_size
        self.data_start = data_start
        self.data_end = data_end

        # transforms: resize/centercrop to image_size and map to [-1, 1]
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Build index: list of (tar_path, image_member_name, json_member_name)
        self.samples: List[Tuple[str, str, Optional[str]]] = self._build_index(self.input_spec)
        # Cache opened tar files per worker
        self._tar_cache: dict[str, tarfile.TarFile] = {}

    def _expand_specs(self, spec_str: str) -> List[str]:
        import glob as _glob
        paths: List[str] = []
        for s in [s.strip() for s in spec_str.split(',') if s.strip()]:
            expanded: List[str] = []
            # Best-effort brace expansion: {00000..00127}.tar
            if '{' in s and '}' in s:
                try:
                    # naive brace expansion for numeric ranges
                    prefix, rest = s.split('{', 1)
                    range_part, suffix = rest.split('}', 1)
                    start_str, end_str = range_part.split('..')
                    start, end = int(start_str), int(end_str)
                    width = max(len(start_str), len(end_str))
                    for i in range(start, end + 1):
                        expanded.append(f"{prefix}{i:0{width}d}{suffix}")
                except Exception:
                    expanded = []
            if not expanded:
                expanded = _glob.glob(s)
            if expanded:
                paths.extend(expanded)
            else:
                paths.append(s)
        return paths

    def _build_index(self, input_spec: str) -> List[Tuple[str, str, Optional[str]]]:
        samples: List[Tuple[str, str, Optional[str]]] = []
        paths = self._expand_specs(input_spec)

        def index_single_tar(tar_path: str):
            local: List[Tuple[str, str, Optional[str]]] = []
            try:
                with tarfile.open(tar_path, mode="r:*") as tf:
                    members = [m for m in tf.getmembers() if m.isfile()]
                    # Collect images and json by filename stem
                    images_by_stem: dict[str, str] = {}
                    json_by_stem: dict[str, str] = {}
                    for m in members:
                        lower = m.name.lower()
                        base = os.path.basename(m.name)
                        stem, ext = os.path.splitext(base)
                        if lower.endswith(self.IMG_EXTENSIONS):
                            images_by_stem[stem] = m.name
                        elif lower.endswith('.json'):
                            json_by_stem[stem] = m.name
                    # Pair by stem; prefer pairs with JSON, but still include images without JSON (caption="")
                    paired = 0
                    for stem, img_member in images_by_stem.items():
                        json_member = json_by_stem.get(stem)
                        local.append((tar_path, img_member, json_member))
                        if json_member is not None:
                            paired += 1
                print(f"[TarIndex] tar={tar_path} imgs={len(local)} paired_json={paired}", flush=True)
            except Exception as e:
                print(f"[TarIndex][WARN] Failed to index tar={tar_path}: {e}", flush=True)
                return
            samples.extend(local)

        for p in paths:
            if os.path.isdir(p):
                for dirpath, _, filenames in os.walk(p):
                    shard_files = sorted([f for f in filenames if f.endswith('.tar')])
                    if isinstance(self.data_start, int) and isinstance(self.data_end, int):
                        shard_files = shard_files[self.data_start:self.data_end + 1]
                    for fname in shard_files:
                        index_single_tar(os.path.join(dirpath, fname))
            elif os.path.isfile(p) and p.endswith('.tar'):
                index_single_tar(p)
            else:
                continue

        print(f"[TarIndex] total_samples={len(samples)} unique_shards={len(set([s[0] for s in samples]))}", flush=True)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _open_tar_cached(self, tar_path: str) -> tarfile.TarFile:
        tf = self._tar_cache.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, mode="r:*")
            self._tar_cache[tar_path] = tf
        return tf

    @staticmethod
    def _extract_caption_from_json_bytes(data: bytes) -> str:
        try:
            obj = json.loads(data.decode('utf-8', errors='ignore'))
        except Exception:
            return ""
        # Try common keys
        for k in [
            'caption', 'text', 'TEXT', 'alt',
        ]:
            val = obj.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Nested possibilities
        try:
            meta = obj.get('meta') or obj.get('metadata') or {}
            if isinstance(meta, dict):
                val = meta.get('caption') or meta.get('text')
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
        # Some datasets store a list under 'sentences'
        try:
            sents = obj.get('sentences')
            if isinstance(sents, list) and sents:
                cand = sents[0]
                if isinstance(cand, dict):
                    raw = cand.get('raw') or cand.get('caption') or cand.get('text')
                    if isinstance(raw, str) and raw.strip():
                        return raw.strip()
                if isinstance(cand, str) and cand.strip():
                    return cand.strip()
        except Exception:
            pass
        return ""

    def __getitem__(self, index: int):
        tar_path, img_member, json_member = self.samples[index]
        tf = self._open_tar_cached(tar_path)

        # Load image
        try:
            fobj = tf.extractfile(img_member)
            assert fobj is not None, f"member not found: {img_member}"
            with fobj:
                img_bytes = fobj.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(f"[Load][WARN] idx={index} tar={tar_path} member={img_member} error={e}", flush=True)
            img = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(img)

        # Load caption if json present
        caption: str = ""
        if json_member is not None:
            try:
                fobj = tf.extractfile(json_member)
                if fobj is not None:
                    with fobj:
                        jbytes = fobj.read()
                    caption = self._extract_caption_from_json_bytes(jbytes)
            except Exception as e:
                print(f"[Load][WARN] idx={index} failed to read caption json={json_member}: {e}", flush=True)

        return img_tensor, caption


# ---------------------------- HF Dataset (saved to disk) ----------------------------
class HFDiskImageCaptionDataset(Dataset):
    """
    Load image+caption data from HuggingFace datasets saved to disk (Dataset.save_to_disk).

    - Expects one or more shard directories under a root directory (e.g., part_00, part_01, ...),
      or a single saved dataset directory.
    - Decodes images via `datasets.Image` and returns (image_tensor, caption_str).
    - Caption and image column names are configurable.
    """

    def __init__(self, root_dir: str, image_size: int = 384, image_key: str = "image", caption_key: str = "text", data_start: Optional[int] = None, data_end: Optional[int] = None):
        super().__init__()
        self.root_dir = root_dir
        self.image_key = image_key
        self.caption_key = caption_key

        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Lazily import datasets to avoid hard dependency when unused
        try:
            from datasets import load_from_disk, concatenate_datasets, Image as HFImage
        except Exception as e:
            raise ImportError("Please `pip install datasets` to use HF dataset options.") from e

        # Determine shard directories
        shard_dirs: List[str] = []
        if os.path.isdir(root_dir) and os.path.exists(os.path.join(root_dir, "dataset_info.json")):
            shard_dirs = [root_dir]
        else:
            subs = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
            # Prefer part_* style directories if present
            part_dirs = [d for d in subs if d.startswith("part_")]
            shard_dirs = part_dirs if len(part_dirs) > 0 else subs
            if isinstance(data_start, int) and isinstance(data_end, int):
                shard_dirs = shard_dirs[data_start:data_end + 1]
            shard_dirs = [os.path.join(root_dir, d) for d in shard_dirs]

        if len(shard_dirs) == 0:
            raise ValueError(f"No HF dataset shards found under: {root_dir}")

        # Load and concatenate shards
        loaded = []
        for d in shard_dirs:
            try:
                ds = load_from_disk(d)
                # Ensure image column is Image feature for proper decoding
                if self.image_key in ds.column_names:
                    try:
                        ds = ds.cast_column(self.image_key, HFImage())
                    except Exception:
                        pass
                loaded.append(ds)
            except Exception as e:
                print(f"[HFLoader][WARN] Failed to load shard: {d} ({e})", flush=True)

        if len(loaded) == 0:
            raise ValueError(f"Failed to load any shards from: {root_dir}")

        if len(loaded) == 1:
            self.ds = loaded[0].with_format("python")
        else:
            self.ds = concatenate_datasets(loaded).with_format("python")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        from PIL import Image as PILImage
        rec = self.ds[int(idx)]
        img_obj = rec.get(self.image_key)
        caption = rec.get(self.caption_key, "")

        # Decode image
        if isinstance(img_obj, PILImage.Image):
            img = img_obj
        elif isinstance(img_obj, dict):
            path = img_obj.get("path")
            if isinstance(path, str) and os.path.exists(path):
                img = PILImage.open(path).convert("RGB")
            else:
                by = img_obj.get("bytes")
                if by is None:
                    img = PILImage.new('RGB', (self.transform.transforms[0].size, self.transform.transforms[0].size), (0, 0, 0))
                else:
                    img = PILImage.open(io.BytesIO(by)).convert("RGB")
        else:
            # Fallback: try to open if it's a file path, else black image
            try:
                if isinstance(img_obj, str) and os.path.exists(img_obj):
                    img = PILImage.open(img_obj).convert("RGB")
                else:
                    img = PILImage.new('RGB', (self.transform.transforms[0].size, self.transform.transforms[0].size), (0, 0, 0))
            except Exception:
                img = PILImage.new('RGB', (self.transform.transforms[0].size, self.transform.transforms[0].size), (0, 0, 0))

        img_tensor = self.transform(img)
        if not isinstance(caption, str):
            caption = "" if caption is None else str(caption)
        return img_tensor, caption


# ---------------------------- HF Streaming Utility ----------------------------
def stream_hf_dataset_to_disk(
    dataset_name: str,
    split: str,
    n_samples: int,
    chunk_size: int,
    output_dir: str,
    shuffle_buffer_size: int = 10000,
    seed: int = 0,
) -> int:
    """
    Stream a HuggingFace dataset and save in chunked shards to disk.
    Returns total number of saved samples.
    """
    ensure_dir(output_dir)
    try:
        from datasets import load_dataset, Dataset as HFDataset
    except Exception as e:
        raise ImportError("Please `pip install datasets` to use --hf-stream-dataset.") from e

    from itertools import islice
    stream = load_dataset(dataset_name, split=split, streaming=True)
    # Shuffle the streaming iterator to approximate random sampling without downloading the full dataset
    if n_samples is None or n_samples <= 0:
        # still allow shuffled streaming (full pass) if user asked for all
        try:
            stream = stream.shuffle(seed=seed, buffer_size=max(1000, shuffle_buffer_size))
        except Exception:
            pass
    else:
        try:
            stream = stream.shuffle(seed=seed, buffer_size=max(1000, shuffle_buffer_size))
        except Exception:
            pass
    i = 0
    it = iter(stream)
    ready_marker_written = False
    while (n_samples <= 0) or (i < n_samples):
        remaining = None if n_samples <= 0 else max(0, n_samples - i)
        this_chunk = chunk_size if remaining is None else min(chunk_size, remaining)
        chunk = list(islice(it, this_chunk))
        if not chunk:
            break
        part_dir = os.path.join(output_dir, f"part_{i // chunk_size:02d}")
        HFDataset.from_list(chunk).save_to_disk(part_dir)
        i += len(chunk)
        print(f"[HFStream] Saved chunk {i // chunk_size} ({i} samples) -> {part_dir}", flush=True)
        # As soon as the first shard is saved, write a READY marker so other ranks can proceed
        if not ready_marker_written:
            try:
                with open(os.path.join(output_dir, "_STREAM_READY"), 'w', encoding='utf-8') as f:
                    f.write(str(i))
            except Exception:
                pass
            ready_marker_written = True
    return i

# ---------------------------- Janus TF logits ----------------------------
@torch.no_grad()
def build_prompt_tokens(vl_chat_processor: VLChatProcessor, captions: List[str]) -> torch.LongTensor:
    prompts = []
    for c in captions:
        conv = [
            {"role": "<|User|>", "content": c},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conv, sft_format=vl_chat_processor.sft_format, system_prompt=""
        )
        prompts.append(sft + vl_chat_processor.image_start_tag)
    token_lists = [vl_chat_processor.tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in token_lists)
    pad_id = vl_chat_processor.pad_id
    tokens = torch.full((len(token_lists), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(token_lists):
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return tokens


def tf_logits_janus(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt_tokens: torch.LongTensor,
    tf_input: torch.LongTensor,
    cfg_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute teacher-forcing logits over all positions of tf_input.
    Returns [B, T, V] of guided logits for image token vocabulary.
    """
    device = next(mmgpt.parameters()).device
    bsz, t_len = tf_input.shape
    pad_id = vl_chat_processor.pad_id

    # Fast path: when cfg_scale ~= 1.0, use conditional path only (no cond/uncond duplication)
    if abs(float(cfg_scale) - 1.0) < 1e-6:
        cond_tokens = prompt_tokens.to(device)  # [B, Lp]
        cond_prompt_embeds = mmgpt.language_model.get_input_embeddings()(cond_tokens)  # [B, Lp, D]
        # Build attention mask: valid prompt tokens then all ones for TF image tokens
        cond_prompt_mask = (cond_tokens != pad_id).long()  # [B, Lp]
        tf_seq = tf_input.to(device)  # [B, T]
        img_embed_list = []
        for i in range(t_len):
            next_tok = tf_seq[:, i]
            img_embeds = mmgpt.prepare_gen_img_embeds(next_tok)  # [B, D]
            img_embed_list.append(img_embeds.unsqueeze(1))
        img_embeds_seq = torch.cat(img_embed_list, dim=1)  # [B, T, D]
        inputs_full = torch.cat([cond_prompt_embeds, img_embeds_seq], dim=1)  # [B, Lp+T, D]
        attn_mask_full = torch.cat([
            cond_prompt_mask.to(device),
            torch.ones((cond_prompt_mask.shape[0], t_len), dtype=cond_prompt_mask.dtype, device=device)
        ], dim=1)
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_full, attention_mask=attn_mask_full, use_cache=False)
        hidden = outputs.last_hidden_state  # [B, Lp+T, D]
        Lp = cond_prompt_embeds.shape[1]
        step_h = hidden[:, Lp:Lp + t_len, :]  # [B, T, D]
        B, T, D = step_h.shape
        step_logits = mmgpt.gen_head(step_h.reshape(B * T, D)).view(B, T, -1)  # [B, T, V]
        return step_logits

    # General path: duplicate cond/uncond and combine
    tokens = torch.zeros((bsz * 2, prompt_tokens.shape[1]), dtype=torch.long, device=device)
    for i in range(bsz * 2):
        idx = i // 2
        tokens[i, :] = prompt_tokens[idx].to(device)
        if i % 2 != 0:  # uncond
            tokens[i, 1:-1] = pad_id

    prompt_embeds = mmgpt.language_model.get_input_embeddings()(tokens)  # [B*2, Lp, D]
    # Build prompt attention masks; set uncond rows to all ones (like reference)
    prompt_mask = (tokens != pad_id).long()  # [B*2, Lp]
    if prompt_mask.shape[0] >= 2:
        prompt_mask[1::2] = 1
    dup_tf = torch.cat([tf_input, tf_input], dim=0).to(device)  # [B*2, T]
    img_embed_list = []
    for i in range(t_len):
        next_tok = dup_tf[:, i]
        img_embeds = mmgpt.prepare_gen_img_embeds(next_tok)  # [B*2, D]
        img_embed_list.append(img_embeds.unsqueeze(1))
    img_embeds_seq = torch.cat(img_embed_list, dim=1)  # [B*2, T, D]

    inputs_embeds_full = torch.cat([prompt_embeds.to(device), img_embeds_seq], dim=1)  # [B*2, Lp+T, D]
    attn_mask_full = torch.cat([
        prompt_mask.to(device),
        torch.ones((prompt_mask.shape[0], t_len), dtype=prompt_mask.dtype, device=device)
    ], dim=1)
    outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds_full, attention_mask=attn_mask_full, use_cache=False)
    hidden = outputs.last_hidden_state  # [B*2, Lp+T, D]
    Lp = prompt_embeds.shape[1]
    step_h = hidden[:, Lp:Lp + t_len, :]  # [B*2, T, D]
    B2, T, D = step_h.shape
    step_logits_all = mmgpt.gen_head(step_h.reshape(B2 * T, D)).view(B2, T, -1)  # [B*2, T, V]

    logit_cond = step_logits_all[0::2, :, :]  # [B, T, V]
    logit_uncond = step_logits_all[1::2, :, :]  # [B, T, V]
    guided_logits = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
    return guided_logits


def sample_one_sequence_from_tf(logits_bt_vocab: torch.Tensor, temperature: float, top_k: int, top_p: float):
    b, t, v = logits_bt_vocab.shape
    logits = logits_bt_vocab / max(temperature, 1e-5)
    logits_flat = logits.reshape(b * t, v).contiguous()
    if top_k > 0 or top_p < 1.0:
        logits_flat = top_k_top_p_filtering(logits_flat, top_k=top_k, top_p=top_p)
    log_probs_flat = torch.log_softmax(logits_flat, dim=-1)
    probs_flat = torch.softmax(logits_flat, dim=-1)
    idx_flat = torch.multinomial(probs_flat, num_samples=1)
    idx_T = idx_flat.view(b, t)
    gather_lp = log_probs_flat.gather(1, idx_flat).view(b, t)
    seq_log_prob = torch.sum(gather_lp, dim=-1)
    return idx_T, gather_lp, seq_log_prob


def perturb_teacher_forcing_inputs(
    gt_tokens_full: torch.Tensor, use_noise: bool, noise_prob: float, codebook_size: int, rng: Optional[torch.Generator]
):
    # Use full latent length to keep sequence length consistent with latent_T
    tf_base = gt_tokens_full
    n, t_minus_1 = tf_base.shape
    if not use_noise:
        return tf_base, torch.zeros((n,), device=gt_tokens_full.device, dtype=torch.float32)
    prob = max(0.0, min(1.0, float(noise_prob)))
    eps = torch.full((n,), prob, device=gt_tokens_full.device, dtype=torch.float32)
    bern = torch.bernoulli(eps.view(n, 1).expand(n, t_minus_1))
    bern_bool = bern.bool()
    ui = torch.randint(low=0, high=int(codebook_size), size=(n, t_minus_1), device=gt_tokens_full.device, generator=rng, dtype=tf_base.dtype)
    tf_noisy = torch.where(bern_bool, ui, tf_base)
    return tf_noisy, eps


# ---------------------------- Main Train ----------------------------
def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU."

    init_distributed_mode(args)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.manual_seed(args.global_seed * world_size + rank)
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    # Processor & Model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    mmgpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    mmgpt = mmgpt.to(torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16 if args.mixed_precision == 'fp16' else torch.float32)
    mmgpt.gen_vision_model.requires_grad_(False)  # freeze VQ
    mmgpt = mmgpt.to(device)

    # Derive model's actual codebook size once to avoid relying on CLI defaults
    try:
        codebook_size_model = None
        # common attribute names across VQ/VAE implementations
        if hasattr(mmgpt.gen_vision_model, 'codebook_size'):
            codebook_size_model = int(getattr(mmgpt.gen_vision_model, 'codebook_size'))
        elif hasattr(mmgpt.gen_vision_model, 'codebook') and hasattr(mmgpt.gen_vision_model.codebook, 'num_embeddings'):
            codebook_size_model = int(mmgpt.gen_vision_model.codebook.num_embeddings)
        elif hasattr(mmgpt.gen_vision_model, 'vq') and hasattr(mmgpt.gen_vision_model.vq, 'num_embeddings'):
            codebook_size_model = int(mmgpt.gen_vision_model.vq.num_embeddings)
        if codebook_size_model is None:
            codebook_size_model = int(args.codebook_size)
    except Exception:
        codebook_size_model = int(args.codebook_size)
    
    print(f"codebook_size_model: {codebook_size_model}")

    # Reduce memory: disable KV cache during training; enable gradient checkpointing if available
    try:
        if hasattr(mmgpt, 'config'):
            mmgpt.config.use_cache = False
        if hasattr(mmgpt, 'language_model') and hasattr(mmgpt.language_model, 'config'):
            mmgpt.language_model.config.use_cache = False
            # Prefer memory-efficient attention when available
            print("language_model config attn_implementation sdpa")
            setattr(mmgpt.language_model.config, 'attn_implementation', 'sdpa')
    except Exception:
        pass
    try:
        if hasattr(mmgpt, 'gradient_checkpointing_enable'):
            print("gradient_checkpointing_enable")
            mmgpt.gradient_checkpointing_enable()
    except Exception:
        pass
    try:
        
        if hasattr(mmgpt, 'language_model') and hasattr(mmgpt.language_model, 'gradient_checkpointing_enable'):
            print("language_model gradient_checkpointing_enable")
            mmgpt.language_model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Optionally compile (single GPU only)
    if args.torch_compile and (not dist.is_initialized() or world_size == 1):
        logger.info("Compiling model ...")
        mmgpt = torch.compile(mmgpt)

    # LPIPS
    if args.reward_perceptual_weight > 0.0:
        if LPIPS is None:
            raise ImportError("lpips is not installed. Please `pip install lpips`.")
        lpips_model = LPIPS().to(device)
        lpips_model.eval()
        lpips_model.requires_grad_(False)
    else:
        lpips_model = None

    # Dataset / Loader
    dataset = None
    hf_dir = None

    # Optionally stream HF dataset and save to disk (rank 0 only)
    if getattr(args, 'hf_stream_dataset', None):
        if not getattr(args, 'hf_stream_output_dir', None):
            raise ValueError("--hf-stream-output-dir is required when using --hf-stream-dataset")
        marker_path = os.path.join(args.hf_stream_output_dir, "_STREAM_DONE")
        ready_marker_path = os.path.join(args.hf_stream_output_dir, "_STREAM_READY")
        marker_exists = os.path.exists(marker_path)
        if marker_exists:
            # Reuse existing streamed dataset
            if rank == 0:
                logger.info(f"Found existing HF stream marker at {marker_path}; reusing dataset in {args.hf_stream_output_dir}")
            hf_dir = args.hf_stream_output_dir
        else:
            # Perform streaming on rank 0, others poll for completion marker
            if rank == 0:
                logger.info(f"Streaming HF dataset {args.hf_stream_dataset} split={args.hf_stream_split} -> {args.hf_stream_output_dir} (n_samples={int(getattr(args, 'hf_stream_n_samples', 0))}, buffer={int(getattr(args, 'hf_stream_shuffle_buffer', 10000))}, seed={int(getattr(args, 'hf_stream_seed', 0))})")
                total_saved = stream_hf_dataset_to_disk(
                    dataset_name=args.hf_stream_dataset,
                    split=args.hf_stream_split,
                    n_samples=int(getattr(args, 'hf_stream_n_samples', 0)),
                    chunk_size=int(getattr(args, 'hf_stream_chunk_size', 2000)),
                    output_dir=args.hf_stream_output_dir,
                    shuffle_buffer_size=int(getattr(args, 'hf_stream_shuffle_buffer', 10000)),
                    seed=int(getattr(args, 'hf_stream_seed', 0)),
                )
                logger.info(f"HF streaming complete: {total_saved} samples saved")
                # Write a simple done marker for other ranks to detect
                try:
                    with open(marker_path, 'w', encoding='utf-8') as f:
                        f.write(str(total_saved))
                except Exception as e:
                    logger.warning(f"Failed to write HF stream done marker: {e}")
            # Poll for readiness/done instead of holding a distributed barrier open during long I/O
            wait_timeout = int(os.environ.get('HF_STREAM_WAIT_TIMEOUT', '86400'))  # default: 24h
            poll_interval = float(os.environ.get('HF_STREAM_POLL_INTERVAL', '5'))
            start_ts = time.time()
            # Proceed when either DONE or READY marker exists or at least one shard (part_*) appears
            def has_at_least_one_shard(path: str) -> bool:
                try:
                    if not os.path.isdir(path):
                        return False
                    for name in os.listdir(path):
                        if name.startswith("part_") and os.path.isdir(os.path.join(path, name)):
                            di = os.path.join(path, name, "dataset_info.json")
                            if os.path.exists(di):
                                return True
                    return False
                except Exception:
                    return False
            while (not os.path.exists(marker_path)) and (not os.path.exists(ready_marker_path)) and (not has_at_least_one_shard(args.hf_stream_output_dir)):
                if (time.time() - start_ts) > wait_timeout:
                    raise RuntimeError(
                        f"Timeout waiting for HF streaming readiness. Neither READY nor DONE markers found under: {args.hf_stream_output_dir}"
                    )
                time.sleep(poll_interval)
            hf_dir = args.hf_stream_output_dir
        # Ensure all ranks proceed together after dataset availability decision
        dist.barrier()

    # Use HF dataset saved on disk if provided
    if getattr(args, 'hf_from_disk_dir', None):
        hf_dir = args.hf_from_disk_dir

    if hf_dir:
        dataset = HFDiskImageCaptionDataset(
            root_dir=hf_dir,
            image_size=args.image_size,
            image_key=getattr(args, 'hf_image_key', 'image'),
            caption_key=getattr(args, 'hf_caption_key', 'text'),
            data_start=getattr(args, 'data_start', None),
            data_end=getattr(args, 'data_end', None),
        )
        if rank == 0:
            logger.info(f"Using HFDiskImageCaptionDataset from {hf_dir} (image_key={getattr(args, 'hf_image_key', 'image')}, caption_key={getattr(args, 'hf_caption_key', 'text')})")
    elif getattr(args, 'data_path', None):
        dataset = LaionCocoTarCaptionDataset(
            data_path=args.data_path,
            image_size=args.image_size,
            data_start=getattr(args, 'data_start', None),
            data_end=getattr(args, 'data_end', None),
        )
        if rank == 0:
            logger.info("Using LaionCocoTarCaptionDataset from --data-path")
    elif getattr(args, 'data_tsv', None):
        dataset = LaionCocoDataset(args.data_tsv, image_size=args.image_size)
        if rank == 0:
            logger.info("Using LaionCocoDataset from --data-tsv")
    else:
        raise ValueError("Please provide either --hf-from-disk-dir, --hf-stream-dataset, --data-path (tar shards), or --data-tsv (TSV)")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.sample_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    if rank == 0:
        logger.info(f"Dataset size: {len(dataset):,}")

    # DeepSpeed or DDP
    # Synchronize before entering DeepSpeed initialization to avoid uneven arrival to group creation
    dist.barrier()
    use_deepspeed = bool(getattr(args, 'use_deepspeed', False))
    if use_deepspeed:
        assert deepspeed is not None, "Please install deepspeed to use --use-deepspeed"
        # Load config
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        # Resolve bf16/fp16 auto flags based on args.mixed_precision
        if args.mixed_precision == 'bf16' and 'bf16' in ds_config:
            ds_config['bf16']['enabled'] = True
            if 'fp16' in ds_config:
                ds_config['fp16']['enabled'] = False
        elif args.mixed_precision == 'fp16' and 'fp16' in ds_config:
            ds_config['fp16']['enabled'] = True
            if 'bf16' in ds_config:
                ds_config['bf16']['enabled'] = False
        # Ensure numeric batch size fields (avoid 'auto' strings)
        try:
            world = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            world = 1
        micro = int(getattr(args, 'train_batch_size', 1))
        ga_steps = int(getattr(args, 'gradient_accumulation_steps', 1))
        total = int(max(1, micro) * max(1, ga_steps) * max(1, world))
        ds_config['train_micro_batch_size_per_gpu'] = micro
        ds_config['gradient_accumulation_steps'] = ga_steps
        ds_config['train_batch_size'] = total
        # Enable ZeRO-aware grad clipping inside DeepSpeed
        ds_config['gradient_clipping'] = float(getattr(args, 'max_grad_norm', 1.0))
        # Ensure optimizer is explicitly defined for DS engine
        if 'optimizer' not in ds_config or not isinstance(ds_config['optimizer'], dict):
            ds_config['optimizer'] = {
                'type': 'AdamW',
                'params': {
                    'lr': float(getattr(args, 'lr', 1e-4)),
                    'betas': [float(getattr(args, 'beta1', 0.9)), float(getattr(args, 'beta2', 0.95))],
                    'eps': 1e-8,
                    'weight_decay': float(getattr(args, 'weight_decay', 5e-2)),
                }
            }
        # Initialize DS engine (let DS create optimizer)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=mmgpt,
            model_parameters=[p for p in mmgpt.parameters() if p.requires_grad],
            config=ds_config,
        )
        model_engine.train()
        scaler = None
    else:
        mmgpt = DDP(mmgpt, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=True, find_unused_parameters=False)
        mmgpt.train()
        model_engine = mmgpt
        optimizer = torch.optim.AdamW(
            [p for p in model_engine.parameters() if p.requires_grad],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Weights & Biases init (rank 0 only)
    if getattr(args, "use_wandb", False) and rank == 0:
        wandb.init(
            project=getattr(args, "wandb_project", "janus-grpo"),
            name=(getattr(args, "wandb_run_name", None) or f"janus-t2i-grpo-lr-{args.lr}-mse-{args.reward_rec_weight}-lpips-{args.reward_perceptual_weight}-ce-{args.aux_ce_weight}-clip-{args.clip_range}-cfg-{args.cfg_scale}-mode-{args.sample_model_mode}-noise-{args.use_token_noise}-{args.token_noise_prob}"),
            config={
                "model_path": args.model_path,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "codebook_size": args.codebook_size,
                "codebook_embed_dim": args.codebook_embed_dim,
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
                "kl_coef": args.kl_coef,
                "aux_ce_weight": args.aux_ce_weight,
                "cfg_scale": args.cfg_scale,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "sample_model_mode": args.sample_model_mode,
                "use_token_noise": args.use_token_noise,
                "token_noise_prob": args.token_noise_prob,
                "use_deepspeed": bool(getattr(args, 'use_deepspeed', False)),
            },
            mode="online",
        )

    # Optional base model for KL
    base_model = None
    if args.use_kl_loss and args.kl_coef > 0.0:
        base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            args.model_path if args.kl_base_path is None else args.kl_base_path, trust_remote_code=True
        )
        base_model = base_model.to(next(model_engine.parameters()).dtype).to(device)
        base_model.eval()
        base_model.gen_vision_model.requires_grad_(False)
        for p in base_model.parameters():
            p.requires_grad_(False)

    global_step = 0
    # VQ geometry will be inferred from encode() outputs per-batch

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        data_iter = iter(loader)
        model_engine.eval()

        # 1) Sampling per batch (teacher-forced logits)
        batch = next(data_iter)
        gt_img, captions = batch
        gt_img = gt_img.to(device, non_blocking=True)

        print(f"gt_img.shape: {gt_img.shape}")
        print(f"captions: {captions}")
        # Encode GT image to code indices
        with torch.no_grad():
            # Gather ZeRO-3 partitioned params for the vision (VQ) model before calling submodule methods
            with zero3_gather_params(model_engine.module.gen_vision_model):
                quant, _, info = model_engine.module.gen_vision_model.encode(gt_img)
            # quant expected shape: [B, C, H, W]
            latent_h = int(quant.shape[2])
            latent_w = int(quant.shape[3])
            codebook_embed_dim_model = int(quant.shape[1])
            indices = info[2]  # [B*H*W]
            gt_tokens = indices.view(gt_img.shape[0], -1).long()
            latent_T = int(gt_tokens.shape[1])

        # Build prompts and TF inputs
        prompt_tokens = build_prompt_tokens(vl_chat_processor, list(captions))  # [B, Lp]
        prompt_tokens = prompt_tokens.to(device)
        tf_inputs_for_sampling, eps_vec = perturb_teacher_forcing_inputs(
            gt_tokens,
            use_noise=args.use_token_noise,
            noise_prob=args.token_noise_prob,
            codebook_size=codebook_size_model,
            rng=None,
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=ptdtype):
                logits_bt_vocab = tf_logits_janus(
                    model_engine.module, vl_chat_processor, prompt_tokens, tf_inputs_for_sampling, cfg_scale=args.cfg_scale
                )

        # Sample G sequences for each source
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
        token_indices = torch.stack(all_token_indices, dim=1).view(-1, latent_T).clamp_(0, codebook_size_model - 1)
        token_log_probs = torch.stack(all_token_log_probs, dim=1).view(-1, latent_T)
        seq_log_probs = torch.stack(all_seq_log_probs, dim=1).view(-1)

        # Decode and rewards
        with torch.no_grad():
            def decode_chunk(chunk_indices: torch.Tensor) -> torch.Tensor:
                # Use model-inferred geometry and embedding dim
                chunk_shape = [chunk_indices.shape[0], codebook_embed_dim_model, latent_h, latent_w]
                with zero3_gather_params(model_engine.module.gen_vision_model):
                    return model_engine.module.gen_vision_model.decode_code(chunk_indices, shape=chunk_shape)

            pred_imgs = process_in_chunks(decode_chunk, token_indices, chunk_size=args.decode_chunk)
            gt_img_rep = repeat_tensor(gt_img, args.num_generations)

            if args.reward_rec_type == 'l1':
                rec_loss_vec = torch.mean(torch.abs(pred_imgs - gt_img_rep), dim=(1, 2, 3))
            else:
                rec_loss_vec = torch.mean((pred_imgs - gt_img_rep) ** 2, dim=(1, 2, 3))

            if lpips_model is not None and args.reward_perceptual_weight > 0.0:
                def lpips_chunk(pi, gi):
                    val = lpips_model(pi, gi)
                    if val.dim() > 1:
                        val = val.view(val.shape[0], -1).mean(dim=1)
                    return val
                perc_loss_vec = process_in_chunks(lpips_chunk, pred_imgs, gt_img_rep, chunk_size=args.decode_chunk)
            else:
                perc_loss_vec = torch.zeros_like(rec_loss_vec)

            combined_loss = args.reward_rec_weight * rec_loss_vec + args.reward_perceptual_weight * perc_loss_vec
            rewards = -combined_loss

        # Log reward statistics
        rewards_world = gather_tensor(rewards)
        dist.barrier()
        if getattr(args, "use_wandb", False) and rank == 0:
            wandb.log({
                "reward_mean": rewards_world.mean().item(),
                "reward_std": rewards_world.std().item(),
                "epoch": epoch,
                "rec_loss_mean": rec_loss_vec.mean().item(),
                "perc_loss_mean": perc_loss_vec.mean().item(),
            }, step=global_step)

        # Group-normalized advantages per source
        n_groups = rewards.shape[0] // args.num_generations
        advantages = torch.zeros_like(rewards)
        for i in range(n_groups):
            s = i * args.num_generations
            e = (i + 1) * args.num_generations
            r = rewards[s:e]
            advantages[s:e] = (r - r.mean()) / (r.std() + 1e-8)

        # Prepare repeated tensors for training
        gt_tokens_rep = repeat_tensor(gt_tokens, args.num_generations)
        prompt_tokens_rep = repeat_tensor(prompt_tokens, args.num_generations)
        tf_inputs_rep = repeat_tensor(tf_inputs_for_sampling, args.num_generations)
        eps_rep = repeat_tensor(eps_vec, args.num_generations)

        samples = {
            "gt_token_indices": gt_tokens_rep,
            "prompt_tokens": prompt_tokens_rep,
            "token_indices": token_indices,
            "token_log_probs": token_log_probs.detach(),
            "seq_log_probs": seq_log_probs.detach(),
            "rewards": rewards.detach(),
            "tf_teacher_inputs": tf_inputs_rep.detach(),
            "tf_noise_eps": eps_rep.detach(),
            "advantages": advantages.detach(),
        }

        # 2) Training over repeated samples (PPO/GRPO)
        model_engine.train()
        if not use_deepspeed:
            optimizer.zero_grad(set_to_none=True)
        num_samples = samples["token_indices"].shape[0]
        step_idx = 0

        for start in tqdm(range(0, num_samples, args.train_batch_size),
                          desc=f"Epoch {epoch}: training", position=0,
                          disable=(not dist.is_initialized() or rank != 0)):
            end = min(start + args.train_batch_size, num_samples)
            batch_sample = {k: (v[start:end] if isinstance(v, torch.Tensor) else v) for k, v in samples.items()}

            prev_training_mode = model_engine.training
            if args.sample_model_mode in ["eval", "twice"]:
                model_engine.eval()

            with torch.amp.autocast("cuda", dtype=ptdtype):
                new_logits = tf_logits_janus(
                    model_engine.module, vl_chat_processor, batch_sample["prompt_tokens"], batch_sample["tf_teacher_inputs"], cfg_scale=args.cfg_scale
                )  # [N, T, V]

            if args.sample_model_mode == "twice" and prev_training_mode:
                model_engine.train()
                with torch.amp.autocast("cuda", dtype=ptdtype):
                    logits_for_ce = tf_logits_janus(
                        model_engine.module, vl_chat_processor, batch_sample["prompt_tokens"], batch_sample["tf_teacher_inputs"], cfg_scale=args.cfg_scale
                    )
            else:
                logits_for_ce = new_logits

            logits = new_logits / max(args.temperature, 1e-5)
            log_probs = torch.log_softmax(logits, dim=-1)
            new_per_token_logps = log_probs.gather(-1, batch_sample["token_indices"].unsqueeze(-1)).squeeze(-1)

            per_token_kl = None
            if base_model is not None and args.use_kl_loss and args.kl_coef > 0.0:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=ptdtype):
                        base_logits = tf_logits_janus(
                            base_model, vl_chat_processor, batch_sample["prompt_tokens"], batch_sample["tf_teacher_inputs"], cfg_scale=args.cfg_scale
                        )
                temp = max(args.temperature, 1e-5)
                cur_lp = torch.log_softmax((new_logits / temp).to(torch.float32), dim=-1)
                ref_lp = torch.log_softmax((base_logits / temp).to(torch.float32), dim=-1)
                cur_tok = cur_lp.gather(-1, batch_sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                ref_tok = ref_lp.gather(-1, batch_sample["token_indices"].unsqueeze(-1)).squeeze(-1)
                per_token_kl = torch.exp(ref_tok - cur_tok) - (ref_tok - cur_tok) - 1

            adv = torch.clamp(batch_sample["advantages"], -args.adv_clip_max, args.adv_clip_max)
            ratio = torch.exp(new_per_token_logps - batch_sample["token_log_probs"])  # [N, T]
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)

            # clipping stats
            clip_mask = (torch.abs(ratio - 1.0) > args.clip_range)
            clip_ratio = clip_mask.float().mean().item()

            per_token_unclipped = -adv.unsqueeze(1) * ratio
            per_token_clipped = -adv.unsqueeze(1) * clipped_ratio
            per_token_loss = torch.maximum(per_token_unclipped, per_token_clipped)
            if per_token_kl is not None and args.use_kl_loss:
                per_token_loss = per_token_loss + args.kl_coef * per_token_kl

            seq_loss = per_token_loss.mean(dim=-1)
            loss = seq_loss.mean()

            if args.aux_ce_weight > 0:
                ce_val = F.cross_entropy(
                    logits_for_ce.reshape(-1, logits_for_ce.shape[-1]).to(torch.float32),
                    batch_sample["gt_token_indices"].reshape(-1),
                    reduction='mean'
                )
                loss = loss + args.aux_ce_weight * ce_val

            if use_deepspeed:
                # DeepSpeed manages gradient accumulation; do not scale loss manually
                model_engine.backward(loss)
            else:
                loss = loss / args.gradient_accumulation_steps
                if args.mixed_precision == 'fp16':
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            step_idx += 1
            print(f"step_idx: {step_idx}")

            # For DeepSpeed, call step() every iteration; DS performs the optimizer step at GA boundary internally
            if use_deepspeed:
                model_engine.step()

            if (step_idx % args.gradient_accumulation_steps) == 0:
                if not use_deepspeed:
                    if args.mixed_precision == 'fp16':
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Log training metrics
                if getattr(args, "use_wandb", False) and rank == 0:
                    grad_norm_val = 0.0
                    try:
                        if use_deepspeed and hasattr(model_engine, "get_global_grad_norm"):
                            grad_norm_val = float(model_engine.get_global_grad_norm())
                        else:
                            grad_norm_val = float(grad_norm)
                    except Exception:
                        pass
                    metrics = {
                        "loss": float(loss.item() * args.gradient_accumulation_steps),
                        "seq_loss_mean": float(seq_loss.mean().item()),
                        "clip_ratio": float(clip_ratio),
                        "ratio_mean": float(ratio.mean().item()),
                        "clipped_ratio_mean": float(clipped_ratio.mean().item()),
                        "grad_norm": float(grad_norm_val),
                        "clip_range": float(args.clip_range),
                        "aux_ce_weight": float(args.aux_ce_weight),
                        "kl_to_base": (float(per_token_kl.mean().detach().item()) if per_token_kl is not None else 0.0),
                    }
                    if args.aux_ce_weight > 0:
                        metrics["aux_ce_loss"] = float(ce_val.detach().item())
                    wandb.log(metrics, step=global_step)

                # checkpoint
                if args.ckpt_every > 0 and ((global_step + 1) % args.ckpt_every == 0):
                    ckpt_dir = os.path.join(
                        args.ckpt_dir,
                        f"janus-t2i-grpo-lr-{args.lr}-mse-{args.reward_rec_weight}-lpips-{args.reward_perceptual_weight}-ce-{args.aux_ce_weight}-clip-{args.clip_range}-cfg-{args.cfg_scale}-topk-{args.top_k}-topp-{args.top_p}-temp-{args.temperature}-mode-{args.sample_model_mode}-noise-{args.use_token_noise}-{args.token_noise_prob}-{global_step:07d}"
                    )
                    if use_deepspeed:
                        if rank == 0:
                            ensure_dir(ckpt_dir)
                        dist.barrier()
                        # Save model weights only (no optimizer state)
                        with zero3_gather_params(model_engine.module):
                            if rank == 0:
                                torch.save(model_engine.module.state_dict(), os.path.join(ckpt_dir, "consolidated.pth"))
                                logger.info(f"Saved model weights (no optimizer) to {ckpt_dir}")
                    else:
                        if rank == 0:
                            ensure_dir(ckpt_dir)
                            module = model_engine.module
                            torch.save(module.state_dict(), os.path.join(ckpt_dir, "consolidated.pth"))
                            logger.info(f"Saved checkpoint to {ckpt_dir}")
                        dist.barrier()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                global_step += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"Finished epoch {epoch}")

    model_engine.module.eval()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data-tsv", type=str, default=None, help="TSV with <abs_image_path>\t<caption>")
    parser.add_argument("--data-path", type=str, default=None, help="Tar shards dir/file(s) for LAION/COCO (supports comma, glob, brace)")
    parser.add_argument("--data-start", type=int, default=None, help="Start index (inclusive) to slice shards when input is a directory")
    parser.add_argument("--data-end", type=int, default=None, help="End index (inclusive) to slice shards when input is a directory")
    # HF dataset options
    parser.add_argument("--hf-stream-dataset", type=str, default=None, help="HF dataset name to stream and save to disk (e.g., 'LucasFang/FLUX-Reason-6M')")
    parser.add_argument("--hf-stream-split", type=str, default="train", help="Split for HF streaming (default: train)")
    parser.add_argument("--hf-stream-n-samples", type=int, default=0, help="Num samples to stream (0=all)")
    parser.add_argument("--hf-stream-chunk-size", type=int, default=2000, help="Chunk size to save per shard when streaming")
    parser.add_argument("--hf-stream-output-dir", type=str, default=None, help="Output directory to save streamed HF shards")
    parser.add_argument("--hf-stream-shuffle-buffer", type=int, default=10000, help="Shuffle buffer size for HF streaming (approx random sampling)")
    parser.add_argument("--hf-stream-seed", type=int, default=0, help="Random seed for HF streaming shuffle")
    parser.add_argument("--hf-from-disk-dir", type=str, default=None, help="Directory containing HF Dataset.save_to_disk shards (e.g., part_00, part_01, ...)")
    parser.add_argument("--hf-image-key", type=str, default="image", help="Column name for image in HF dataset (default: image)")
    parser.add_argument("--hf-caption-key", type=str, default="text", help="Column name for caption in HF dataset (default: text)")
    parser.add_argument("--num-workers", type=int, default=8)

    # model & tokenizer
    parser.add_argument("--model-path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    # training schedule
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--torch-compile", action='store_true')

    # sampling
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--decode-chunk", type=int, default=8)

    # training setup
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--adv-clip-max", type=float, default=5.0)
    parser.add_argument("--clip-range", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--sample-model-mode", type=str, choices=["train", "eval", "twice"], default="eval")

    # rewards
    parser.add_argument("--reward-rec-type", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--reward-rec-weight", type=float, default=1.0)
    parser.add_argument("--reward-perceptual-weight", type=float, default=1.0)
    parser.add_argument("--aux-ce-weight", type=float, default=0.0)

    # KL
    parser.add_argument("--use-kl-loss", action='store_true')
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--kl-base-path", type=str, default=None)

    # checkpointing
    parser.add_argument("--ckpt-every", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")

    # teacher-forcing noise
    parser.add_argument("--use-token-noise", action='store_true')
    parser.add_argument("--token-noise-prob", type=float, default=0.5)

    # DDP runtime (env var driven)
    parser.add_argument("--local-rank", type=int, default=0)
    # DeepSpeed
    parser.add_argument("--use-deepspeed", action='store_true')
    parser.add_argument("--deepspeed-config", type=str, default=os.path.join(os.path.dirname(__file__), 'configs/zero3.json'))

    # wandb
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project", type=str, default="janus-grpo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()
    main(args)


