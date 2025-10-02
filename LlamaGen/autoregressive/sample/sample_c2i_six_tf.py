import os
import time
import argparse
import json
import random

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image as tv_to_pil_image
from torchvision.datasets import ImageNet as TorchVisionImageNet
import tarfile
from glob import glob
try:
    import datasets as hf_datasets
    HAS_HF_DATASETS = True
except Exception:
    HAS_HF_DATASETS = False

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import top_k_top_p_filtering
from dataset.build import build_dataset
from dataset.augmentation import center_crop_arr
from PIL import Image, ImageDraw, ImageFont

# Encoding raw images to codes will use VQ model's encode() as in extract_codes_c2i.py

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def has_imagenet_folders(root: str) -> bool:
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        return False
    # ensure there are subfolders (classes)
    def has_subdirs(d):
        try:
            for name in os.listdir(d):
                if os.path.isdir(os.path.join(d, name)):
                    return True
        except Exception:
            return False
        return False
    return has_subdirs(train_dir) and has_subdirs(val_dir)


def try_prepare_with_torchvision(root: str) -> bool:
    try:
        # This will parse archives if in official format; otherwise raises.
        TorchVisionImageNet(root=root, split='train')
        TorchVisionImageNet(root=root, split='val')
        return has_imagenet_folders(root)
    except Exception:
        return False


def extract_archives_fallback(archives_root: str, dest_root: str):
    # ensure destination directories exist (user-writable)
    ensure_dir(os.path.join(dest_root, 'train'))
    ensure_dir(os.path.join(dest_root, 'val'))
    # Heuristics: extract any archives matching *train* to train/, *val* to val/
    archives = []
    archives.extend(glob(os.path.join(archives_root, '*.tar')))  # raw tar
    archives.extend(glob(os.path.join(archives_root, '*.tar.gz')))  # gz
    for arch in sorted(archives):
        lower = os.path.basename(arch).lower()
        if ('train' in lower) or ('training' in lower):
            dest = os.path.join(dest_root, 'train')
        elif ('val' in lower) or ('validation' in lower):
            dest = os.path.join(dest_root, 'val')
        else:
            # skip unknown
            continue
        try:
            with tarfile.open(arch, 'r:*') as tf:
                tf.extractall(dest)
            print(f"Extracted {arch} -> {dest}")
        except Exception as e:
            print(f"Warning: failed to extract {arch}: {e}")


def prepare_imagenet_if_needed(data_root: str, archive_subdir: str | None = None, extract_root: str | None = None) -> str:
    """
    Ensures an ImageNet folder structure with train/ and val/ exists and returns the prepared root path.
    - data_root: path that may contain prepared folders or an archive subdir
    - archive_subdir: optional subfolder name (e.g., 'data') where archives live
    - extract_root: optional destination to extract into (should be user-writable)
    """
    prepared_root = extract_root if extract_root else data_root
    if has_imagenet_folders(prepared_root):
        return prepared_root
    print("Preparing ImageNet folders from archives...")
    # First try torchvision parser only when extracting into prepared_root
    if prepared_root == data_root and try_prepare_with_torchvision(prepared_root):
        print("Prepared using torchvision.datasets.ImageNet parser.")
        return prepared_root
    # Fallback: extract our own archives from archives_root -> prepared_root
    archives_root = os.path.join(data_root, archive_subdir) if (archive_subdir and os.path.isdir(os.path.join(data_root, archive_subdir))) else data_root
    try:
        extract_archives_fallback(archives_root, prepared_root)
    except PermissionError as e:
        raise PermissionError(f"No write permission to '{prepared_root}'. Please set --extract-root to a writable path. Original error: {e}")
    if not has_imagenet_folders(prepared_root):
        raise RuntimeError("Failed to prepare ImageNet folder structure. Please ensure archives are under the specified archive subdir and contain class subfolders or images in per-class tarballs.")
    return prepared_root


def select_one_per_class_raw(loader, wanted_classes=None, max_scan: int = 200000):
    """Iterate loader and pick one example per class.
    - If wanted_classes is provided and non-empty: only collect those classes and stop early when all are found or max_scan reached.
    - If wanted_classes is None or empty: collect the first seen example for any class (useful for random sampling), scanning up to max_scan.
    Returns dict: {label_int: (img, label, path)}
    """
    found = {}
    scanned = 0
    filter_enabled = wanted_classes is not None and len(wanted_classes) > 0
    wanted_set = set(wanted_classes) if filter_enabled else None
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            img, label = batch
            path = None
        elif isinstance(batch, (list, tuple)) and len(batch) == 3:
            img, label, path = batch
        else:
            img, label = batch
            path = None
        label_int = int(label.reshape(-1)[0].item())
        if (not filter_enabled) or (label_int in wanted_set):
            if label_int not in found:
                found[label_int] = (img, label, path)
                if filter_enabled and len(found) == len(wanted_set):
                    break
        scanned += 1
        if scanned >= max_scan:
            break
    return found


class HFImageDataset(torch.utils.data.Dataset):
    """HuggingFace arrow-based dataset loader for images and labels.
    Expects a directory saved via datasets.save_to_disk with data-*.arrow shards,
    dataset_info.json, state.json, and an 'image' column and a 'label' column.
    """
    def __init__(self, root: str, transform=None):
        if not HAS_HF_DATASETS:
            raise ImportError("Please install 'datasets' (pip install datasets) to use --hf-train-dir.")
        self.ds = hf_datasets.load_from_disk(root)
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


def load_hf_label_names(ds_root: str):
    """Try to read ClassLabel names from a HuggingFace save_to_disk directory.
    Looks for dataset_info.json in <root>/ and <root>/train/.
    Returns a list of names or None if unavailable.
    """
    candidates = [ds_root, os.path.join(ds_root, 'train')]
    for base in candidates:
        info_path = os.path.join(base, 'dataset_info.json')
        if os.path.isfile(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Typical structure: {"features": {"label": {"names": [...]}}}
                features = data.get('features') or {}
                label_feat = features.get('label') or {}
                names = label_feat.get('names')
                if isinstance(names, list) and len(names) > 0:
                    return names
            except Exception:
                pass
    return None


def safe_slug(text: str) -> str:
    """Make a simple filesystem-friendly slug from label text."""
    s = str(text).strip().replace(' ', '_')
    return ''.join(ch for ch in s if (ch.isalnum() or ch in '._-'))


def to_vis_range(img: torch.Tensor) -> torch.Tensor:
    """Clamp [-1,1] and map to [0,1] for visualization."""
    img = img.detach().clamp(min=-1.0, max=1.0)
    return (img + 1.0) * 0.5


def mse_loss(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean squared error between two image tensors [3,H,W] in same range."""
    return float(torch.mean((a - b) ** 2).item())


def save_images_with_titles(images, titles, footers, out_path: str, font_size: int = 14, tile_pad: int = 10, col_pad: int = 10):
    """
    Compose a single-row image with per-tile titles/footers.
    - images: list of [3,H,W] tensors in [-1,1]
    - titles: list of strings
    - footers: list of strings or list of list-of-strings (lines)
    """
    assert len(images) == len(titles) == len(footers)
    pil_tiles = [tv_to_pil_image(to_vis_range(im)) for im in images]
    widths, heights = zip(*[p.size for p in pil_tiles])
    W = max(widths)
    H = max(heights)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Normalize footers to list of lines
    footer_lines = []
    for f in footers:
        if isinstance(f, (list, tuple)):
            footer_lines.append([str(x) for x in f])
        else:
            footer_lines.append([str(f)])

    # Measurement helpers using font metrics when available
    def measure_line_height(text: str) -> int:
        if font and hasattr(font, 'getbbox'):
            box = font.getbbox(text if len(text) > 0 else "A")
            return max(1, box[3] - box[1])
        return font_size

    def measure_lines_height(lines) -> int:
        if not lines:
            return 0
        total = 0
        for line in lines:
            total += measure_line_height(line) + 4  # 4px spacing between lines
        total += 6  # bottom margin to avoid clipping descenders
        return total

    title_h = measure_lines_height(["A"]) if titles else 0
    footer_hs = [measure_lines_height(lines) for lines in footer_lines]
    footer_h = max(footer_hs) if footer_lines else 0
    tile_h = H + title_h + footer_h + tile_pad * 2
    total_w = len(pil_tiles) * W + (len(pil_tiles) - 1) * col_pad
    total_h = tile_h
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for idx, tile in enumerate(pil_tiles):
        # center image within tile
        img_x = x + (W - tile.size[0]) // 2
        img_y = tile_pad + title_h
        canvas.paste(tile, (img_x, img_y))
        # title
        if titles and font:
            title = titles[idx]
            tw = draw.textlength(title, font=font) if hasattr(draw, 'textlength') else len(title) * (font_size // 2)
            tx = x + (W - int(tw)) // 2
            ty = tile_pad // 2
            draw.text((tx, ty), title, fill=(0, 0, 0), font=font)
        # footer lines
        fy = img_y + H + 2
        for line in footer_lines[idx]:
            if font:
                draw.text((x + 4, fy), line, fill=(0, 0, 0), font=font)
                fy += measure_line_height(line) + 4
            else:
                draw.text((x + 4, fy), line, fill=(0, 0, 0))
                fy += font_size + 4
        x += W + col_pad

    canvas.save(out_path)


def tf_logits(model, cls_ids: torch.Tensor, gt_tokens: torch.Tensor, cls_token_num: int):
    """Run a teacher-forcing forward pass to obtain per-position logits.
    cls_ids: [B]
    gt_tokens: [B, T] ground-truth token indices (full target sequence)
    Returns logits aligned to predict each target position (length T).
    """
    device = gt_tokens.device
    # teacher forcing input excludes the last target token
    tf_input = gt_tokens[:, :-1]
    # sequence length seen by the model includes class tokens
    input_pos = torch.arange(0, tf_input.shape[1] + cls_token_num, device=device)
    logits, _ = model(idx=tf_input, cond_idx=cls_ids, input_pos=input_pos)
    # align to targets as in training: start at cls_token_num-1, yielding T positions
    aligned_logits = logits[:, cls_token_num - 1:]
    return aligned_logits


def logits_to_sequences(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                        num_samples: int = 5):
    """Given TF logits [B, T, V], produce 1 greedy + N sampled sequences per batch.
    Returns tensor of shape [B, 1+num_samples, T].
    """
    B, T, V = logits.shape
    # Greedy: [B, T]
    greedy = torch.argmax(logits, dim=-1).unsqueeze(1)  # -> [B, 1, T]
    # Sampling
    logits_scaled = logits / max(temperature, 1e-5)
    logits_flat = logits_scaled.reshape(B * T, V).contiguous()
    if top_k > 0 or top_p < 1.0:
        logits_flat = top_k_top_p_filtering(logits_flat, top_k=top_k, top_p=top_p)
    probs_flat = torch.softmax(logits_flat, dim=-1)
    samples = []
    for _ in range(num_samples):
        idx_flat = torch.multinomial(probs_flat, num_samples=1)  # [B*T, 1]
        idx_bt = idx_flat.view(B, T).unsqueeze(1)  # [B, 1, T]
        samples.append(idx_bt)
    sampled = torch.cat(samples, dim=1) if samples else torch.empty(B, 0, T, dtype=torch.long, device=logits.device)
    return torch.cat([greedy, sampled], dim=1)


def main(args):
    # Setup
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
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
    elif "model" in gpt_ckpt:
        model_weight = gpt_ckpt["model"]
    elif "module" in gpt_ckpt:
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
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)
    else:
        print("no need to compile model in demo")

    # Classes (default to random classes if not explicitly provided)
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    has_explicit_labels = args.class_labels is not None and len(args.class_labels) > 0
    if has_explicit_labels:
        class_labels = [int(x) for x in args.class_labels.split(',')]

    # Dataset: load raw ImageNet images with center-crop and normalization like extract_codes_c2i.py
    crop_size = int(args.image_size)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    label_names = None
    # Use HuggingFace Arrow dataset
    dataset = HFImageDataset(args.hf_train_dir, transform=transform)
    # Try to load textual class names from dataset_info.json
    label_names = load_hf_label_names(args.hf_train_dir)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Sequential mode: consume first N samples directly if no explicit classes provided
    if not has_explicit_labels:
        ensure_dir(args.output_dir)
        block_T = latent_size ** 2
        num_to_take = max(1, int(args.num_samples))
        taken = 0
        for batch in loader:
            print(f'batch size: {len(batch)}')
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                img_cpu, label_cpu = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 3:
                img_cpu, label_cpu, _ = batch
            else:
                img_cpu, label_cpu = batch
            print(f'img_cpu shape: {img_cpu.shape}')
            print(f'label_cpu shape: {label_cpu.shape}')
            cls_id = int(label_cpu.reshape(-1)[0].item())

            x = img_cpu.to(device, non_blocking=True)
            with torch.no_grad():
                quant, _, [_, _, indices] = vq_model.encode(x)
            # indices already flattened over spatial dims (B*H*W). Use full vector.
            print(f'quant shape: {quant.shape}')
            print(f'indices shape: {indices.shape}')
            indices = indices.view(-1)
            print(f'indices shape: {indices.shape}')
            # Ensure length exactly T = latent_size*latent_size
            if indices.numel() > block_T:
                z = indices[:block_T]
            elif indices.numel() < block_T:
                pad = torch.full((block_T - indices.numel(),), fill_value=0, dtype=indices.dtype, device=indices.device)
                z = torch.cat([indices, pad], dim=0)
            else:
                z = indices
            z = z.unsqueeze(0).long()
            # Clamp to valid range for safety
            z = z.clamp_(min=0, max=int(args.codebook_size) - 1)
            cls = torch.tensor([cls_id], device=device, dtype=torch.long)

            t0 = time.time()
            logits = tf_logits(gpt_model, cls, z, args.cls_token_num)
            seqs = logits_to_sequences(
                logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_samples=5,
            )
            index_sample = seqs.squeeze(0)
            # Ensure indices are within valid codebook range to avoid invalid decodes
            index_sample = index_sample.clamp(min=0, max=int(args.codebook_size) - 1)
            print(f"TF logits computed for class {cls_id} in {time.time()-t0:.2f}s")

            qzshape = [index_sample.shape[0], args.codebook_embed_dim, latent_size, latent_size]
            images = vq_model.decode_code(index_sample, qzshape)
            # Decode ground-truth tokens as reference image
            qzshape_gt = [1, args.codebook_embed_dim, latent_size, latent_size]
            gt_image = vq_model.quantize.get_codebook_entry(z, qzshape_gt, channel_first=True)
            gt_image = vq_model.decode(gt_image)  # [1, 3, H, W]
            # Also decode reconstruction directly from quant features (sanity check)
            # recon_from_quant = vq_model.decode(quant).detach().cpu()  # [1, 3, H, W]

            # Prepare tiles: GT(raw), Token, Greedy, Randoms
            raw_3chw = x.detach().cpu().squeeze(0)
            token_3chw = gt_image.cpu().squeeze(0)
            # quant_3chw = recon_from_quant.squeeze(0)
            pred_tiles = [images[i].cpu() for i in range(images.shape[0])]  # [6, 3, H, W]

            tiles = [raw_3chw, token_3chw]
            titles = ["GT Image", "Token Image"]
            footers = ["", ""]
            pred_titles = ["Greedy Sample"] + [f"Random Sample {i}" for i in range(1, images.shape[0])]
            # MSE for Quant vs GT, Token vs GT
            # mse_q_gt = mse_loss(quant_3chw, raw_3chw)
            mse_t_gt = mse_loss(token_3chw, raw_3chw)
            # footers[1] = [f"MSE vs GT: {mse_q_gt:.6f}"]
            footers[1] = [f"MSE vs GT: {mse_t_gt:.6f}"]
            # MSE for preds vs GT and vs Token
            pred_footers = []
            for im in pred_tiles:
                mse_gt = mse_loss(im, raw_3chw)
                mse_tok = mse_loss(im, token_3chw)
                pred_footers.append([f"MSE vs GT: {mse_gt:.6f}", f"MSE vs Token: {mse_tok:.6f}"])
            tiles.extend(pred_tiles)
            titles.extend(pred_titles)
            footers.extend(pred_footers)

            if label_names is not None and 0 <= int(cls_id) < len(label_names):
                cls_name = label_names[int(cls_id)]
            else:
                cls_name = None
            prefix = f"sample_{taken:04d}_class_{cls_id}"
            out_name = prefix + ("_" + safe_slug(cls_name) if cls_name else "") + "_tf_with_metrics.png"
            out_path = os.path.join(args.output_dir, out_name)
            save_images_with_titles(tiles, titles, footers, out_path)
            # Write MSE report
            report_path = out_path.replace('.png', '.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Tile, MSE_vs_GT, MSE_vs_Token\n")
                f.write("GT Image, 0.0, -\n")
                # f.write(f"Quant Image, {mse_q_gt:.8f}, -\n")
                f.write(f"Token Image, {mse_t_gt:.8f}, -\n")
                for i, im in enumerate(pred_tiles):
                    mse_gt = mse_loss(im, raw_3chw)
                    mse_tok = mse_loss(im, token_3chw)
                    name = pred_titles[i]
                    f.write(f"{name}, {mse_gt:.8f}, {mse_tok:.8f}\n")
            print(f"Saved grid and metrics -> {out_path}")

            taken += 1
            if taken >= num_to_take:
                break
        return

    # Explicit-class mode (unchanged flow)
    missing = [c for c in class_labels if c not in picked]
    if len(missing) > 0:
        print(f"Warning: classes not found within first {args.max_scan} samples: {missing}")

    ensure_dir(args.output_dir)

    block_T = latent_size ** 2

    for cls_id in class_labels:
        if cls_id not in picked:
            continue
        img_cpu, label_cpu, _ = picked[cls_id]
        x = img_cpu.to(device, non_blocking=True)
        with torch.no_grad():
            quant, _, [_, _, indices] = vq_model.encode(x)
        # indices already flattened over spatial dims (B*H*W). Use full vector.
        indices = indices.view(-1)
        if indices.numel() > block_T:
            z = indices[:block_T]
        elif indices.numel() < block_T:
            pad = torch.full((block_T - indices.numel(),), fill_value=0, dtype=indices.dtype, device=indices.device)
            z = torch.cat([indices, pad], dim=0)
        else:
            z = indices
        z = z.unsqueeze(0).long()
        z = z.clamp_(min=0, max=int(args.codebook_size) - 1)
        cls = torch.tensor([cls_id], device=device, dtype=torch.long)

        t0 = time.time()
        logits = tf_logits(gpt_model, cls, z, args.cls_token_num)
        seqs = logits_to_sequences(
            logits,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_samples=5,
        )
        index_sample = seqs.squeeze(0)
        # Ensure indices are within valid codebook range to avoid invalid decodes
        index_sample = index_sample.clamp(min=0, max=int(args.codebook_size) - 1)
        print(f"TF logits computed for class {cls_id} in {time.time()-t0:.2f}s")

        qzshape = [index_sample.shape[0], args.codebook_embed_dim, latent_size, latent_size]
        images = vq_model.decode_code(index_sample, qzshape)
        # Decode ground-truth tokens as reference image
        qzshape_gt = [1, args.codebook_embed_dim, latent_size, latent_size]
        gt_image = vq_model.quantize.get_codebook_entry(z, qzshape_gt, channel_first=True)
        gt_image = vq_model.decode(gt_image)  # [1, 3, H, W]
        # Also decode reconstruction directly from quant features
        # recon_from_quant = vq_model.decode(quant).detach().cpu()

        # Assemble tiles and titles
        raw_3chw = x.detach().cpu().squeeze(0)
        token_3chw = gt_image.cpu().squeeze(0)
        pred_tiles = [images[i].cpu() for i in range(images.shape[0])]
        tiles = [raw_3chw, token_3chw] + pred_tiles
        titles = ["GT Image", "Token Image"] + ["Greedy Sample"] + [f"Random Sample {i}" for i in range(1, images.shape[0])]

        # Footers with MSEs (vs GT and vs Token)
        mse_t_gt = mse_loss(token_3chw, raw_3chw)
        footers = ["", [f"MSE vs GT: {mse_t_gt:.6f}"]]
        for im in pred_tiles:
            mse_gt = mse_loss(im, raw_3chw)
            mse_tok = mse_loss(im, token_3chw)
            footers.append([f"MSE vs GT: {mse_gt:.6f}", f"MSE vs Token: {mse_tok:.6f}"])

        if label_names is not None and 0 <= int(cls_id) < len(label_names):
            cls_name = label_names[int(cls_id)]
        else:
            cls_name = None
        if cls_name:
            out_name = f"class_{cls_id}_" + safe_slug(cls_name) + "_tf_with_metrics.png"
        else:
            out_name = f"class_{cls_id}_tf_with_metrics.png"
        out_path = os.path.join(args.output_dir, out_name)
        save_images_with_titles(tiles, titles, footers, out_path)
        # Write metrics file
        report_path = out_path.replace('.png', '.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Tile, MSE_vs_GT, MSE_vs_Token\n")
            f.write("GT Image, 0.0, -\n")
            f.write(f"Token Image, {mse_t_gt:.8f}, -\n")
            pred_names = ["Greedy Sample"] + [f"Random Sample {i}" for i in range(1, images.shape[0])]
            for name, im in zip(pred_names, pred_tiles):
                mse_gt = mse_loss(im, raw_3chw)
                mse_tok = mse_loss(im, token_3chw)
                f.write(f"{name}, {mse_gt:.8f}, {mse_tok:.8f}\n")
        print(f"Saved TF grid with metrics for class {cls_id} -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model + sampling
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
    parser.add_argument("--output-dir", type=str, default="samples_c2i_six_tf", help="directory to save 6-image TF grids per class")
    parser.add_argument("--class-labels", type=str, default=None, help="comma-separated class ids; default uses 8 predefined labels")
    parser.add_argument("--num-random-classes", type=int, default=8, help="when --class-labels is not provided, randomly sample this many classes (deprecated if --num-samples is set)")
    parser.add_argument("--num-samples", type=int, default=None, help="sequentially take first N samples from the loader when no --class-labels")
    # dataset
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--data-path", type=str, help='path to ImageNet root for raw images')
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scan", type=int, default=200000, help='max images to scan to find requested classes')
    parser.add_argument("--hf-train-dir", type=str, default=None, help='path to HF datasets save_to_disk directory containing train split .arrow shards')
    # imagenet archives handling
    parser.add_argument("--archive-subdir", type=str, default='data', help='subdirectory under data-path containing ImageNet archives (e.g., data)')
    parser.add_argument("--extract-root", type=str, default=None, help='writable directory to extract train/val folders; defaults to <output-dir>/imagenet_prepared')
    args = parser.parse_args()
    main(args)