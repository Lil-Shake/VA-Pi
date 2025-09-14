import torch
from typing import Tuple, Any, Optional
try:
    # TokLIP tokenizer (IBQ/VQ) components
    from tokenizer.ibq_modules.ibqgan import IBQ
except Exception:
    IBQ = None  # optional import; only needed if building TokLIP tokenizer here


class FrozenTokenizer:
    """
    Thin wrapper over the IBQ/VQ tokenizer/decoder exposing only encode/decode APIs.
    Assumes the underlying tokenizer is already loaded with pretrained weights.
    """
    def __init__(self, vq_model: Any):
        self.vq = vq_model.eval()
        for p in self.vq.parameters():
            p.requires_grad = False

    @property
    def codebook_size(self) -> int:
        return int(self.vq.quantize.n_e)

    @torch.no_grad()
    def encode_indices(self, images: torch.Tensor) -> torch.LongTensor:
        """Return code indices [B, H*W] or model-specific shape from frozen encoder/quantizer."""
        _, _, info = self.vq.encode(images)
        # info: (_, _, min_encoding_indices)
        idx = info[-1]
        return idx

    @torch.no_grad()
    def decode_code(self, indices: torch.LongTensor) -> torch.Tensor:
        """Decode code indices back to images via frozen decoder.
        indices: [B, HW] or [B, T]
        returns images in the tokenizer's output range (e.g., [-1, 1]).
        """
        return self.vq.decode_code(indices)


def build_tokenizer_frozen(vq_model: Any) -> FrozenTokenizer:
    """Wrap a loaded IBQ/VQ model as a frozen tokenizer/decoder."""
    return FrozenTokenizer(vq_model)


def build_gpt_policy(gpt_cls: Any, vocab_size: int, block_size: int, cfg: dict) -> Tuple[Any, torch.optim.Optimizer]:
    """Instantiate GPT policy and its optimizer.

    cfg keys: n_layer, n_head, n_embd, lr, weight_decay, use_pretrained_codebook (bool), codebook_ckpt_path (str),
              embd_pdrop, resid_pdrop, attn_pdrop, n_unmasked, class_num, token_drop, cls_token_number.
    """
    policy = gpt_cls(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 12),
        n_head=cfg.get("n_head", 12),
        n_embd=cfg.get("n_embd", 768),
        embd_pdrop=cfg.get("embd_pdrop", 0.0),
        resid_pdrop=cfg.get("resid_pdrop", 0.0),
        attn_pdrop=cfg.get("attn_pdrop", 0.0),
        n_unmasked=cfg.get("n_unmasked", 0),
        class_num=cfg.get("class_num", 1000),
        token_drop=cfg.get("token_drop", 0.1),
        cls_token_number=cfg.get("cls_token_number", 1),
        use_pretrained_codebook=cfg.get("use_pretrained_codebook", False),
        codebook_ckpt_path=cfg.get("codebook_ckpt_path", None),
        n_codebook_embd=cfg.get("n_codebook_embd", cfg.get("n_embd", 768)),
    )
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.get("lr", 2e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=cfg.get("betas", (0.9, 0.95)),
        eps=cfg.get("eps", 1e-8),
    )
    return policy, optimizer


def build_toklip_vq_model(init_args: dict, ckpt_path: Optional[str] = None, device: Optional[str] = None) -> Any:
    """
    Construct the TokLIP tokenizer (IBQ/VQ) model directly and optionally load a checkpoint.

    init_args must match IBQ constructor signature, e.g.:
      {
        "ddconfig": {...},
        "lossconfig": {...},
        "n_embed": 8192,
        "embed_dim": 256,
        "l2_normalize": False,
        "use_ema": False,
        "stage": None,
      }

    Returns the IBQ model instance suitable for wrapping with FrozenTokenizer, or for direct use in train loop cfg as vq_model.
    """
    if IBQ is None:
        raise ImportError("TokLIP IBQ module not found. Ensure 'src/tokenizer/ibq_modules/ibqgan.py' is importable.")
    model = IBQ(**init_args, ckpt_path=ckpt_path, stage=init_args.get("stage", None))
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def build_toklip_tokenizer(init_args: dict, ckpt_path: Optional[str] = None, device: Optional[str] = None) -> FrozenTokenizer:
    """
    Convenience wrapper: build TokLIP IBQ/VQ model and wrap as FrozenTokenizer.
    Use this if you want a ready-to-use encode/decode interface with frozen params.
    """
    model = build_toklip_vq_model(init_args, ckpt_path=ckpt_path, device=device)
    return build_tokenizer_frozen(model)


def build_ar_policy(vocab_size: int, block_size: int, ar_cfg: dict) -> Tuple[Any, torch.optim.Optimizer]:
    """
    Build an autoregressive policy. Supports two options via ar_cfg["type"]:
      - "gpt": expects ar_cfg["cls"] to be a GPT-like class (e.g., mingpt.GPT). Uses same keys as build_gpt_policy.
      - "llammagen": expects ar_cfg["cls"] to be a LlamaGen-like class with a similar signature.

    Common optimizer keys: lr, weight_decay, betas, eps
    """
    ar_type = ar_cfg.get("type", "gpt").lower()
    ar_cls = ar_cfg.get("cls", None)
    if ar_cls is None:
        raise ValueError("ar_cfg['cls'] must be provided (class reference for GPT or LlamaGen model)")

    # Instantiate model using provided class and common kwargs
    if ar_type == "gpt":
        policy = ar_cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=ar_cfg.get("n_layer", 12),
            n_head=ar_cfg.get("n_head", 12),
            n_embd=ar_cfg.get("n_embd", 768),
            embd_pdrop=ar_cfg.get("embd_pdrop", 0.0),
            resid_pdrop=ar_cfg.get("resid_pdrop", 0.0),
            attn_pdrop=ar_cfg.get("attn_pdrop", 0.0),
            n_unmasked=ar_cfg.get("n_unmasked", 0),
            class_num=ar_cfg.get("class_num", 1000),
            token_drop=ar_cfg.get("token_drop", 0.1),
            cls_token_number=ar_cfg.get("cls_token_number", 1),
            use_pretrained_codebook=ar_cfg.get("use_pretrained_codebook", False),
            codebook_ckpt_path=ar_cfg.get("codebook_ckpt_path", None),
            n_codebook_embd=ar_cfg.get("n_codebook_embd", ar_cfg.get("n_embd", 768)),
        )
    elif ar_type == "llammagen":
        # For LlamaGen-style classes, we assume a similar constructor; users can pass extra kwargs via ar_cfg["extra"]
        extra = ar_cfg.get("extra", {})
        policy = ar_cls(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=ar_cfg.get("n_layer", 24),
            n_head=ar_cfg.get("n_head", 16),
            n_embd=ar_cfg.get("n_embd", 1024),
            **extra,
        )
    else:
        raise ValueError(f"Unsupported AR type: {ar_type}")

    policy.train()
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=ar_cfg.get("lr", 2e-4),
        weight_decay=ar_cfg.get("weight_decay", 0.01),
        betas=ar_cfg.get("betas", (0.9, 0.95)),
        eps=ar_cfg.get("eps", 1e-8),
    )
    return policy, optimizer