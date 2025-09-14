RL fine-tuning for Autoregressive (AR) code models (TokLIP tokenizers)

Context
- Goal: reduce the train-test gap between teacher-forcing training and autoregressive decoding by fine-tuning a GPT-like AR policy directly on the tokens that decode well through the (frozen) tokenizer/decoder.
- Reward: image-domain losses between the input image and the reconstruction obtained by decoding the sampled token sequence. E.g., reconstruction L2/L1, PSNR, SSIM, LPIPS (perceptual), etc. The policy is optimized to minimize reconstruction/perceptual loss (i.e., maximize negative loss as reward).

High-level Pipeline
1) Frozen tokenizer: Use the existing IBQ/VQ tokenizer/decoder from `src/tokenizer/ibq_modules/ibqgan.py`. Do not update its parameters.
2) AR policy: Use `src/tokenizer/ibq_modules/transformer/mingpt.py::GPT` as the autoregressive policy over codebook indices.
3) Rollout: For each image, start from a condition (e.g., class token or an empty prefix), sample a full token sequence with the policy, decode it to an image via the frozen decoder, compute reward = -reconstruction_loss(input, decoded).
4) Optimize: Use REINFORCE or PPO to update the policy with advantages derived from rewards. Minimal viable route starts with REINFORCE.

Minimal Code Structure (new module)
- New package: `src/rl_ar_finetune/`
  - `builder.py`: builders for frozen tokenizer and GPT policy.
  - `sampling.py`: autoregressive sampling that also returns per-step log-probs.
  - `rewards.py`: reconstruction/perceptual rewards; supports pluggable perceptual metrics.
  - `optimize.py`: REINFORCE/PPO update steps with standardization of advantages.
  - `train_loop.py`: end-to-end training loop composing the above pieces.

Key APIs (summary)
- build_toklip_tokenizer(init_args, ckpt_path=None, device=None) -> FrozenTokenizer
  - Build the TokLIP tokenizer (IBQ/VQ) and wrap it frozen: exposes encode(images)->indices; decode_code(indices)->images; codebook_size.
  - If you already have a loaded IBQ/VQ model instance, use `build_tokenizer_frozen(vq_model)` instead.
- build_ar_policy(vocab_size, block_size, ar_cfg) -> (policy, optimizer)
  - Build the AR policy. `ar_cfg["type"]` supports "gpt" or "llammagen".
  - Provide the model class via `ar_cfg["cls"]`. For GPT, accepts typical GPT kwargs; for LlamaGen-style, pass extra args in `ar_cfg["extra"]`.
- build_gpt_policy(cfg, vocab_size, block_size) -> (policy, optimizer)
  - Instantiate `mingpt.GPT`. Optionally initialize token embeddings from tokenizer codebook (`use_pretrained_codebook=True`).
- sample_codes_with_logprobs(policy, start_tokens, max_len, temperature, top_k, top_p) -> (tokens, logprobs)
  - Greedy/top-k/top-p sampling over the vocabulary, returning both sampled tokens and their log-probs per-step.
- reconstruction_reward(input_images, decoded_images, type="l2"|"l1"|"lpips", normalize=True) -> rewards
  - Computes negative loss as reward (higher is better). Support LPIPS if available.
- reinforce_update(policy, optimizer, tokens, logprobs, rewards, mask=None, entropy_coef=0.0, max_grad_norm=None)
  - Minimal REINFORCE update: advantages = standardized(rewards); loss = -E[sum_t logp_t * adv] - entropy_bonus.
- train_rl_loop(cfg)
  - Orchestrates data loading, rollout, reward, and optimization, with logging and checkpointing.

Assumptions and Shapes
- Token sequences: LongTensor [batch, seq_len], values in [0, vocab_size-1].
- Start tokens: may include a class token via LabelEmbedder in GPT; pass as a tuple (idx, idx_cls) if used.
- Rewards: scalar per example; for simplicity, broadcast to all steps in that sequence during REINFORCE.

How to Extend
- PPO: add old_logprobs buffer, compute ratios and clipped surrogate loss; optionally add value head for GAE.
- Conditional generation: add text features or other conditioning by concatenating condition tokens or adding cross-attention to GPT.
- Mixed rewards: combine reconstruction and perceptual scores with weights.

Safety / Performance Notes
- Decode in small batches to control VRAM.
- Use AMP where safe for reward networks (e.g., LPIPS), but keep tokenizer decoder numerically stable.
- Clip gradients on the policy.

Entry Point
- Define a small script (or notebook) to import `train_rl_loop(cfg)` from `src/rl_ar_finetune/train_loop.py` and pass dataset + hyperparameters.

Example: Using TokLIP Tokenizer + GPT or LlamaGen AR
```python
from rl_ar_finetune.builder import build_toklip_tokenizer, build_ar_policy
from rl_ar_finetune.train_loop import train_rl_loop
from tokenizer.ibq_modules.ibqgan import IBQ  # ensure importable

# 1) Build TokLIP tokenizer (frozen)
tok_init = {
    "ddconfig": {...},
    "lossconfig": {...},
    "n_embed": 8192,
    "embed_dim": 256,
    "l2_normalize": False,
    "use_ema": False,
    "stage": None,
}
tokenizer = build_toklip_tokenizer(tok_init, ckpt_path="/path/to/toklip_vq.ckpt", device="cuda")

# 2) Prepare cfg for training
cfg = {
  "vq_model": tokenizer.vq,  # or directly pass tokenizer and adapt train_loop
  # Choose AR policy:
  # Option A: GPT
  "ar": {"type": "gpt", "cls": mingpt.GPT, "n_layer": 12, "n_head": 12, "n_embd": 768, "lr": 2e-4},
  # Option B: LlamaGen-like
  # "ar": {"type": "llammagen", "cls": MyLlamaGen, "n_layer": 24, "n_head": 16, "n_embd": 1024, "extra": {"rope": True}},
  "block_size": 1024,
  "dataloader": my_image_loader,  # images already normalized to tokenizer's expected range
  "rollout": {"max_len": 1024, "temperature": 1.0},
  "reward": {"type": "l2", "normalize": True},
  "grpo": {"enabled": True, "num_generations": 4, "clip_range": 0.2},
  "train": {"iters": 10000, "device": "cuda", "log_every": 50, "save_every": 1000, "ckpt_path": "./grpo.pt"},
}
train_rl_loop(cfg)
```


