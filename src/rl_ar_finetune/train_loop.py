import torch
from typing import Dict, Any

from .builder import build_tokenizer_frozen, build_gpt_policy, build_ar_policy
from .sampling import sample_codes_with_logprobs, sample_multiple_generations
from .rewards import reconstruction_reward
from .optimize import reinforce_update, grpo_update, groupwise_advantages


def train_rl_loop(cfg: Dict[str, Any]):
    """
    Minimal RL loop for AR policy with reconstruction/perceptual rewards.

    cfg keys (suggested):
      - vq_model: a loaded IBQ/VQ model instance (already has weights)
      - gpt_cls: class (e.g., mingpt.GPT)
      - gpt: dict (hyperparameters for GPT & optimizer)
      - block_size: int (max sequence length)
      - dataloader: PyTorch DataLoader yielding images in tokenizer's expected range
      - rollout: {max_len, temperature, top_k, top_p}
      - reward: {type: "l2"|"l1"|"lpips", normalize: bool}
      - train: {iters, log_every, save_every, ckpt_path, device, max_grad_norm}
    """
    device = torch.device(cfg.get("train", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 1) frozen tokenizer
    tok = build_tokenizer_frozen(cfg["vq_model"]).vq.to(device)
    tokenizer = build_tokenizer_frozen(tok)

    # 2) policy (supports GPT or LlamaGen)
    vocab_size = tokenizer.codebook_size
    if "ar" in cfg:
        policy, optimizer = build_ar_policy(vocab_size, cfg["block_size"], cfg["ar"]) 
    else:
        # backward-compat: GPT path
        policy, optimizer = build_gpt_policy(cfg["gpt_cls"], vocab_size, cfg["block_size"], cfg["gpt"]) 
    policy.to(device)

    iters = cfg.get("train", {}).get("iters", 10000)
    log_every = cfg.get("train", {}).get("log_every", 50)
    save_every = cfg.get("train", {}).get("save_every", 1000)
    ckpt_path = cfg.get("train", {}).get("ckpt_path", None)
    max_grad_norm = cfg.get("train", {}).get("max_grad_norm", None)

    rollout_cfg = cfg.get("rollout", {})
    grpo = cfg.get("grpo", {"enabled": True, "num_generations": 4, "clip_range": 0.2})
    reward_cfg = cfg.get("reward", {})

    step = 0
    policy.train()
    for it in range(iters):
        for images in cfg["dataloader"]:
            images = images.to(device)

            B = images.size(0)
            start_tokens = torch.empty((B, 0), dtype=torch.long, device=device)

            if grpo.get("enabled", True):
                # GRPO group sampling
                G = int(grpo.get("num_generations", 4))
                tokens, step_logps = sample_multiple_generations(
                    policy,
                    start_tokens,
                    rollout_cfg.get("max_len", cfg["block_size"]),
                    G,
                    rollout_cfg.get("temperature", 1.0),
                    rollout_cfg.get("top_k", None),
                    rollout_cfg.get("top_p", None),
                )
                with torch.no_grad():
                    rec = tokenizer.decode_code(tokens)
                    images_tiled = images.repeat_interleave(G, dim=0)
                    rewards = reconstruction_reward(images_tiled, rec, reward_cfg.get("type", "l2"), reward_cfg.get("normalize", True))
                    old_logp_seq = step_logps.sum(dim=1).detach()
                    adv = groupwise_advantages(rewards, group_size=G)
                loss = grpo_update(policy, optimizer, tokens, old_logp_seq, adv, clip_range=float(grpo.get("clip_range", 0.2)), max_grad_norm=max_grad_norm)
            else:
                # REINFORCE baseline
                tokens, step_logps = sample_codes_with_logprobs(
                    policy,
                    start_tokens,
                    rollout_cfg.get("max_len", cfg["block_size"]),
                    rollout_cfg.get("temperature", 1.0),
                    rollout_cfg.get("top_k", None),
                    rollout_cfg.get("top_p", None),
                )
                with torch.no_grad():
                    rec = tokenizer.decode_code(tokens)
                    rewards = reconstruction_reward(images, rec, reward_cfg.get("type", "l2"), reward_cfg.get("normalize", True))
                loss = reinforce_update(policy, optimizer, tokens, step_logps, rewards, mask=None, entropy_coef=0.0, max_grad_norm=max_grad_norm)

            step += 1
            if step % log_every == 0:
                print(f"[RL-AR-GRPO] step={step} loss={loss:.4f} reward_mean={rewards.mean().item():.4f}")

            if ckpt_path and step % save_every == 0:
                state = {
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                torch.save(state, ckpt_path)

            if step >= iters:
                return


