import torch
import torch.nn.functional as F
from typing import Optional


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean()
    std = x.std() + eps
    return (x - mean) / std


def reinforce_update(policy, optimizer, tokens: torch.LongTensor, logprobs: torch.FloatTensor,
                     rewards: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None,
                     entropy_coef: float = 0.0, max_grad_norm: Optional[float] = None) -> float:
    """
    Minimal REINFORCE: loss = -E[sum_t logp_t * adv], adv = standardized(reward).
    tokens [B, T], logprobs [B, T], rewards [B]
    """
    policy.train()
    B, T = tokens.shape
    adv = standardize(rewards.detach())  # [B]
    adv = adv.view(B, 1).expand_as(logprobs)

    if mask is not None:
        logprobs = logprobs.masked_fill(~mask, 0.0)
        adv = adv.masked_fill(~mask, 0.0)

    reinforce_loss = -(logprobs * adv).sum(dim=1).mean()

    # Optional entropy bonus: approximate with categorical entropy from current logits if needed.
    loss = reinforce_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.detach().cpu().item())


def compute_sequence_logprob(policy, tokens: torch.LongTensor) -> torch.Tensor:
    """
    Compute sequence-level log-prob under current policy via teacher-forcing.
    tokens: [B, T]
    Returns: logp_seq [B]
    """
    device = next(policy.parameters()).device
    tokens = tokens.to(device)
    B, T = tokens.shape
    if T <= 1:
        return torch.zeros(B, device=device)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    # Dummy class token input (unused minimal path). Shape [B, 1]
    idx_cls = torch.zeros(B, 1, dtype=torch.long, device=device)
    logits, _ = policy((inputs, idx_cls))
    # Keep only token positions (drop class positions if present at the beginning)
    logits_token = logits[:, -inputs.size(1):, :]
    log_probs = F.log_softmax(logits_token, dim=-1)
    token_logp = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    logp_seq = token_logp.sum(dim=1)
    return logp_seq


def grpo_update(policy, optimizer, tokens: torch.LongTensor, old_logp_seq: torch.Tensor, advantages: torch.Tensor,
                clip_range: float = 0.2, max_grad_norm: Optional[float] = None) -> float:
    """
    Group Relative Policy Optimization (sequence-level):
      loss = -mean(min(ratio * adv, clip(ratio, 1-ε, 1+ε) * adv))
    tokens: [N, T]
    old_logp_seq: [N] (detached rollout log-prob sums)
    advantages: [N] (group-normalized advantages)
    """
    policy.train()
    new_logp_seq = compute_sequence_logprob(policy, tokens)
    ratio = torch.exp(new_logp_seq - old_logp_seq)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -torch.mean(torch.minimum(unclipped, clipped))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.detach().cpu().item())


def groupwise_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize rewards within each group of size `group_size` to produce advantages.
    rewards: [N], where N is divisible by group_size and groups are contiguous.
    returns: advantages [N]
    """
    assert rewards.numel() % group_size == 0, "rewards size must be divisible by group_size"
    N = rewards.numel()
    num_groups = N // group_size
    rewards = rewards.view(num_groups, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True) + eps
    adv = (rewards - mean) / std
    return adv.view(N)


