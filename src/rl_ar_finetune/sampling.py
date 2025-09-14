import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def sample_codes_with_logprobs(policy, start_tokens: torch.LongTensor, max_len: int, temperature: float = 1.0,
                               top_k: int = None, top_p: float = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Autoregressively sample tokens from `policy`, returning tokens and per-step log-probs.
    start_tokens: [B, T0]  (pure token ids sequence, no cls embed injection here)
    returns tokens [B, T], logprobs [B, T]
    """
    device = next(policy.parameters()).device
    x = start_tokens.to(device)
    B = x.size(0)
    tokens = []
    logps = []

    for t in range(max_len):
        # GPT.forward expects (idx, idx_cls) if class tokens used. For minimal path, we pass (x, None) style input.
        logits, _ = policy((x, torch.zeros(B, 1, dtype=torch.long, device=device)))
        step_logits = logits[:, -1, :]
        step_logits = step_logits / max(temperature, 1e-8)

        if top_k is not None or top_p is not None:
            # top-k / nucleus
            probs = F.softmax(step_logits, dim=-1)
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                probs_masked = torch.zeros_like(probs).scatter(-1, topk_idx, topk_vals)
                probs = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > float(top_p)
                mask[..., 0] = False
                sorted_probs[mask] = 0
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
            logp = torch.log(torch.gather(probs, -1, next_token).clamp_min(1e-12))
        else:
            probs = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            logp = torch.log(torch.gather(probs, -1, next_token).clamp_min(1e-12))

        x = torch.cat([x, next_token], dim=1)
        tokens.append(next_token)
        logps.append(logp)

    tokens = torch.cat(tokens, dim=1)
    logps = torch.cat(logps, dim=1).squeeze(-1)
    return tokens, logps


@torch.no_grad()
def sample_multiple_generations(policy, start_tokens: torch.LongTensor, max_len: int, num_generations: int,
                                temperature: float = 1.0, top_k: int = None, top_p: float = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Repeat sampling for each input example `num_generations` times (GRPO groups).
    Returns stacked tokens/logprobs with shape [B*num_generations, T] and [B*num_generations, T].
    """
    B = start_tokens.size(0)
    all_tokens = []
    all_logps = []
    for _ in range(num_generations):
        t, l = sample_codes_with_logprobs(policy, start_tokens, max_len, temperature, top_k, top_p)
        all_tokens.append(t)
        all_logps.append(l)
    tokens = torch.cat(all_tokens, dim=0)
    logps = torch.cat(all_logps, dim=0)
    return tokens, logps


