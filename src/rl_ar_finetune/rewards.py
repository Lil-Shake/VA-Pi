import torch
import torch.nn.functional as F
from typing import Optional

try:
    import lpips  # optional perceptual metric
except Exception:
    lpips = None


@torch.no_grad()
def reconstruction_reward(x: torch.Tensor, x_rec: torch.Tensor, loss_type: str = "l2", normalize: bool = True) -> torch.Tensor:
    """
    Compute negative reconstruction loss as reward. Inputs are expected to be in same range.
    loss_type: "l2" | "l1" | "lpips" (requires pip install lpips)
    returns: rewards [B], where higher is better
    """
    assert x.shape == x_rec.shape
    B = x.shape[0]

    if loss_type == "l2":
        loss = F.mse_loss(x_rec, x, reduction="none").flatten(1).mean(dim=1)
    elif loss_type == "l1":
        loss = F.l1_loss(x_rec, x, reduction="none").flatten(1).mean(dim=1)
    elif loss_type == "lpips":
        assert lpips is not None, "lpips package is not installed"
        net = lpips.LPIPS(net='vgg').to(x.device)
        scores = net(x, x_rec).flatten()
        loss = scores
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    rewards = -loss
    if normalize:
        mean = rewards.mean()
        std = rewards.std() + 1e-6
        rewards = (rewards - mean) / std
    return rewards


