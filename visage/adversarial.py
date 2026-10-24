"""Latent adversarial post-training (LADD, arXiv:2403.12015; APT, arXiv:2501.08316).

The critic is a frozen copy of the DiT with small trainable heads on a few of its
blocks. It judges noised clean-latent predictions against noised real latents, so
nothing is decoded to pixels. Only the heads train: a fully trainable critic on a
face corpus at batch 128 overfits long before it teaches the generator anything.
The flow loss stays on next to the adversarial term, which keeps multi-step
sampling intact.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .dit import VisageDiT


class CriticHead(nn.Module):
    """Two convolutions over one block's image tokens, FiLM-conditioned on t and the
    pooled caption. Outputs a logit per latent position, since texture defects are local."""

    def __init__(self, feature_dim: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, feature_dim, affine=False)
        self.to_hidden = nn.Conv2d(feature_dim, hidden, 1)
        self.film = nn.Linear(cond_dim, 2 * hidden)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 1, 1)
        # Zero init: the critic starts with no opinion instead of a random one.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.to_hidden(self.norm(features)) * (1 + scale) + shift)
        return self.out(F.silu(self.conv(h)))


class LatentCritic(nn.Module):
    """Trainable heads on blocks `layers` of a frozen DiT `backbone`."""

    def __init__(self, backbone: VisageDiT, layers, hidden: int = 256, center_latents: bool = True) -> None:
        super().__init__()
        # A plain attribute, so the frozen backbone stays out of state_dict() and the optimizer.
        object.__setattr__(self, "backbone", backbone.requires_grad_(False).eval())
        self.layers = tuple(layers)
        self.center_latents = center_latents
        c = backbone.config
        cond_dim = 256
        self.cond = nn.Sequential(nn.Linear(c.text_dim + 128, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.heads = nn.ModuleList(CriticHead(c.width, cond_dim, hidden) for _ in self.layers)

    def conditioning(self, t: Tensor, text: Tensor, text_mask: Tensor | None) -> Tensor:
        weights = torch.ones_like(text[..., :1]) if text_mask is None else text_mask.to(text.dtype)[..., None]
        pooled = (text * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        freqs = torch.exp(-math.log(10000.0) * torch.arange(64, device=t.device) / 64)
        angles = t.float().view(-1, 1) * freqs * 1000.0
        return self.cond(torch.cat([pooled.float(), angles.sin(), angles.cos()], dim=-1))

    def forward(self, latent: Tensor, t: Tensor, text: Tensor, text_mask: Tensor | None) -> list[Tensor]:
        if self.center_latents:
            # Per-image, per-channel mean removed from real and fake alike. Without it the
            # generator fools the critic by shifting colour: in the first 50 steps of an
            # uncentred run, backgrounds went olive and skin yellow-green. Colour is already
            # supervised by the flow loss.
            latent = latent - latent.mean(dim=(-2, -1), keepdim=True)
        # No no_grad: the generator's gradient reaches its sample through these features.
        _, features = self.backbone(latent, t.view(-1, 1), text, text_mask, return_features_at=self.layers)
        cond = self.conditioning(t, text, text_mask)
        return [head(f.float(), cond) for head, f in zip(self.heads, features, strict=True)]


def sample_critic_t(n: int, device, mean: float = -1.0, std: float = 1.0) -> Tensor:
    """Logit-normal noise level for the critic. Mean -1 puts the median at t = 0.27, where
    the critic judges local detail; higher t reaches composition."""
    return torch.sigmoid(torch.randn(n, device=device) * std + mean)


def generator_loss(fake: list[Tensor]) -> Tensor:
    """Non-saturating, averaged over heads and positions."""
    return torch.stack([F.softplus(-logit).mean() for logit in fake]).mean()


def critic_loss(real: list[Tensor], fake: list[Tensor]) -> Tensor:
    return (
        torch.stack([F.softplus(-logit).mean() for logit in real]).mean()
        + torch.stack([F.softplus(logit).mean() for logit in fake]).mean()
    )


def approximate_r1(critic: LatentCritic, real_latent: Tensor, t, text, text_mask, real_logits, sigma: float) -> Tensor:
    """APT's approximated R1: the change in the critic's output under a small perturbation
    of the real input. One extra forward instead of a double backward."""
    perturbed = critic(real_latent + sigma * torch.randn_like(real_latent), t, text, text_mask)
    return torch.stack([(a - b).square().mean() for a, b in zip(real_logits, perturbed, strict=True)]).mean()
