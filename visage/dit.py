"""Visage-Image DiT: a single-stream transformer that predicts rectified-flow velocity.

Text tokens and image tokens share one sequence and one set of weights in every
block (as in U-ViT); each block is a standard sequential adaLN-Zero DiT block.
Conditioning on the timestep is adaLN; conditioning on the caption is attention
alone (no pooled text vector). Captions are token states from a frozen
Qwen3-0.6B cut to its first 8 layers (sampling.TextEncoder).

Captions are padded "mask-free": each caption is filled with learned
per-position pad embeddings up to the smallest length bucket that fits it, so a
sample never depends on what it was batched with, and attention needs no mask
unless one batch mixes buckets. Learned register tokens are prepended to the
text stream.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HEAD_DIM = 64


@dataclass
class DiTConfig:
    image_size: int = 16  # latent grid (256px image / f16 VAE)
    in_channels: int = 96  # VAE latent channels
    patch_size: int = 1
    width: int = 768
    depth: int = 18
    mlp_ratio: int = 4
    text_dim: int = 1024  # Qwen3-0.6B hidden size
    noise_embed_dims: int = 256
    time_min_freq: float = 0.01
    time_max_freq: float = 100.0
    text_length_buckets: tuple[int, ...] = field(default=(32, 64, 96, 128))
    n_text_registers: int = 16

    def __post_init__(self) -> None:
        self.text_length_buckets = tuple(sorted(int(b) for b in self.text_length_buckets))


class SinusoidalEmbedding(nn.Module):
    def __init__(self, min_freq: float, max_freq: float, dims: int) -> None:
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), dims // 2))
        self.register_buffer("angular_speeds", 2.0 * torch.pi * freqs)

    def forward(self, t: Tensor) -> Tensor:
        return torch.cat([torch.sin(self.angular_speeds * t), torch.cos(self.angular_speeds * t)], dim=-1)


def spiral_rope_tables(
    height: int, width: int, head_dim: int, device, base: float = 10000.0, directions: int = 4
) -> tuple[Tensor, Tensor]:
    """2D "spiral" RoPE on integer latent-cell coordinates: rotary pairs cycle through
    `directions` orientations in [0, pi), each at its own frequency. Returns cos/sin
    tables of shape [1, 1, H*W, head_dim // 2]."""
    pairs = head_dim // 2
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    index = torch.arange(pairs, device=device)
    freq_count = max((pairs + directions - 1) // directions, 1)
    inv_freq = 1.0 / (base ** ((index // directions).float() / max(freq_count - 1, 1)))
    theta = torch.pi * (index % directions).float() / directions
    angles = (coords @ torch.stack((theta.cos(), theta.sin()), dim=-1).T) * inv_freq
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = (r.to(x.dtype) for r in rope)
    first, second = x.unflatten(-1, (-1, 2)).unbind(dim=-1)
    return torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1).flatten(-2)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """SwiGLU: fc1 produces value and gate together."""

    def __init__(self, width: int, ratio: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width * ratio * 2)
        self.fc2 = nn.Linear(width * ratio, width)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(value) * gate)


class Attention(nn.Module):
    """Joint attention over [text, image]; RoPE on image tokens only."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out_proj = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(width // heads, eps=1e-6)
        self.k_norm = nn.RMSNorm(width // heads, eps=1e-6)

    def forward(self, x: Tensor, n_text: int, mask: Tensor | None, rope) -> Tensor:
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q = torch.cat((q[..., :n_text, :], apply_rope(q[..., n_text:, :], rope)), dim=-2)
        k = torch.cat((k[..., :n_text, :], apply_rope(k[..., n_text:, :], rope)), dim=-2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class Block(nn.Module):
    def __init__(self, width: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.attn = Attention(width, width // HEAD_DIM)
        self.mlp = MLP(width, mlp_ratio)
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x: Tensor, cond: Tensor, n_text: int, mask: Tensor | None, rope) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod(cond).chunk(6, dim=-1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1), n_text, mask, rope)
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class VisageDiT(nn.Module):
    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = c = config
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(c.time_min_freq, c.time_max_freq, c.noise_embed_dims),
            nn.Linear(c.noise_embed_dims, c.width),
            nn.SiLU(),
            nn.Linear(c.width, c.width),
        )
        self.text_proj = nn.Linear(c.text_dim, c.width)
        # Per-position pads: the text stream has no RoPE, so one shared pad vector
        # could only act as a single register copied N times.
        self.text_pad_embed = nn.Parameter(torch.randn(c.text_length_buckets[-1], c.width) * 0.02)
        self.text_registers = nn.Parameter(torch.randn(c.n_text_registers, c.width) * 0.02)
        self.patch_embed = nn.Conv2d(c.in_channels, c.width, c.patch_size, stride=c.patch_size)
        self.blocks = nn.ModuleList(Block(c.width, c.mlp_ratio) for _ in range(c.depth))
        self.final_norm = nn.LayerNorm(c.width, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(c.width, 2 * c.width))
        self.out_proj = nn.Linear(c.width, c.in_channels * c.patch_size**2)
        for layer in (self.final_mod[-1], self.out_proj):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @torch.compiler.disable
    def prepare_text(self, text: Tensor, text_mask: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Fill each caption with learned pads up to its bucket; prepend registers.
        Returns the text stream and a key mask (None when the batch shares one bucket).
        Assumes right padding."""
        batch, length, _ = text.shape
        lengths = (
            torch.full((batch,), length, device=text.device)
            if text_mask is None
            else text_mask.to(text.device, torch.bool).sum(dim=1)
        )
        buckets = torch.tensor(self.config.text_length_buckets, device=text.device)
        if int(lengths.max()) > buckets[-1]:
            raise ValueError(f"caption of {int(lengths.max())} tokens exceeds the largest bucket {int(buckets[-1])}")
        sample_bucket = buckets[torch.searchsorted(buckets, lengths)]
        width = int(sample_bucket.max())
        if length < width:
            text = torch.cat([text, text.new_zeros(batch, width - length, text.shape[-1])], dim=1)
        text = text[:, :width]
        positions = torch.arange(width, device=text.device)
        is_pad = positions[None] >= lengths[:, None]
        text = torch.where(is_pad.unsqueeze(-1), self.text_pad_embed[:width].to(text.dtype)[None], text)
        keep = positions[None] < sample_bucket[:, None]
        mask = None if bool(keep.all()) else keep
        registers = self.text_registers.to(text.dtype).expand(batch, -1, -1)
        text = torch.cat([registers, text], dim=1)
        if mask is not None:
            mask = torch.cat([mask.new_ones(batch, registers.shape[1]), mask], dim=1)
        return text, mask

    def forward(
        self, x: Tensor, t: Tensor, text: Tensor, text_mask: Tensor | None = None, return_features_at=None
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """x: noisy latent [B, C, H, W]; t: [B, 1] in [0, 1] (1 = noise);
        text: caption token states [B, L, text_dim]. Returns velocity (noise - data).
        With `return_features_at` (block indices), also returns the image tokens after
        each of those blocks as [B, width, H/p, W/p] maps."""
        p = self.config.patch_size
        grid_h, grid_w = x.shape[-2] // p, x.shape[-1] // p
        cond = self.time_embed(t)
        txt, mask = self.prepare_text(self.text_proj(text), text_mask)
        img = self.patch_embed(x).flatten(2).transpose(1, 2)
        n_text = txt.shape[1]
        if mask is not None:
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], img.shape[1])], dim=1)[:, None, None, :]
        rope = spiral_rope_tables(grid_h, grid_w, HEAD_DIM, x.device)
        h = torch.cat([txt, img], dim=1)
        features = {}
        for i, block in enumerate(self.blocks):
            h = block(h, cond, n_text, mask, rope)
            if return_features_at is not None and i in return_features_at:
                features[i] = h[:, n_text:].transpose(1, 2).reshape(x.shape[0], -1, grid_h, grid_w)
        h = h[:, n_text:]
        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        h = self.out_proj(modulate(self.final_norm(h), shift, scale))
        h = h.view(x.shape[0], grid_h, grid_w, -1, p, p)
        velocity = h.permute(0, 3, 1, 4, 2, 5).reshape(x.shape[0], -1, grid_h * p, grid_w * p)
        if return_features_at is None:
            return velocity
        return velocity, [features[i] for i in return_features_at]
