"""Visage-Image VAE: f16, 96 latent channels, 256px faces.

Encoder: LiteVAE-style. A 3-level Haar wavelet pyramid, a small conv extractor
per level, aggregation on the f8 grid (with self-attention), one DC-AE downsample
to f16 (with self-attention). Two parameter-free shortcuts (Qwen-Image-VAE-2.0):
a global skip that folds the pixels straight into the deepest stage, and a
latent skip that adds a channel-group mean of the folded pixels to the latent.

Decoder: a ViT on the 16x16 latent grid, each token split 2x2 into f8 tokens for
a second ViT stage, an 8x8 linear patch head, and two small residual conv
refiners that work across patch seams.

The latent is deterministic (posterior mean, no KL). ``latent_mean`` /
``latent_std`` hold dataset statistics: ``encode`` returns normalized latents
(what the DiT trains on) and ``decode`` takes them.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class VAEConfig:
    latent_channels: int = 96
    encoder_widths: tuple[int, int, int, int] = (120, 240, 376, 496)
    encoder_attention: int = 2  # attention blocks after the f8 aggregator
    encoder_attention_f16: int = 2  # attention blocks after the f16 downsample
    decoder_width: int = 768
    decoder_depth: int = 6  # f16 stage
    decoder_heads: int = 12
    hires_width: int = 768
    hires_depth: int = 6  # f8 stage
    hires_heads: int = 12
    refiner_width: int = 48
    refiner2_width: int = 96
    layer_scale: float = 1e-4


# ---------------------------------------------------------------- encoder


def _groups(channels: int) -> int:
    divisors = [g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0]
    wide = [g for g in divisors if channels // g >= 8]
    return wide[0] if wide else divisors[-1]


class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)


def _blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(ConvResBlock(channels, channels) for _ in range(count)))


class SpatialAttention(nn.Module):
    """Residual multi-head self-attention over the pixels of a feature map."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y = self.attn(y, y, y, need_weights=False)[0]
        return x + self.proj(y).transpose(1, 2).reshape(b, c, h, w)


def _attention(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(SpatialAttention(channels) for _ in range(count)))


class HaarDWT(nn.Module):
    """One orthonormal Haar level: [B, C, H, W] -> [B, 4C, H/2, W/2], LL band first."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        low, high = torch.tensor((1.0, 1.0)) / 2**0.5, torch.tensor((1.0, -1.0)) / 2**0.5
        bands = torch.stack([r[:, None] * c[None, :] for r, c in ((low, low), (low, high), (high, low), (high, high))])
        self.register_buffer(
            "kernel", bands[None].expand(channels, 4, 2, 2).reshape(channels * 4, 1, 2, 2).contiguous()
        )

    def forward(self, x: Tensor) -> Tensor:
        # fp32: detail bands are small differences that bf16 would quantize away.
        with torch.autocast(device_type=x.device.type, enabled=False):
            bands = F.conv2d(x.float(), self.kernel.float(), stride=2, groups=self.channels)
        bands = bands.to(self.kernel.dtype)  # a bf16 model takes bf16 bands
        b, _, h, w = bands.shape
        return bands.view(b, self.channels, 4, h, w).transpose(1, 2).reshape(b, 4 * self.channels, h, w)


def fold_mean(x: Tensor, factor: int, out_channels: int) -> Tensor:
    """Pixel-unshuffle by `factor`, then average channel groups down to `out_channels`."""
    folded = F.pixel_unshuffle(x, factor)
    b, c, h, w = folded.shape
    return folded.view(b, out_channels, c // out_channels, h, w).mean(dim=2)


class DCAEDown(nn.Module):
    """2x downsample: conv + pixel-unshuffle, plus a parameter-free fold_mean shortcut."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_unshuffle(self.conv(x), 2) + fold_mean(x, 2, self.channels)


class WaveletEncoder(nn.Module):
    def __init__(
        self, latent_channels: int, widths: tuple[int, int, int, int], attention: int = 2, attention_f16: int = 2
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.dwt = HaarDWT(3)
        self.extractors = nn.ModuleList()
        self.reducers = nn.ModuleList()
        for level, width in enumerate(widths[:3], start=1):
            self.extractors.append(nn.Sequential(nn.Conv2d(12, width, 3, padding=1), _blocks(width, 2)))
            stride = 2 ** (3 - level)  # bring level `level` down to the f8 grid
            self.reducers.append(nn.AvgPool2d(stride) if stride > 1 else nn.Identity())
        agg = widths[3]
        self.aggregate = nn.Sequential(nn.Conv2d(sum(widths[:3]), agg, 1), _blocks(agg, 4), _attention(agg, attention))
        self.down16 = nn.Sequential(DCAEDown(agg), _blocks(agg, 2), _attention(agg, attention_f16))
        self.global_skip = nn.Conv2d(3 * 16**2, agg, 1)
        self.to_moments = nn.Sequential(
            nn.GroupNorm(_groups(agg), agg), nn.SiLU(), nn.Conv2d(agg, 2 * latent_channels, 1)
        )

    def forward(self, image: Tensor) -> Tensor:
        features, low = [], image
        for extractor, reducer in zip(self.extractors, self.reducers, strict=True):
            bands = self.dwt(low)
            low = bands[:, :3]
            features.append(reducer(extractor(bands)))
        h = self.down16(self.aggregate(torch.cat(features, dim=1)))
        h = h + self.global_skip(F.pixel_unshuffle(image, 16))
        # Only the mean half of the moments is used (deterministic latent).
        mean = self.to_moments(h)[:, : self.latent_channels]
        return mean + fold_mean(image, 16, self.latent_channels)


# ---------------------------------------------------------------- decoder


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).type_as(x)


def axial_rope_tables(head_dim: int, grid: int, device, theta: float = 100.0) -> tuple[Tensor, Tensor]:
    quarter = head_dim // 4
    freqs = 1.0 / (theta ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    angles = torch.arange(grid, dtype=torch.float32)[:, None] * freqs[None, :]
    rows = angles[:, None, :].expand(grid, grid, quarter)
    cols = angles[None, :, :].expand(grid, grid, quarter)
    table = torch.cat([rows, cols], dim=-1).reshape(grid * grid, head_dim // 2)
    return table.cos().to(device), table.sin().to(device)


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    first, second = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, n, w = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = _rope(self.q_norm(q), cos, sin), _rope(self.k_norm(k), cos, sin)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w))


class SwiGLU(nn.Module):
    def __init__(self, width: int, multiple: int = 64) -> None:
        super().__init__()
        hidden = multiple * ((int(8 * width / 3) + multiple - 1) // multiple)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, layer_scale: float) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(width), Attention(width, heads)
        self.norm2, self.mlp = RMSNorm(width), SwiGLU(width)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(width))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(width))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x), cos, sin)
        return x + self.gamma2 * self.mlp(self.norm2(x))


class ViTDecoder(nn.Module):
    def __init__(self, c: VAEConfig) -> None:
        super().__init__()
        self.from_latent = nn.Linear(c.latent_channels, c.decoder_width)
        self.blocks = nn.ModuleList(
            Block(c.decoder_width, c.decoder_heads, c.layer_scale) for _ in range(c.decoder_depth)
        )
        self.norm = RMSNorm(c.decoder_width)
        # Split each f16 token into its 2x2 f8 children.
        self.to_hires = nn.Linear(c.decoder_width, 4 * c.hires_width)
        self.hires_blocks = nn.ModuleList(
            Block(c.hires_width, c.hires_heads, c.layer_scale) for _ in range(c.hires_depth)
        )
        self.hires_norm = RMSNorm(c.hires_width)
        self.head = nn.Linear(c.hires_width, 8 * 8 * 3)
        self.refiner = nn.Sequential(
            nn.Conv2d(3, c.refiner_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.refiner_width, c.refiner_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.refiner_width, 3, 3, padding=1),
        )
        self.refiner2 = nn.Sequential(
            nn.Conv2d(3, c.refiner2_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.refiner2_width, c.refiner2_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c.refiner2_width, 3, 3, padding=1),
        )
        self.head_dims = (c.decoder_width // c.decoder_heads, c.hires_width // c.hires_heads)
        self.apply(self._init)
        # Start at mid-grey with identity refiners.
        for layer in (self.head, self.refiner[-1], self.refiner2[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, latent: Tensor) -> Tensor:
        b, _, h, w = latent.shape
        if h != w:
            raise ValueError("latent grid must be square")
        x = self.from_latent(latent.flatten(2).transpose(1, 2))
        cos, sin = axial_rope_tables(self.head_dims[0], h, latent.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.to_hires(self.norm(x)).transpose(1, 2).reshape(b, -1, h, w)
        x = F.pixel_shuffle(x, 2).flatten(2).transpose(1, 2)
        cos, sin = axial_rope_tables(self.head_dims[1], 2 * h, latent.device)
        for block in self.hires_blocks:
            x = block(x, cos, sin)
        x = self.head(self.hires_norm(x)).transpose(1, 2).reshape(b, 3 * 64, 2 * h, 2 * w)
        image = F.pixel_shuffle(x, 8)
        image = image + self.refiner(image)
        image = image + self.refiner2(image)
        return torch.tanh(image)


# ---------------------------------------------------------------- autoencoder


class VisageVAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.wavelet_encoder = WaveletEncoder(
            config.latent_channels, config.encoder_widths, config.encoder_attention, config.encoder_attention_f16
        )
        self.vit_decoder = ViTDecoder(config)
        self.register_buffer("latent_mean", torch.zeros(config.latent_channels))
        self.register_buffer("latent_std", torch.ones(config.latent_channels))

    def encode_raw(self, image: Tensor) -> Tensor:
        """[-1, 1] images [B, 3, H, W] (H, W multiples of 16) -> raw latents."""
        return self.wavelet_encoder(image.to(self.latent_mean.dtype))

    def decode_raw(self, latent: Tensor) -> Tensor:
        return self.vit_decoder(latent.to(self.latent_mean.dtype))

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.latent_mean[None, :, None, None]) / self.latent_std[None, :, None, None]

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.latent_std[None, :, None, None] + self.latent_mean[None, :, None, None]

    def encode(self, image: Tensor) -> Tensor:
        """Images -> normalized latents (zero mean, unit std per channel)."""
        return self.normalize(self.encode_raw(image))

    def decode(self, latent: Tensor) -> Tensor:
        """Normalized latents -> [-1, 1] images."""
        return self.decode_raw(self.denormalize(latent))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Training path: (reconstruction, raw latent)."""
        latent = self.encode_raw(image)
        return self.decode_raw(latent), latent
