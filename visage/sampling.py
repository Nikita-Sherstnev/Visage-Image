"""Text encoding and rectified-flow sampling.

Convention: x_t = (1 - t) * x_data + t * noise, the model predicts noise - x_data,
and sampling integrates from t = 1 (noise) to t = 0 (data).
"""

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

TEXT_ENCODER = "Qwen/Qwen3-0.6B"
TEXT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TEXT_LAYER = 8
TEXT_MAX_LENGTH = 128


class TextEncoder(nn.Module):
    """Frozen Qwen3-0.6B cut to its first `layer` decoder layers (no final norm), so the
    output is hidden_states[layer] of the full model. Returns per-token states and the mask.

    A fixed prefix runs ahead of the caption and its states are dropped: in a causal LM
    the first tokens act as an attention sink with outsized norms, and the prefix absorbs
    that instead of the caption's first word. Padding is on the right and attention is
    causal, so a caption's states do not depend on what it is batched with.
    """

    PREFIX = "Describe the image."

    def __init__(
        self,
        name: str = TEXT_ENCODER,
        layer: int = TEXT_LAYER,
        max_length: int = TEXT_MAX_LENGTH,
        revision: str | None = TEXT_REVISION,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        # What text_encoder/config.json records, so load_text_encoder rebuilds this exact encoder.
        self.info = {"name": name, "layer": layer, "max_length": max_length, "revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModel.from_pretrained(name, revision=revision, dtype=dtype)
        model.layers = model.layers[:layer]
        model.config.num_hidden_layers = layer
        model.config.layer_types = model.config.layer_types[:layer]
        model.norm = nn.Identity()
        self.model = model.requires_grad_(False).eval()
        self.prefix_ids = self.tokenizer(self.PREFIX, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def forward(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        device = next(self.model.parameters()).device
        ids = [
            x[: self.max_length] for x in self.tokenizer([" " + p for p in prompts], add_special_tokens=False).input_ids
        ]
        n_pre, width = len(self.prefix_ids), max(map(len, ids))
        input_ids = torch.full((len(ids), n_pre + width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(ids), n_pre + width, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : n_pre + len(x)] = torch.tensor(self.prefix_ids + x)
            mask[i, : n_pre + len(x)] = 1
        hidden = self.model(
            input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False
        ).last_hidden_state[:, n_pre:]
        # Captions were cached as float8 e4m3 for training; match that rounding.
        return hidden.float().to(torch.float8_e4m3fn).half(), mask[:, n_pre:].bool().to(device)


def shifted_timesteps(n_steps: int, shift: float, device) -> Tensor:
    """Uniform grid 1 -> 0, warped toward the noisy end: t' = s t / (1 + (s - 1) t)."""
    t = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    return shift * t / (1.0 + (shift - 1.0) * t)


@torch.no_grad()
def sample(
    model: nn.Module,
    text: Tensor,
    text_mask: Tensor,
    latent_size: int = 16,
    steps: int = 50,
    cfg: float = 3.0,
    solver: str = "sde",
    churn: float = 0.3,
    shift: float = 3.3,
    seed: int = 0,
    sde_t_max: float = 0.9,
) -> Tensor:
    """Generate normalized latents [B, C, latent_size, latent_size].

    solver "sde": before each step, re-noise from t up to t_hat = min(t (1 + churn), sde_t_max)
    (exact for this interpolant), then take an Euler step from t_hat. "euler" is the
    deterministic ODE. Classifier-free guidance uses a zero text embedding as the
    unconditional branch, which is what training dropped captions to.

    Defaults are the tuned recipe for the released checkpoint (sde, churn 0.3, cfg 3,
    50 steps, shift 3.3). Steps and churn are coupled: churn is applied per step, so
    more steps at the same churn inject more total noise and wash the image out.
    """
    if solver not in ("sde", "euler"):
        raise ValueError("solver must be 'sde' or 'euler'")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = text.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        batch, model.config.in_channels, latent_size, latent_size, device=device, dtype=dtype, generator=gen
    )
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1)

    text = text.to(device, dtype)
    text_mask = text_mask.to(device)
    if cfg != 1.0:
        text = torch.cat([text, torch.zeros_like(text)])
        text_mask = torch.cat([text_mask, text_mask])

    timesteps = shifted_timesteps(steps, shift, device)
    for t_cur, t_next in zip(timesteps[:-1], timesteps[1:], strict=True):
        t = t_cur
        if solver == "sde" and churn > 0 and 0 < t_cur < sde_t_max:
            t = torch.clamp(t_cur * (1 + churn), max=sde_t_max)
            scale = (1 - t) / (1 - t_cur)
            var = (t**2 - (scale * t_cur) ** 2).clamp(min=0)
            z = torch.randn(x.shape, device=device, dtype=x.dtype, generator=noise_gen)
            x = scale * x + var.sqrt() * z
        x_in = torch.cat([x, x]) if cfg != 1.0 else x
        t_in = torch.full((x_in.shape[0], 1), float(t), device=device, dtype=dtype)
        v = model(x_in, t_in, text, text_mask)
        if cfg != 1.0:
            v_cond, v_uncond = v.chunk(2)
            v = v_uncond + cfg * (v_cond - v_uncond)
        x = x + (t_next - t) * v
    return x
