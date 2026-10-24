"""Train the Visage-Image VAE.   python scripts/train_vae.py configs/vae.yaml

Loss: Charbonnier(recon, image) + w_p * LPIPS + w_a * (mcos + w_d * mdms) + w_t * phase,
where the alignment terms tie the latent to frozen DINOv2 middle-layer features through a
small trained projector (Qwen-Image-VAE-2.0). Their margins loosen over align_ramp_steps,
but only to align_margin_cap of the way, so alignment pressure never switches off. The
phase term removes the decoder's fixed 8 px sub-pixel pattern (a faint 2x4 px hatch) and
is switched on at phase_template_start. No KL, no GAN: the latent is the encoder mean.
After training, compute per-channel latent stats over the data (see compute_latent_stats)
so `encode` returns normalized latents for the DiT.
"""

import sys
import time
from dataclasses import asdict
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from visage.loading import save_component
from visage.training import ImageCaptions, count, infinite, load_config, muon_adamw, set_lr
from visage.vae import VAEConfig, VisageVAE


def is_muon(name: str, p: Tensor) -> bool:
    """Muon on the decoder's hidden matrices; its input/output layers stay on AdamW."""
    return (
        name.startswith("vit_decoder.")
        and p.ndim == 2
        and not name.startswith(("vit_decoder.from_latent.", "vit_decoder.head."))
    )


def charbonnier(prediction: Tensor, target: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


class DINOAlignment(nn.Module):
    """mcos = ReLU(1 - cos(z_p, f_p) - m_cos); mdms = ReLU(|cos(z_p,z_q) - cos(f_p,f_q)| - m_dist)."""

    def __init__(self, latent_channels: int, model: str, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.backbone = timm.create_model(model, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.backbone.requires_grad_(False).eval()
        self.layer = len(self.backbone.blocks) // 2  # middle layer beats the last one
        width = self.backbone.embed_dim
        self.projector = nn.Sequential(nn.Conv2d(latent_channels, 1024, 1), nn.GELU(), nn.Conv2d(1024, width, 1))
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    @torch.no_grad()
    def features(self, image: Tensor) -> Tensor:
        image = F.interpolate(image, size=(self.resolution,) * 2, mode="bilinear", antialias=True)
        image = (image.add(1).mul(0.5) - self.mean) / self.std
        return self.backbone.get_intermediate_layers(image, n=[self.layer - 1], norm=True)[0].float()

    def forward(self, latent: Tensor, image: Tensor, m_cos: float, m_dist: float) -> tuple[Tensor, Tensor]:
        f = F.normalize(self.features(image), dim=-1)
        z = self.projector(latent)
        grid = self.resolution // 14
        if z.shape[-1] != grid:
            z = F.interpolate(z, size=(grid, grid), mode="bilinear")
        z = F.normalize(z.flatten(2).transpose(1, 2).float(), dim=-1)
        mcos = F.relu(1.0 - (z * f).sum(-1) - m_cos).mean()
        mdms = F.relu((z @ z.transpose(1, 2) - f @ f.transpose(1, 2)).abs() - m_dist).mean()
        return mcos, mdms


def phase_template_penalty(recon: Tensor, image: Tensor, period: int = 8) -> Tensor:
    """L1 of the residual's image-independent pattern over the period x period sub-pixel
    phases: the residual averaged over batch and blocks per (channel, phase), minus each
    channel's mean (a colour bias, not a pattern)."""
    b, c, h, w = recon.shape
    residual = (recon.float() - image.float()).reshape(b, c, h // period, period, w // period, period)
    template = residual.mean(dim=(0, 2, 4))
    return (template - template.mean(dim=(1, 2), keepdim=True)).abs().mean()


@torch.no_grad()
def compute_latent_stats(vae: VisageVAE, loader, max_batches: int = 500) -> None:
    """Per-channel mean/std of raw latents, written into the VAE's buffers."""
    total, sq, n = 0, 0, 0
    for i, (image, _) in enumerate(loader):
        if i == max_batches:
            break
        z = vae.encode_raw(image.cuda()).float()
        total = total + z.sum((0, 2, 3))
        sq = sq + z.square().sum((0, 2, 3))
        n += z.shape[0] * z.shape[2] * z.shape[3]
    mean = total / n
    vae.latent_mean.copy_(mean)
    vae.latent_std.copy_((sq / n - mean.square()).clamp_min(1e-12).sqrt())


def main() -> None:
    cfg = load_config(sys.argv[1])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    vae_config = VAEConfig()
    vae = VisageVAE(vae_config).to(device).to(memory_format=torch.channels_last)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").net.to(device).eval().requires_grad_(False)
    align = DINOAlignment(vae_config.latent_channels, cfg["align_model"], cfg["align_resolution"]).to(device)
    print(f"vae {count(vae)}, projector {count(align.projector)}", flush=True)

    optimizers = muon_adamw(vae.named_parameters(), cfg["lr"], cfg["lr"], cfg["weight_decay"], is_muon)
    optimizers += muon_adamw(align.projector.named_parameters(), cfg["lr"], cfg["lr"], 0.0, lambda n, p: False)

    step = 0
    if cfg.get("resume"):
        state = torch.load(cfg["resume"], map_location=device, weights_only=False)
        vae.load_state_dict(state["vae"])
        align.projector.load_state_dict(state["projector"])
        for opt, s in zip(optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(s)
        step = state["step"]

    data = ImageCaptions(cfg["data"], cfg["resolution"], hflip=cfg.get("hflip", False))
    loader = DataLoader(
        data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["workers"],
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    forward = torch.compile(vae) if cfg["compile"] else vae

    def save(path: Path) -> None:
        torch.save(
            {
                "vae": vae.state_dict(),
                "projector": align.projector.state_dict(),
                "optimizers": [o.state_dict() for o in optimizers],
                "step": step,
                "config": cfg,
            },
            path,
        )

    start, t0 = step, time.time()
    for image, _ in infinite(loader):
        if step >= cfg["steps"]:
            break
        lr_factor = set_lr(optimizers, step, cfg["warmup_steps"], cfg["steps"], cfg["lr_final_frac"])
        frac = cfg["align_margin_cap"] * min(1.0, step / cfg["align_ramp_steps"])
        phase_weight = cfg["phase_template_weight"] if step >= cfg["phase_template_start"] else 0.0
        image = image.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon, latent = forward(image)
            pixel = charbonnier(recon.float(), image)
            perceptual = lpips(recon.float().clamp(-1, 1), image).mean()
            mcos, mdms = align(latent, image, 0.5 * frac, 0.3 * frac)
            phase = phase_template_penalty(recon, image) if phase_weight else recon.new_zeros(())
        loss = (
            pixel
            + cfg["perceptual_weight"] * perceptual
            + cfg["align_weight"] * (mcos + cfg["align_mdms_weight"] * mdms)
            + phase_weight * phase
        )
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        step += 1

        if step % cfg["log_every"] == 0:
            rate = (step - start) / (time.time() - t0)
            eta = (cfg["steps"] - step) / rate / 3600
            print(
                f"step {step} loss {loss.item():.4f} pixel {pixel.item():.4f} lpips {perceptual.item():.4f} "
                f"mcos {mcos.item():.4f} mdms {mdms.item():.4f} phase {phase.item():.5f} lr x{lr_factor:.3f} "
                f"{rate:.2f} it/s eta {eta:.1f}h",
                flush=True,
            )
        if step % cfg["save_every"] == 0:
            save(out / f"step_{step:07d}.pt")

    save(out / "last.pt")
    print("computing latent stats", flush=True)
    vae.eval()
    compute_latent_stats(vae, DataLoader(data, batch_size=32, shuffle=True, num_workers=cfg["workers"]))
    save_component(out / "vae", {**asdict(vae_config), "training_step": step}, vae.state_dict())
    print(f"exported {out / 'vae'}", flush=True)


if __name__ == "__main__":
    main()
