"""Train the Visage-Image DiT.   python scripts/train_dit.py configs/dit.yaml

Rectified flow: x_t = (1 - t) x + t noise, target v = noise - x, MSE. Timesteps are
logit-normal, shifted by sqrt(tokens / shift_ref_tokens) so larger grids see more
noise. Images are encoded on the fly with the frozen VAE and captions with frozen Qwen3-0.6B;
for long runs it is much faster to cache both once.

With an `adversarial` block in the config, a latent critic (visage/adversarial.py) adds a
non-saturating GAN term next to the flow loss. The release used it for 1000 steps on top
of the annealed model; see configs/dit_adversarial.yaml.
"""

import copy
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from visage import TextEncoder, load_vae, sample
from visage.adversarial import LatentCritic, approximate_r1, critic_loss, generator_loss, sample_critic_t
from visage.dit import DiTConfig, VisageDiT
from visage.loading import resolve, save_component
from visage.training import ImageCaptions, count, infinite, load_config, muon_adamw, set_lr

# Embeddings, input/output projections and adaLN modulators stay on AdamW.
ADAM_ONLY = ("time_embed.", "patch_embed.", "text_proj.", "out_proj.", "text_pad_embed", "text_registers", "final_mod.")


def is_muon(name: str, p: torch.Tensor) -> bool:
    return p.ndim == 2 and p.numel() > 4096 and not name.startswith(ADAM_ONLY) and ".mod." not in name


def main() -> None:
    cfg = load_config(sys.argv[1])
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    config = DiTConfig(**cfg["model"])
    model = VisageDiT(config).to(device)
    if cfg.get("init"):
        model.load_state_dict(load_file(resolve(cfg["init"]) / "transformer" / "model.safetensors"))
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    vae = load_vae(cfg["vae"], device).requires_grad_(False)
    text_encoder = TextEncoder().to(device)
    print(f"dit {count(model)}", flush=True)

    optimizers = muon_adamw(model.named_parameters(), cfg["lr_adam"], cfg["lr_muon"], cfg["weight_decay"], is_muon)
    step = 0
    if cfg.get("resume"):
        state = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        for opt, s in zip(optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(s)
        step = state["step"]

    adv = cfg.get("adversarial")
    critic = None
    if adv:
        # The critic's backbone is a frozen copy of the model at the fork point, not the
        # generator itself, which would be judging its own output distribution.
        backbone = VisageDiT(config).to(device)
        backbone_dir = resolve(adv["backbone"] or cfg["init"])
        backbone.load_state_dict(load_file(backbone_dir / "transformer" / "model.safetensors"))
        critic = LatentCritic(backbone, adv["layers"], adv["head_hidden"], adv["center_latents"]).to(device)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=adv["lr"], betas=(0.0, 0.99))
        # The heads start at zero; the generator term waits until they have an opinion.
        adv_start = step + adv["warmup_steps"]
        if cfg.get("resume") and "critic" in state:
            critic.load_state_dict(state["critic"])
            critic_opt.load_state_dict(state["critic_opt"])
            adv_start = step
        print(f"critic heads {count(critic)} on blocks {adv['layers']}", flush=True)

    data = ImageCaptions(cfg["data"], cfg["resolution"])
    loader = DataLoader(
        data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["workers"],
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    forward = torch.compile(model) if cfg["compile"] else model
    tokens = (cfg["resolution"] // 16 // config.patch_size) ** 2
    log_shift = math.log(math.sqrt(tokens / cfg["shift_ref_tokens"]))
    ema_decay = cfg["ema_decay"] ** cfg["ema_every"]

    def save(path: Path) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "step": step,
                "config": cfg,
                "optimizers": [o.state_dict() for o in optimizers],
                **({"critic": critic.state_dict(), "critic_opt": critic_opt.state_dict()} if critic else {}),
            },
            path,
        )

    def preview() -> None:
        text, mask = text_encoder(cfg["sample_prompts"])
        latents = sample(ema, text, mask, latent_size=config.image_size, seed=0)
        with torch.no_grad():
            images = vae.decode(latents).float()
        save_image((images + 1) / 2, out / f"sample_{step:07d}.png", nrow=len(images), padding=0)

    batches = infinite(loader)
    start, t0 = step, time.time()
    while step < cfg["steps"]:
        lr_factor = set_lr(optimizers, step, cfg["warmup_steps"], cfg["steps"], cfg["lr_final_frac"])
        total = adv_total = 0.0
        for _ in range(cfg["grad_accum"]):
            image, captions = next(batches)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = vae.encode(image.to(device, non_blocking=True)).float()
            text, mask = text_encoder(captions)
            # CFG dropout zeroes the embedding but keeps the caption's mask, exactly
            # like the unconditional branch at sampling time.
            dropped = torch.rand(len(captions), device=device) < cfg["caption_dropout"]
            text = text.clone().masked_fill(dropped[:, None, None], 0.0)

            t = torch.sigmoid(torch.randn(x.shape[0], 1, device=device) + log_shift)
            noise = torch.randn_like(x)
            x_t = (1 - t[:, :, None, None]) * x + t[:, :, None, None] * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = forward(x_t, t, text, mask)
            loss = F.mse_loss(pred.float(), noise - x)
            total += loss.item() / cfg["grad_accum"]
            if critic is not None:
                # The generator's sample is its one-step clean prediction, x0 = x_t - t v, re-noised
                # to the critic's own level; real and fake share that noise draw.
                pick = torch.randperm(x.shape[0], device=device)[: adv["slice"]]
                fake = x_t[pick] - t[pick, :, None, None] * pred[pick].float()
                t_d = sample_critic_t(len(pick), device, adv["t_mean"], adv["t_std"])
                adv_noise = torch.randn_like(fake)
                t_img = t_d.view(-1, 1, 1, 1)
                noisy_fake = (1 - t_img) * fake + t_img * adv_noise
                noisy_real = (1 - t_img) * x[pick] + t_img * adv_noise
                adv_text, adv_mask = text[pick], None if mask is None else mask[pick]

                # Critic step on a detached sample, every micro-batch.
                critic.requires_grad_(True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    real_logits = critic(noisy_real, t_d, adv_text, adv_mask)
                    d_loss = critic_loss(real_logits, critic(noisy_fake.detach(), t_d, adv_text, adv_mask))
                    if adv["r1_weight"]:
                        d_loss = d_loss + adv["r1_weight"] * approximate_r1(
                            critic, noisy_real, t_d, adv_text, adv_mask, real_logits, adv["r1_sigma"]
                        )
                critic_opt.zero_grad(set_to_none=True)
                d_loss.backward()
                critic_opt.step()

                # Generator term with the heads held fixed.
                if step >= adv_start:
                    critic.requires_grad_(False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        g_loss = generator_loss(critic(noisy_fake, t_d, adv_text, adv_mask))
                    loss = loss + adv["weight"] * g_loss
                    adv_total += g_loss.item() / cfg["grad_accum"]
            (loss / cfg["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg["ema_every"] == 0:
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - ema_decay)
        if step % cfg["log_every"] == 0:
            rate = (step - start) / (time.time() - t0)
            eta = (cfg["steps"] - step) / rate / 3600
            line = f"step {step} loss {total:.4f}"
            if critic is not None:
                line += f" adv {adv_total:.4f} critic {d_loss.item():.4f}"
            print(f"{line} lr x{lr_factor:.3f} {rate:.2f} it/s eta {eta:.1f}h", flush=True)
        if step % cfg["sample_every"] == 0:
            preview()
        if step % cfg["save_every"] == 0:
            save(out / f"step_{step:07d}.pt")

    save(out / "last.pt")
    save_component(out / "transformer", {**asdict(config), "training_step": step, "weights": "EMA"}, ema.state_dict())
    save_component(out / "text_encoder", text_encoder.info)
    print(f"exported {out / 'transformer'}", flush=True)


if __name__ == "__main__":
    main()
