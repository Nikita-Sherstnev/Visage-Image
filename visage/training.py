"""Data, optimizer and schedule shared by both training loops."""

import json
import math
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


class ImageCaptions(Dataset):
    """A JSONL manifest, one {"image": path, "caption": text} per line. Relative image
    paths resolve against the manifest's directory. Captions are optional for the VAE."""

    def __init__(self, manifest: str, resolution: int, hflip: bool = False) -> None:
        root = Path(manifest).parent
        self.rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        for row in self.rows:
            row["image"] = str(root / row["image"])
        flip = [T.RandomHorizontalFlip()] if hflip else []
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.Resize(resolution, antialias=True),
                T.CenterCrop(resolution),
                *flip,
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = self.transform(Image.open(row["image"]).convert("RGB"))
        return image, row.get("caption", "")


def muon_adamw(
    named_params, lr_adam: float, lr_muon: float, weight_decay: float, is_muon, betas=(0.9, 0.99)
) -> list[torch.optim.Optimizer]:
    """Muon for the hidden 2D matrices `is_muon(name, p)` picks, fused AdamW for the rest."""
    muon, decay, no_decay = [], [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if is_muon(name, p):
            muon.append(p)
        elif p.ndim > 1:
            decay.append(p)
        else:
            no_decay.append(p)
    fused = torch.cuda.is_available()
    optimizers = [
        torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr_adam,
            betas=betas,
            fused=fused,
        )
    ]
    if muon:
        optimizers.append(
            torch.optim.Muon(muon, lr=lr_muon, momentum=0.95, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
        )
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def set_lr(optimizers, step: int, warmup: int, total: int, final_frac: float) -> float:
    """Linear warmup, then cosine down to final_frac of the base LR."""
    if step < warmup:
        factor = (step + 1) / warmup
    else:
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        factor = final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * factor
    return factor


def infinite(loader):
    while True:
        yield from loader


def count(model: nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
