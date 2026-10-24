"""Build the released models from a model directory or a Hugging Face repo id.

One folder per component, as on the Hub: transformer/ and vae/ hold config.json +
model.safetensors; text_encoder/config.json names the frozen Qwen3 encoder, which is
fetched from its own hub repo. The root config.json holds the sampler settings tuned for
these weights.
"""

import json
from dataclasses import fields
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from .dit import DiTConfig, VisageDiT
from .sampling import TextEncoder
from .vae import VAEConfig, VisageVAE

REPO_ID = "Sherstnev/Visage-Image"


def resolve(model: str | Path) -> Path:
    """A local directory as is; anything else is a hub repo id, downloaded to the HF cache."""
    path = Path(model)
    if path.is_dir():
        return path
    return Path(snapshot_download(str(model), allow_patterns=["config.json", "*/config.json", "*/model.safetensors"]))


def _config(cls, path: Path):
    data = json.loads(path.read_text())
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def load_dit(model: str | Path = REPO_ID, device="cuda", dtype=torch.bfloat16) -> VisageDiT:
    folder = resolve(model) / "transformer"
    dit = VisageDiT(_config(DiTConfig, folder / "config.json"))
    dit.load_state_dict(load_file(folder / "model.safetensors"))
    return dit.to(device, dtype).eval()


def load_text_encoder(model: str | Path = REPO_ID, device="cuda", dtype=torch.bfloat16) -> TextEncoder:
    """The encoder text_encoder/config.json records: model, layer cut, caption length and hub revision."""
    info = json.loads((resolve(model) / "text_encoder" / "config.json").read_text())
    return TextEncoder(info["name"], info["layer"], info["max_length"], info.get("revision"), dtype).to(device)


def load_vae(model: str | Path = REPO_ID, device="cuda", dtype=torch.bfloat16) -> VisageVAE:
    folder = resolve(model) / "vae"
    vae = VisageVAE(_config(VAEConfig, folder / "config.json"))
    vae.load_state_dict(load_file(folder / "model.safetensors"))
    return vae.to(device, dtype).eval()


def load_sampling(model: str | Path = REPO_ID) -> dict:
    """The sampler settings the root config.json records, as keyword arguments for sample().

    Empty for a folder without one (e.g. a fresh training output); sample()'s defaults apply.
    """
    path = resolve(model) / "config.json"
    return json.loads(path.read_text())["sampling"] if path.is_file() else {}


def save_component(folder: Path, config: dict, state: dict | None = None) -> None:
    """Write one component folder in the layout the loaders read."""
    folder.mkdir(parents=True, exist_ok=True)
    if state is not None:
        save_file({k: v.contiguous() for k, v in state.items()}, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps(config, indent=2))
