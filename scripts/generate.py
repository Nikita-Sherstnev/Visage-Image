"""Generate faces with Visage-Image.

python scripts/generate.py "a woman with red hair and freckles, soft window light" --n 4
python scripts/generate.py "..." --model path/to/model --steps 50 --cfg 3 --churn 0.3 --seed 7

Sampler flags left unset take the values in the model's config.json, which records the
settings tuned for those weights.
"""

import argparse

import torch
from torchvision.utils import save_image

from visage import REPO_ID, load_dit, load_sampling, load_text_encoder, load_vae, sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--model", default=REPO_ID, help="model dir or hub repo id")
    parser.add_argument("--output", default="visage.png")
    parser.add_argument("--n", type=int, default=4, help="images, one batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--solver", choices=("sde", "euler"))
    parser.add_argument("--churn", type=float)
    parser.add_argument("--shift", type=float)
    args = parser.parse_args()
    flags = {k: getattr(args, k) for k in ("steps", "cfg", "solver", "churn", "shift")}
    settings = load_sampling(args.model) | {k: v for k, v in flags.items() if v is not None}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    text_encoder = load_text_encoder(args.model, device, dtype)
    text, mask = text_encoder([args.prompt] * args.n)
    del text_encoder

    dit = load_dit(args.model, device, dtype)
    latents = sample(
        dit,
        text,
        mask,
        latent_size=dit.config.image_size,
        seed=args.seed,
        **settings,
    )
    del dit
    vae = load_vae(args.model, device, dtype)
    with torch.no_grad():
        images = vae.decode(latents).float()
    save_image((images + 1) / 2, args.output, nrow=min(args.n, 4), padding=0)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
