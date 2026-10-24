"""Rebuild assets/showcase.png: five 256px samples, left to right.

    python scripts/make_showcase.py --output showcase.png

Each tile is one image (batch 1) with the sampler settings from the model's config.json
(SDE, churn 0.3, CFG 3.0, 50 steps, shift 3.3; bf16 on CUDA) and the seed listed below, so
any tile is also `python scripts/generate.py "<prompt>" --n 1 --seed <seed>`. GPU bf16
rounding can move pixels slightly between machines.
"""

import argparse

import torch
from PIL import Image

from visage import REPO_ID, load_dit, load_sampling, load_text_encoder, load_vae, sample

TILES = [
    {
        "name": "smiling",
        "seed": 41,
        "prompt": "A woman in her 20s, amateur phone photo, fair with visible freckles, hair red, long, wavy, "
        "eyes green, lips open in a wide smile, upper teeth visible, joyful, makeup subtle, natural, "
        "white top, background out of focus green trees, lit by bright, natural light.",
    },
    {
        "name": "angry",
        "seed": 7,
        "prompt": "A man in his 30s, slightly chin down, studio photo, tan with visible texture, hair short, black, "
        "eyes brown, eyebrows dark, thick, furrowed, short dark stubble, lips closed, pressed together, "
        "angry, denim jacket, background out of focus red brick wall, lit by natural daylight.",
    },
    {
        "name": "surprised",
        "seed": 4,
        "prompt": "A woman in her 20s, studio photo, fair, hair long, dark brown, wavy, eyes brown, "
        "eyebrows dark, defined, lips open, surprised, makeup dark eyeliner, mascara, glossy pink lipstick, "
        "black top, background out of focus, white wall, lit by soft, even light.",
    },
    {
        "name": "laughing",
        "seed": 5,
        "prompt": "A Middle Eastern man in his 30s, studio photo, olive, hair short, black, curly, eyes brown, "
        "lips open, upper teeth visible, laughing, dark shirt, background solid dark backdrop, "
        "lit by soft, directional light.",
    },
    {"name": "tongue out", "seed": 88, "prompt": "woman sticking her tongue out, outdoors in a park"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=REPO_ID, help="model dir or hub repo id")
    parser.add_argument("--output", default="showcase.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    text_encoder = load_text_encoder(args.model, device, dtype)
    dit = load_dit(args.model, device, dtype)
    vae = load_vae(args.model, device, dtype)
    settings = load_sampling(args.model)

    strip = Image.new("RGB", (256 * len(TILES), 256))
    for i, tile in enumerate(TILES):
        text, mask = text_encoder([tile["prompt"]])
        latents = sample(dit, text, mask, latent_size=dit.config.image_size, seed=tile["seed"], **settings)
        with torch.no_grad():
            image = vae.decode(latents)[0].float()
        pixels = ((image + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        strip.paste(Image.fromarray(pixels), (256 * i, 0))
        print(f"{i + 1}/{len(TILES)} {tile['name']}", flush=True)
    strip.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
