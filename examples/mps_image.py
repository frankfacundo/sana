#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from diffusers import SanaPipeline


LOCAL_SANA_600M_512_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/Sana_600M_512px_diffusers"
LOCAL_SANA15_16B_1024_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
DEFAULT_MODEL_PATH = LOCAL_SANA15_16B_1024_MODEL


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with Sana on Mac MPS.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SANA_IMAGE_MODEL_PATH", DEFAULT_MODEL_PATH),
        help=(
            "A local Sana image diffusers model path. Defaults to the local SANA1.5 1.6B 1024px model. "
            f"For a lighter local model, use: {LOCAL_SANA_600M_512_MODEL}"
        ),
    )
    parser.add_argument("--prompt", default='a cyberpunk cat with a neon sign that says "Sana"')
    parser.add_argument("--output", default="sana_mps_image.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")

    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = get_device()

    pipe = SanaPipeline.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)

    if hasattr(pipe, "vae"):
        pipe.vae.to(dtype)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.to(dtype)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

    image.save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
