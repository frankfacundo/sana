#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import SanaVideoPipeline
from diffusers.utils import export_to_video


DEFAULT_VIDEO_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
LOCAL_IMAGE_MODEL = "/Users/frankfacundo/Models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_video_model(model_path: str) -> None:
    path = Path(model_path).expanduser()
    if not path.exists():
        return

    model_index = path / "model_index.json"
    if not model_index.exists():
        return

    class_name = json.loads(model_index.read_text()).get("_class_name")
    if class_name != "SanaVideoPipeline":
        raise ValueError(
            f"{path} is a {class_name}, not a SanaVideoPipeline. "
            "Use examples/mps_image.py for the local SANA1.5 image model, "
            "or pass --model-path pointing to a Sana-Video diffusers model."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short video with Sana-Video on Mac MPS.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SANA_VIDEO_MODEL_PATH", DEFAULT_VIDEO_MODEL),
        help=(
            "A Sana-Video diffusers model path or Hugging Face repo. "
            f"The local image model is {LOCAL_IMAGE_MODEL} and cannot generate video."
        ),
    )
    parser.add_argument("--prompt", default="A cat walking through green grass, facing the camera.")
    parser.add_argument(
        "--negative-prompt",
        default="motion blur, malformed limbs, jitter, distorted frames, low quality",
    )
    parser.add_argument("--output", default="sana_mps_video.mp4")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--motion-score", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")

    args = parse_args()
    validate_video_model(args.model_path)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = get_device()

    pipe = SanaVideoPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only or Path(args.model_path).expanduser().exists(),
    )
    pipe.vae.to(torch.float32)
    pipe.text_encoder.to(dtype)
    pipe.to(device)

    prompt = f"{args.prompt} motion score: {args.motion_score}."
    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            frames=args.frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).frames[0]

    export_to_video(video, args.output, fps=args.fps)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
