"""First RGB frame + first SMPL-X motion -> video.mp4 and motion.npz."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed body, artifacts, "
    "jpeg compression, oversaturated, extra limbs, bad anatomy, camera motion, camera shake"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="First RGB frame (.png/.jpg)")
    parser.add_argument("--motion", required=True, type=Path, help="First SMPL-X pose (.npz)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--prompt", default="", help="Optional English description of the desired motion")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--vae", type=Path, help="Optional alternate Wan2.2_VAE.pth path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--frames", type=int, default=121, help="Total frames including frame zero, 4n+1 (default: 121)"
    )
    parser.add_argument("--resolution", choices=["256", "480", "720"], default="720")
    args = parser.parse_args()
    if args.frames < 5 or args.frames > 121 or (args.frames - 1) % 4:
        parser.error("--frames must be 4n+1 between 5 and 121")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.seed < 0 or args.seed > 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295")
    for path in (args.image, args.motion):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    if any((args.output / name).exists() for name in ("video.mp4", "motion.npz", "inference.json")):
        parser.error("Output already contains inference results; choose a new --output directory")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import imageio.v2 as imageio
    import torch
    from data.preprocess import build_batch, load_first_motion, save_motion
    from scripts.pipeline import load_pipeline

    condition, metadata = load_first_motion(args.motion)
    pipeline = load_pipeline(args.checkpoint_dir, args.vae)
    batch = build_batch(
        args.image, condition, args.prompt, NEGATIVE_PROMPT, frames=args.frames, resolution=args.resolution
    )
    logging.info("Generating %s video and motion frames", args.frames)
    # Do not wrap this call in autocast: temporal RoPE coordinates require FP32.
    with torch.inference_mode():
        video, motion = pipeline.generate(batch, seed=args.seed, num_steps=args.steps)
    if video.shape[1] != args.frames or motion.shape[0] != args.frames:
        raise RuntimeError(f"Frame alignment mismatch: video={video.shape}, motion={motion.shape}")
    if not torch.isfinite(video).all() or not torch.isfinite(motion).all():
        raise RuntimeError("Inference produced non-finite values")
    args.output.mkdir(parents=True, exist_ok=True)
    save_motion(args.output / "motion.npz", motion.float().cpu().numpy(), metadata)
    with imageio.get_writer(
        args.output / "video.mp4", fps=24, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        for frame in video.permute(1, 2, 3, 0):
            writer.append_data(((frame.float().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy())
    details = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    details.update(
        fps=24, width=int(video.shape[-1]), height=int(video.shape[-2]), checkpoint="HOI-mix iter_000020000, EMA BF16"
    )
    (args.output / "inference.json").write_text(json.dumps(details, indent=2), encoding="utf8")
    logging.info("Saved %s and %s", args.output / "video.mp4", args.output / "motion.npz")


if __name__ == "__main__":
    main()
