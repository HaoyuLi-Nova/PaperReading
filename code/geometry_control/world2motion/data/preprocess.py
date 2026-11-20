"""First-frame RGB/SMPL-X preprocessing and numerical motion postprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation

STATS_PATH = Path(__file__).with_name("motion_stats.json")


def statistics():
    stats = json.loads(STATS_PATH.read_text())["global"]
    lo, hi = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
    return lo, np.maximum(hi - lo, 1e-8)


def first_pose(array, dimensions, name):
    value = np.asarray(array, dtype=np.float64)
    if value.shape == (dimensions,) or value.shape == (dimensions // 3, 3):
        value = value.reshape(dimensions)
    elif value.ndim >= 2 and value.shape[0] > 0 and int(np.prod(value.shape[1:])) == dimensions:
        value = value[0].reshape(dimensions)
    else:
        raise ValueError(f"{name}: expected [{dimensions}] or [T,{dimensions}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def load_first_motion(path: Path):
    """Read axis-angle poses in radians and camera-space translation in meters."""
    with np.load(path, allow_pickle=False) as source:
        missing = {"root_pose", "body_pose", "cam_trans"} - set(source.files)
        if missing:
            raise ValueError(f"Missing motion fields: {sorted(missing)}")
        root = first_pose(source["root_pose"], 3, "root_pose")
        body = first_pose(source["body_pose"], 63, "body_pose")
        translation = first_pose(source["cam_trans"], 3, "cam_trans")
        metadata = {"root_pose": root, "body_pose": body, "cam_trans": translation}
        if "betas" in source.files:
            betas = np.asarray(source["betas"], np.float32)
            if betas.ndim > 1:
                betas = betas[0]
            if not np.isfinite(betas).all():
                raise ValueError("betas contains non-finite values")
            metadata["betas"] = betas
        if "gender" in source.files:
            gender = source["gender"]
            metadata["gender"] = np.asarray(str(gender.reshape(-1)[0]))
    rotations = Rotation.from_rotvec(np.concatenate([root[None], body.reshape(21, 3)])).as_matrix()
    # The model uses the FIRST TWO ROWS of each rotation matrix, not columns.
    raw = np.concatenate([np.zeros(3), rotations[:, :2, :].reshape(-1)])
    lo, scale = statistics()
    normalized = (2.0 * (raw - lo) / scale - 1.0).astype(np.float32)
    return normalized, metadata


def build_batch(image_path, normalized_motion, prompt, negative_prompt, frames=121, resolution="720", fps=24):
    import torch
    from data.action.action_processing import ActionProcessingRecord, make_batched_action_processing_fields
    from data.action.json_formatter import ActionPromptJsonFormatter
    from data.action.transforms import (
        build_sequence_plan_from_mode,
        find_closest_target_size,
        reflection_pad_to_target,
    )

    with Image.open(image_path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"), copy=True)
    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(1)
    height, width = image.shape[-2:]
    target_w, target_h = find_closest_target_size(height, width, resolution)
    padded = {"video": image}
    reflection_pad_to_target(padded, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h)
    video = padded["video"].repeat(1, frames, 1, 1)
    actions = torch.zeros(frames, 160, dtype=torch.float32)
    actions[0, :135] = torch.from_numpy(normalized_motion)
    fields = dict(
        viewpoint="ego_view",
        ai_caption=prompt.strip(),
        video=video,
        action=actions,
        conditioning_fps=torch.tensor(fps),
        image_size=padded["image_size"],
    )
    caption = ActionPromptJsonFormatter()(fields)["ai_caption"]
    if isinstance(caption, dict):
        caption = json.dumps(caption)
    plan = build_sequence_plan_from_mode(mode="policy", video_length=frames, action_length=frames, has_text=True)
    return {
        "video": [[video]],
        "action": [[actions]],
        **make_batched_action_processing_fields(ActionProcessingRecord(raw_action_dim=135, action_normalizer=None), 1),
        "ai_caption": [caption],
        "neg_ai_caption": [negative_prompt],
        "prompt": [prompt],
        "mode": ["policy"],
        "conditioning_fps": [torch.tensor(fps)],
        "image_size": padded["image_size"].unsqueeze(0).cuda(),
        "domain_id": [torch.tensor(10)],
        "sequence_plan": [plan],
    }


def rotation_6d_to_axis_angle(rotations):
    first, second = rotations[..., :3], rotations[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-9)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-9)
    matrix = np.stack([first, second, np.cross(first, second)], axis=-2)
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(*rotations.shape[:-1], 3)


def save_motion(path, normalized_motion, metadata, fps=24):
    normalized_motion = np.asarray(normalized_motion, dtype=np.float64)
    if normalized_motion.ndim != 2 or normalized_motion.shape[1] != 135 or not np.isfinite(normalized_motion).all():
        raise ValueError("Generated motion must be a finite [T,135] array")
    lo, scale = statistics()
    raw = (normalized_motion + 1.0) * 0.5 * scale + lo
    result = {
        "root_pose": rotation_6d_to_axis_angle(raw[:, 3:9]).astype(np.float32),
        "body_pose": rotation_6d_to_axis_angle(raw[:, 9:].reshape(-1, 21, 6)).reshape(-1, 63).astype(np.float32),
        "cam_trans": (raw[:, :3] + metadata["cam_trans"]).astype(np.float32),
        "fps": np.asarray(fps, dtype=np.float32),
    }
    # Return the exact supplied conditioning pose in frame zero (before BF16 rounding).
    for key in ("root_pose", "body_pose", "cam_trans"):
        result[key][0] = metadata[key]
    for key in ("betas", "gender"):
        if key in metadata:
            result[key] = metadata[key]
    np.savez_compressed(path, **result)
