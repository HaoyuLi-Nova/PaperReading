# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
import torchvision.transforms.functional as transforms_F
from data.resolutions import VIDEO_RES_SIZE_INFO
from data.sequence_packing import SequencePlan


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """Find the closest predefined target size for a given input resolution.

    Looks up ``VIDEO_RES_SIZE_INFO[resolution]`` and selects the aspect ratio
    whose ``H/W`` ratio is closest to the input ``h/w``.

    Args:
        h: Input height in pixels.
        w: Input width in pixels.
        resolution: Resolution tier key (e.g. ``"256"``, ``"480"``, ``"720"``).

    Returns:
        ``(target_w, target_h)`` from the predefined table.

    Raises:
        ValueError: If *resolution* is not a key in ``VIDEO_RES_SIZE_INFO``.
    """
    if isinstance(resolution, int):
        resolution = str(resolution)
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Resolution '{resolution}' not found in VIDEO_RES_SIZE_INFO. Available: {list(VIDEO_RES_SIZE_INFO.keys())}"
        )
    candidates = VIDEO_RES_SIZE_INFO[resolution]
    input_ratio = h / w
    best_key: str | None = None
    best_diff = float("inf")
    for aspect_key, (cand_w, cand_h) in candidates.items():
        cand_ratio = cand_h / cand_w
        diff = abs(input_ratio - cand_ratio)
        if diff < best_diff:
            best_diff = diff
            best_key = aspect_key
    assert best_key is not None
    target_w, target_h = candidates[best_key]
    return (target_w, target_h)


def reflection_pad_to_target(
    data_dict: dict, keys: list[str], keep_aspect_ratio: bool, target_w: int, target_h: int
) -> dict:
    """Resize (aspect-preserving) and reflection-pad tensors to exact target size.

    For each key in *keys*, the tensor is:

    1. Resized so its spatial dimensions fit within ``(target_h, target_w)``
       while preserving the aspect ratio (matching VFM's
       ``ResizeLargestSideAspectPreserving``).
    2. Reflection-padded (or edge-padded when the padding exceeds the spatial
       dimension) to reach exactly ``(target_h, target_w)`` (matching VFM's
       ``ReflectionPadding``).

    After processing, the following entries are added to *data_dict*:

    - ``"image_size"``: ``torch.Tensor`` of shape ``(4,)`` containing
      ``[target_h, target_w, orig_h_resized, orig_w_resized]`` where
      ``target_h/w`` is the padded canvas size and ``orig_h/w_resized``
      is the original spatial size after aspect-preserving resize (i.e.
      the content region before padding).  After ``default_collate``
      this becomes ``(B, 4)``;  the ``IterativeJointDataLoader`` then
      splits it into per-sample ``(1, 4)`` tensors so the model can
      index as ``data_batch["image_size"][i][0][0]``.

    Args:
        data_dict: The sample dictionary (mutated in-place).
        keys: Data-dict keys whose tensors should be resized and padded.
            Tensors must have shape ``(C, H, W)`` or ``(C, T, H, W)``.
        keep_aspect_ratio: Whether to keep the aspect ratio of the input tensor.
        target_w: Target width in pixels.
        target_h: Target height in pixels.

    Returns:
        The mutated *data_dict*.
    """
    orig_h_resized: int = 0
    orig_w_resized: int = 0
    for key in keys:
        if key not in data_dict:
            continue
        tensor = data_dict[key]
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim == 3:
            orig_h, orig_w = tensor.shape[-2:]
        elif tensor.ndim == 4:
            orig_h, orig_w = tensor.shape[-2:]
        else:
            raise ValueError(f"Unexpected tensor ndim={tensor.ndim} for key '{key}', expected 3 or 4")
        if keep_aspect_ratio:
            scaling_ratio = min(target_w / orig_w, target_h / orig_h, 1.0)
            orig_h_resized = int(scaling_ratio * orig_h + 0.5)
            orig_w_resized = int(scaling_ratio * orig_w + 0.5)
            assert orig_h_resized <= target_h and orig_w_resized <= target_w, (
                f"Resize error: orig ({orig_h}, {orig_w}) target ({target_h}, {target_w}) computed ({orig_h_resized}, {orig_w_resized})"
            )
        else:
            orig_h_resized = target_h
            orig_w_resized = target_w
        if orig_h_resized != orig_h or orig_w_resized != orig_w:
            tensor = transforms_F.resize(
                tensor,
                size=[orig_h_resized, orig_w_resized],
                interpolation=transforms_F.InterpolationMode.BICUBIC,
                antialias=True,
            )
        if orig_w_resized != target_w or orig_h_resized != target_h:
            padding_right = target_w - orig_w_resized
            padding_bottom = target_h - orig_h_resized
            padding = [0, 0, padding_right, padding_bottom]
            if padding_right >= orig_w_resized or padding_bottom >= orig_h_resized:
                tensor = transforms_F.pad(tensor, padding, padding_mode="edge")
            else:
                tensor = transforms_F.pad(tensor, padding, padding_mode="reflect")
        data_dict[key] = tensor
    data_dict["image_size"] = torch.tensor([target_h, target_w, orig_h_resized, orig_w_resized], dtype=torch.float)
    return data_dict


def build_sequence_plan_from_mode(
    mode: str,
    video_length: int,
    action_length: int,
    has_text: bool = True,
    video_temporal_downsample: int = 4,
    num_history_actions: int = 0,
) -> SequencePlan:
    """Build a SequencePlan based on the training mode.

    This function determines whether action should be included and computes the
    appropriate condition frame indexes for vision and action based on the mode.

    Args:
        mode: Training mode. One of:
            - "image2video": Image-to-video generation (no action)
            - "forward_dynamics": Predict video given first frame and all actions
            - "inverse_dynamics": Predict actions given all video frames
            - "policy": Predict both actions and video given first frame
        video_length: Number of video frames (including the conditioning frame).
        action_length: Number of action steps (typically video_length - 1).
        has_text: Whether text conditioning is available. Defaults to True.
        video_temporal_downsample: Temporal downsampling factor of the video
            tokenizer. Used to compute condition frame indexes for inverse
            dynamics mode. Defaults to 4.

    Returns:
        SequencePlan instance with appropriate settings.
        Use ``sequence_plan.has_action`` to check if action should be included.

    Raises:
        ValueError: If mode is not one of the supported modes.

    Example:
        >>> sequence_plan = build_sequence_plan_from_mode(
        ...     mode="policy",
        ...     video_length=5,
        ...     action_length=4,
        ... )
        >>> sequence_plan.has_action
        True
        >>> sequence_plan.as_dict()
        {'has_text': True, 'has_vision': True, 'has_action': True,
         'condition_frame_indexes_vision': [0], 'condition_frame_indexes_action': []}
    """
    valid_modes = ["image2video", "forward_dynamics", "inverse_dynamics", "policy"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode!r}. Must be one of {valid_modes}")
    has_action = mode != "image2video"
    if mode in ["image2video", "forward_dynamics", "policy"]:
        condition_frame_indexes_vision = [0]
    elif mode == "inverse_dynamics":
        condition_frame_indexes_vision = list(range(0, (video_length - 1) // video_temporal_downsample + 1))
    else:
        condition_frame_indexes_vision = []
    base_action_length = action_length - num_history_actions
    if mode == "forward_dynamics":
        condition_frame_indexes_action = list(range(action_length))
    elif base_action_length == video_length - 1:
        condition_frame_indexes_action = list(range(num_history_actions))
    elif base_action_length == video_length:
        condition_frame_indexes_action = list(range(num_history_actions + 1))
    if base_action_length == video_length - 1:
        action_start_frame_offset = 1 - num_history_actions
    if base_action_length == video_length:
        action_start_frame_offset = -num_history_actions
    return SequencePlan(
        has_text=has_text,
        has_vision=True,
        has_action=has_action,
        condition_frame_indexes_vision=condition_frame_indexes_vision,
        condition_frame_indexes_action=condition_frame_indexes_action,
        action_start_frame_offset=action_start_frame_offset,
    )
