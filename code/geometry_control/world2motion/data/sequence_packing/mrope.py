# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import math
import torch


def get_3d_mrope_ids_text_tokens(
    num_tokens: int, temporal_offset: int | float, use_float_positions: bool = False
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for text tokens.

    For text tokens, all three axes (temporal, height, width) share the same
    monotonically increasing position IDs, starting from ``temporal_offset``.

    Args:
        num_tokens: Number of text tokens.
        temporal_offset: Current temporal offset to start from. Can be float when
            FPS modulation is enabled for vision tokens.
        use_float_positions: If ``True``, generate float position IDs (for consistency
            with FPS-modulated vision tokens). If ``False``, generate integer IDs.

    Returns:
        Tuple of:
            - Position IDs tensor of shape ``(3, num_tokens)`` where each row is identical.
            - Updated temporal offset (``temporal_offset + num_tokens``).
    """
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    next_temporal_offset = temporal_offset + num_tokens
    return (mrope_ids, next_temporal_offset)


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
    temporal_positions: torch.Tensor | None = None,
    actual_temporal_compression_factor: int | None = None,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for VAE vision tokens (image/video latents).

    Creates a 3D position grid for vision tokens with shape ``(T, H, W)``, then flattens
    to produce position IDs for each axis. The flattening order is T-major:
    for each temporal frame, iterate over height then width.

    Args:
        grid_t: Number of temporal frames in the latent grid.
        grid_h: Height of the latent grid (after patchification).
        grid_w: Width of the latent grid (after patchification).
        temporal_offset: Current temporal offset. Always applied to the temporal axis.
            When ``reset_spatial_indices=False``, also applied to spatial axes.
            Can be float when FPS modulation is enabled.
        reset_spatial_indices: If ``True``, spatial (height, width) indices start from 0
            for each vision segment, giving the model absolute spatial position
            within each image/video. If ``False``, spatial indices are also offset by
            ``temporal_offset`` (Qwen2VL-style behavior).
        fps: Frames per second of the video. ``None`` disables fps modulation
            (integer positions); pass the real fps for fps-scaled, possibly
            fractional positions. Honored at grid_t=1 too (per-frame AR packs),
            where it collapses to ``scaled_t[0] = temporal_offset``.
        base_fps: Base FPS for normalization. Default is 24.0.
        temporal_compression_factor: VAE temporal compression factor. Default is 4.
        base_temporal_compression_factor: Base temporal compression factor. If ``None``,
            defaults to ``temporal_compression_factor`` (typical case where base matches actual).
        start_frame_offset: Offset added to frame indices before FPS scaling.
            Use 1 for action embeddings so they start at frame 1 instead of 0.
        temporal_positions: Optional explicit temporal coordinates for each latent
            frame, in source-frame / actual-temporal-compression-factor units.
            When provided, positions can be fractional and must have shape ``(grid_t,)``.
        actual_temporal_compression_factor: Temporal compression factor that defines
            ``temporal_positions``. Defaults to ``temporal_compression_factor``.

    Returns:
        Tuple of:
            - Position IDs tensor of shape ``(3, grid_t * grid_h * grid_w)``.
              Row 0: temporal axis (float if FPS modulation enabled, else long).
              Row 1: height axis (long), Row 2: width axis (long).
            - Updated temporal offset for the next segment. When FPS modulation is
              enabled, this is a float representing the next scaled time position.
              Otherwise, it's ``max(all_positions) + 1`` (Qwen3VL design).
    """
    fps_modulation_enabled = fps is not None
    explicit_temporal_positions = temporal_positions is not None
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )
    effective_actual_tcf = (
        actual_temporal_compression_factor
        if actual_temporal_compression_factor is not None
        else temporal_compression_factor
    )
    if explicit_temporal_positions:
        assert temporal_positions is not None
        if temporal_positions.ndim != 1 or temporal_positions.shape[0] != grid_t:
            raise ValueError(
                f"temporal_positions must have shape (grid_t,), got {tuple(temporal_positions.shape)} for grid_t={grid_t!r}."
            )
        frame_indices = temporal_positions.to(dtype=torch.float32)
        if start_frame_offset != 0:
            frame_indices = frame_indices + start_frame_offset / effective_actual_tcf
        if fps_modulation_enabled:
            scaled_t = frame_indices * effective_actual_tcf * (base_fps / effective_base_tcf) / fps + temporal_offset
        else:
            scaled_t = frame_indices + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    elif fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )
    device = t_index.device
    h_index = torch.arange(grid_h, dtype=torch.long, device=device).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long, device=device).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset
    if fps_modulation_enabled or explicit_temporal_positions:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    max_position = mrope_ids.max().item()
    next_temporal_offset = math.ceil(max_position) + 1
    return (mrope_ids, next_temporal_offset)
