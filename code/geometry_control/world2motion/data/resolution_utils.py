# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

_RESOLUTION_768_SHAPES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1184, 880),
    (880, 1184),
    (1360, 768),
    (768, 1360),
)


def get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    """Determine the resolution string from spatial dimensions.

    Maps the spatial shape (height, width) to a resolution string. This is used
    for resolution-dependent shift lookup when using dict-based shift
    configuration.

    Args:
        spatial_shape: Tuple of (height, width) in pixels.

    Returns:
        Resolution string: "256", "480", "720", or "768".

    Raises:
        ValueError: If the spatial shape is unsupported.

    Note:
        See VIDEO_RES_SIZE_INFO for more details on resolution definitions.
        For the current definition of resolution, these conditions are satisfied.
    """
    if spatial_shape in _RESOLUTION_768_SHAPES:
        return "768"
    min_dim = min(spatial_shape[0], spatial_shape[1])
    if min_dim <= 256:
        return "256"
    elif min_dim <= 640:
        return "480"
    elif min_dim <= 960:
        return "720"
    elif min_dim <= 2048:
        return "768"
    else:
        raise ValueError(f"Unsupported resolution: {spatial_shape}")
