# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Any
from typing import Any
import torch
from scripts.attention.utils.determinism import torch_deterministic_mode
from scripts.attention.utils.environment import is_torch_compiling
from scripts.attention.utils.safe_ops import log


def get_arch_tag(device: torch.device | None = None) -> int:
    """
    Returns the compute capability of a given torch device if it's a CUDA device, otherwise returns 0.

    Args:
        device (torch.device | None): torch device. Uses default device if None.

    Returns:
        device_cc (int): compute capability in the SmXXX format (i.e. 90 for Hopper).
    """
    if torch.cuda.is_available() and torch.version.cuda and (device is None or device.type == "cuda"):
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor
    return 0


def log_or_raise_error(msg: str, raise_error: bool = False, exception: Any = RuntimeError):
    if raise_error:
        raise exception(msg)
    else:
        log.debug(msg)


def is_fp8(dtype: torch.dtype) -> bool:
    return dtype in [torch.float8_e5m2, torch.float8_e4m3fn]
