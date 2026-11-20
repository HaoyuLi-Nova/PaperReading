# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
from scripts.attention.utils.safe_ops import log
from scripts.attention.utils.version import version_in_range

FLASH_ATTENTION_V2_MIN_VERSION = "2.7.0"
FLASH_ATTENTION_V2_MAX_VERSION = "2.7.4.post1"


def flash2_supported() -> bool:
    """
    Returns whether Flash Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of Flash Attention, meeting minimum version requirements

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v2 is not supported because PyTorch did not detect CUDA runtime.")
        return False
    try:
        import flash_attn
    except ImportError:
        log.debug("Flash Attention v2 is not supported because the Python package was not found.")
        return False
    except Exception as e:
        log.debug(f"Flash Attention v2 is not supported because importing the Python package failed: {e}")
        return False
    flash2_version_str = None
    if not hasattr(flash_attn, "__version__"):
        from importlib.metadata import version

        flash2_version_str = version("flash_attn")
    else:
        flash2_version_str = flash_attn.__version__
    if not version_in_range(flash2_version_str, FLASH_ATTENTION_V2_MIN_VERSION, FLASH_ATTENTION_V2_MAX_VERSION):
        log.debug(
            f"Flash Attention v2 build is not supported; this backend only supports versions {FLASH_ATTENTION_V2_MIN_VERSION} through {FLASH_ATTENTION_V2_MAX_VERSION}, got {flash2_version_str}."
        )
        return False
    return True


FLASH2_SUPPORTED = flash2_supported()
if FLASH2_SUPPORTED:
    from scripts.attention.flash2.functions import flash2_attention
else:
    from scripts.attention.flash2.stubs import flash2_attention
