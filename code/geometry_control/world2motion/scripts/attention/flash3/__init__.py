# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
from scripts.attention.utils.safe_ops import log
from scripts.attention.utils.version import version_at_least

FLASH_ATTENTION_V3_MIN_VERSION = "1.0.3"


def flash3_supported() -> bool:
    """
    Returns whether Flash Attention is supported in this environment.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of Flash Attention, meeting minimum version requirements

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v3 is not supported because PyTorch did not detect CUDA runtime.")
        return False
    try:
        import flash_attn_3_nv
    except ImportError:
        log.debug("Flash Attention v3 is not supported because the Python package ('flash_attn_3_nv'_) was not found.")
        return False
    except Exception as e:
        log.debug(f"Flash Attention v3 is not supported because importing the Python package failed: {e}")
        return False
    if not version_at_least(flash_attn_3_nv.__version__, FLASH_ATTENTION_V3_MIN_VERSION):
        log.debug(
            f"Flash Attention v3 ('flash_attn_3_nv') build is not supported; minimum required version is {FLASH_ATTENTION_V3_MIN_VERSION}, got {flash_attn_3_nv.__version__}."
        )
        return False
    return True


FLASH3_SUPPORTED = flash3_supported()
if FLASH3_SUPPORTED:
    from scripts.attention.flash3.functions import flash3_attention
else:
    from scripts.attention.flash3.stubs import flash3_attention
