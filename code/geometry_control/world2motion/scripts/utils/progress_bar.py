# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from tqdm.auto import tqdm


def progress_bar(it, **kwargs):
    kwargs.pop("rank0_only", None)
    return tqdm(it, **kwargs)
