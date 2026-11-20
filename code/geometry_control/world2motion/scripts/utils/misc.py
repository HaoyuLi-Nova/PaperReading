# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Any
import numpy as np
import torch


def arch_invariant_rand(shape, dtype, device, seed=None):
    random_array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)
