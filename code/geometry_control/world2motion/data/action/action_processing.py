# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch


class ActionNormalizer(Protocol):
    """Tensor-level action normalization interface used by ActionProcessor."""

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map raw action values into model-space values."""
        ...

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Invert model-space action values back into raw action values."""
        ...


@dataclass(frozen=True)
class ActionProcessingRecord:
    """Per-sample metadata needed to invert Action model-space preprocessing."""

    raw_action_dim: int
    action_normalizer: ActionNormalizer | None


def make_batched_action_processing_fields(
    record: ActionProcessingRecord, batch_size: int, *, action_channel_masking: bool = True
) -> dict[str, list[torch.Tensor | ActionProcessingRecord | None]]:
    """Build batch-list fields whose action width and inverse record cannot drift apart."""
    raw_action_dim = torch.tensor(record.raw_action_dim, dtype=torch.long) if action_channel_masking else None
    return {"raw_action_dim": [raw_action_dim] * batch_size, "action_processing_record": [record] * batch_size}
