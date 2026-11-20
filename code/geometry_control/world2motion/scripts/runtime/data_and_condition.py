# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class GenerationDataClean:
    """
    Container for tokenized states and conditioning info (clean states)
    for the multi-modal (vision, sound, action) MoT training.
    Used for the VFM generation model.
    """

    batch_size: int
    is_image_batch: bool
    raw_state_vision: list[torch.Tensor] | None = None
    x0_tokens_vision: list[torch.Tensor] | None = None
    fps_vision: torch.Tensor | None = None
    temporal_positions_vision: list[torch.Tensor] | None = None
    num_vision_items_per_sample: list[int] | None = None
    raw_state_sound: torch.Tensor | None = None
    x0_tokens_sound: torch.Tensor | None = None
    fps_sound: torch.Tensor | None = None
    raw_state_action: list[torch.Tensor] | None = None
    x0_tokens_action: list[torch.Tensor] | None = None
    fps_action: torch.Tensor | None = None
    action_domain_id: list[torch.Tensor] | None = None
    raw_action_dim: list[torch.Tensor] | None = None
    control_weights: list[list[float]] | None = None


def unwrap_and_densify(raw: list | torch.Tensor | None, to_kwargs: dict) -> list[torch.Tensor] | None:
    """Unwrap nested single-element lists and filter ``None`` entries.

    The joint dataloader can produce data as nested single-element lists
    (e.g. ``[[t1], [None], [t2]]``).  This helper flattens the nesting,
    drops ``None`` entries, and moves the remaining tensors to the target
    device/dtype.

    Args:
        raw: The raw value from ``data_batch``.  May be ``None``, a bare
            tensor, or a (possibly nested) list of tensors / ``None`` s.
            Each tensor in the list has shape ``(...)``.
        to_kwargs: Keyword arguments forwarded to ``torch.Tensor.to``
            (e.g. ``{"device": "cuda"}`` or ``{"device": "cuda", "dtype": torch.bfloat16}``).

    Returns:
        A dense list of device tensors each with shape ``(...)``, or ``None``
        if the input is ``None`` or every entry is ``None``.

    Examples:
        >>> unwrap_and_densify([[t1], [None], [t2]], {"device": "cuda"})
        [t1.cuda(), t2.cuda()]
        >>> unwrap_and_densify(None, {"device": "cuda"})
        None
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        return [raw.to(**to_kwargs)]
    if len(raw) > 0 and isinstance(raw[0], list):
        raw = [item[0] if isinstance(item, list) else item for item in raw]
    dense = [x.to(**to_kwargs) for x in raw if x is not None]
    return dense if dense else None
