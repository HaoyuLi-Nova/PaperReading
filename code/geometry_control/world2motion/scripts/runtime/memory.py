# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

KVToStore = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MemoryValue(ABC):
    """Read-only tensor container safe to pass into ``torch.compile``.

    Concrete subclasses (e.g. ``ARMemoryValue``, ``KVTrainMemoryValue``)
    are plain dataclasses of tensors.  No methods on this class should
    mutate state or perform non-trivial computation.
    """

    @property
    def supports_context_parallel_attention(self) -> bool:
        """Whether this memory value is compatible with context-parallel attention.

        Overridden by ``KVTrainMemoryValue`` to return ``False``.  Used by
        ``ContextParallelDispatch`` to reject an unsupported combination
        without importing the concrete subclass.
        """
        return True


class MemoryState(ABC):
    """Mutable persistent memory that lives outside ``torch.compile``.

    The outer loop in ``_impl_forward`` calls ``read_for_layer`` before
    each decoder layer and ``write_for_layer`` after.  The ``MemoryState``
    object itself is **never** passed into a compiled region.
    """

    @abstractmethod
    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialization per training step.

        Called once before any transformer layers are processed.

        Args:
            hidden_states: The packed sequence (``SequencePack``).
            device: Target device for any new tensors.
        """

    @abstractmethod
    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        """Produce a read-only tensor snapshot for *layer_idx*.

        Used to retrieve KV values from the cache.
        The returned ``MemoryValue`` is passed into the compiled decoder
        layer as ``memory_value``.
        """

    @abstractmethod
    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Store the K/V tensors produced by *layer_idx* back into the cache.

        Called outside the ``torch.compile`` boundary.
        """

    @abstractmethod
    def is_gen_only(self) -> bool:
        """Return ``True`` when only the generation pathway should run.

        When ``True``, the decoder layer assumes that the text caption has
        already been processed and cached in the MemoryState object.
        Used for autoregressive frame-by-frame generation of video.
        """

    def requires_natten_metadata(self) -> bool:
        """Whether the packed-sequence builder should create NATTEN metadata.

        Memory paths whose attention implementation handles temporal
        visibility itself return ``False``.
        """
        return True
