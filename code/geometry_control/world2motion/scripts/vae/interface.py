# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import torch


class VideoTokenizerInterface(ABC):
    @abstractmethod
    def reset_dtype(self):
        """
        Reset the dtype of the model to the dtype its weights were trained with or quantized to.
        """
        pass

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        pass

    @abstractmethod
    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs) -> int:
        pass

    def get_latent_temporal_positions(
        self, num_pixel_frames: int, resolution: str | None = None, num_latent_frames: int | None = None
    ) -> torch.Tensor | None:
        """Return per-latent temporal coordinates when the tokenizer has nonuniform time semantics.

        The default ``None`` preserves legacy latent-index RoPE behavior. Tokenizers
        with boundary or overlap latents can override this to expose one coordinate
        per latent frame.
        """
        del num_pixel_frames, resolution, num_latent_frames
        return None

    @property
    @abstractmethod
    def spatial_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def temporal_compression_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def spatial_resolution(self) -> int:
        pass

    @property
    @abstractmethod
    def pixel_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_chunk_duration(self):
        pass

    @property
    @abstractmethod
    def latent_ch(self) -> int:
        pass

    def compile_encode(
        self,
        warmup_resolutions: Sequence[str],
        output_dir: str,
        aspect_ratio: str | None = None,
        backend: str | None = None,
        mode: str | None = None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
    ) -> None:
        """Compile the tokenizer for the given resolutions.

        Subclasses that support AOT compilation should override this method.
        The default raises ``NotImplementedError``.

        Args:
            warmup_resolutions: Resolution keys to compile for.
            output_dir: Root directory where compiled artifacts are stored
                (typically ``config.job.path_local``).
            aspect_ratio: If given, only compile this single aspect ratio.
            --- Only used if the tokenizer does not support AOT compilation ---
            backend: Backend to use for compilation.
            mode: Mode to use for compilation.
            fullgraph: Whether to compile the full graph.
            dynamic: Whether to compile the dynamic graph.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support compilation")

    @property
    def is_chunk_overlap(self):
        return False

    @property
    def is_causal(self) -> bool:
        return getattr(self, "_causal", True)

    def __init__(self, object_store_credential_path_pretrained=None):
        pass
