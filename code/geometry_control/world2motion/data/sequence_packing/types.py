# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from scripts.runtime.data_and_condition import GenerationDataClean


@dataclass
class ModalityData:
    """Unified container for a single generation modality's data.

    This dataclass serves dual purposes:
    1. During packing: Acts as a builder, accumulating data in lists
    2. After finalize(): Holds finalized tensors ready for model consumption

    Attributes:
        sequence_indexes: Indices in the packed sequence where this modality's tokens appear.
            List during building, Tensor after finalize().
        timesteps: Diffusion timesteps for each noised token.
            List during building, Tensor after finalize().
        mse_loss_indexes: Indices where MSE loss should be computed (noised tokens only).
            List during building, Tensor after finalize().
        token_shapes: Shape metadata for each sample's tokens.
            For vision: list of (T, H, W) tuples.
            For action: list of (T,) tuples.
        tokens: The actual latent tokens. List during build, Tensor after finalize().
        condition_mask: Mask indicating clean frames (1=clean, 0=noised). Only after finalize().
        noisy_frame_indexes: Indices of noised frames. Constructed from condition_mask during
            sequence packing to reduce GPU->CPU synchronization later. Only after finalize().
        domain_id: Domain ID for multi-domain training. Only after finalize(). NOTE: only used for action modality.
        raw_action_dim: Raw action dimension. Only after finalize(). NOTE: only used for action modality.
    """

    sequence_indexes: list[int] | torch.Tensor = field(default_factory=list)
    timesteps: list[float] | torch.Tensor = field(default_factory=list)
    mse_loss_indexes: list[int] | torch.Tensor = field(default_factory=list)
    token_shapes: list = field(default_factory=list)
    tokens: list[torch.Tensor] = field(default_factory=list)
    condition_mask: list[torch.Tensor] = field(default_factory=list)
    noisy_frame_indexes: list[torch.Tensor] = field(default_factory=list)
    domain_id: list[torch.Tensor] = field(default_factory=list)
    raw_action_dim: list[torch.Tensor | None] | None = field(default_factory=list)

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.sequence_indexes, torch.Tensor):
            self.sequence_indexes = self.sequence_indexes.cuda()
        if isinstance(self.timesteps, torch.Tensor):
            self.timesteps = self.timesteps.cuda()
        if isinstance(self.mse_loss_indexes, torch.Tensor):
            self.mse_loss_indexes = self.mse_loss_indexes.cuda()
        self.tokens = [token.cuda() for token in self.tokens]
        self.condition_mask = [cm.cuda() for cm in self.condition_mask]
        self.noisy_frame_indexes = [ni.cuda() for ni in self.noisy_frame_indexes]
        self.domain_id = [d.cuda() for d in self.domain_id]
        if self.raw_action_dim is not None:
            self.raw_action_dim = [d.cuda() if d is not None else None for d in self.raw_action_dim]


@dataclass
class PackedSequence:
    """Unified sequence container - works as builder during packing and final output.

    This dataclass replaces the old SequenceStatus + PackedSequence pattern:
    - Build phase: Accumulate data using lists, modalities use ModalityData builders
    - After finalize(): Ready for model consumption with tensors

    Attributes:
        # Sequence structure
        sample_lens: Length of each sample in the packed sequence.
        split_lens: Length of each split (text/vision/action sections).
        attn_modes: Attention mode for each split ('causal', 'full').
        is_image_batch: Whether this batch contains images (vs videos).
        sequence_length: Total length of packed sequence. Computed during finalize().

        # Build-time tracking (not used after finalize)
        curr: Current position in the packed sequence during building.

        # Text modality (list during build, tensor after finalize)
        text_ids: All text token IDs (including special tokens).
        text_indexes: Indices where text tokens appear in sequence.
        position_ids: RoPE position IDs for all tokens.

        # Loss computation - Cross Entropy (text)
        label_ids: Label IDs for cross-entropy loss.
        ce_loss_indexes: Indices for computing cross-entropy loss.
        ce_loss_weights: Weights for cross-entropy loss.

        # Generation modalities - named fields for type safety
        vision: Vision modality data (images/videos). None if no vision in batch.
        action: Action modality data (robotics). None if no actions in batch.
        sound: Sound modality data (audio). None if no sound in batch.
    """

    sample_lens: list[int] = field(default_factory=list)
    split_lens: list[int] = field(default_factory=list)
    attn_modes: list[str] = field(default_factory=list)
    is_image_batch: bool = False
    sequence_length: int = 0
    curr: int = 0
    text_ids: list[int] | torch.Tensor = field(default_factory=list)
    text_indexes: list[int] | torch.Tensor = field(default_factory=list)
    position_ids: list[torch.Tensor] | torch.Tensor = field(default_factory=list)
    label_ids: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_indexes: list[int] | torch.Tensor | None = field(default_factory=list)
    ce_loss_weights: list[float] | torch.Tensor | None = field(default_factory=list)
    _mrope_temporal_offset: int | float = 0
    _mrope_reset_spatial: bool = True
    null_action_supertokens: bool = False
    num_action_tokens_per_supertoken: int = 0
    vision: ModalityData | None = None
    action: ModalityData | None = None
    sound: ModalityData | None = None
    vision_item_split_lens: list[list[int]] = field(default_factory=list)
    control_weights: list[list[float]] | None = None

    def finalize(self, gen_data_clean: GenerationDataClean) -> "PackedSequence":
        """Convert all lists to tensors and compute derived values.

        Args:
            gen_data_clean: GenerationDataClean for metadata (e.g., action domain IDs).

        Returns:
            New PackedSequence instance with tensors instead of lists.
        """
        sequence_length = sum(self.sample_lens)
        sample_lens = self.sample_lens.copy()
        split_lens = self.split_lens.copy()
        attn_modes = self.attn_modes.copy()
        label_ids: torch.Tensor | None = None
        ce_loss_indexes: torch.Tensor | None = None
        ce_loss_weights: torch.Tensor | None = None
        if self.label_ids and len(self.label_ids) > 0:
            label_ids = torch.tensor(self.label_ids)
            ce_loss_indexes = torch.tensor(self.ce_loss_indexes)
            ce_loss_weights = torch.tensor(self.ce_loss_weights)
        vision: ModalityData | None = None
        if self.vision is not None and len(self.vision.sequence_indexes) > 0:
            vision = ModalityData(
                sequence_indexes=torch.tensor(self.vision.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.vision.timesteps, dtype=torch.float32),
                mse_loss_indexes=torch.tensor(self.vision.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.vision.token_shapes),
                tokens=self.vision.tokens,
                condition_mask=list(self.vision.condition_mask),
                noisy_frame_indexes=list(self.vision.noisy_frame_indexes),
            )
        action: ModalityData | None = None
        if self.action is not None and len(self.action.sequence_indexes) > 0:
            action = ModalityData(
                sequence_indexes=torch.tensor(self.action.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.action.timesteps, dtype=torch.float32),
                mse_loss_indexes=torch.tensor(self.action.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.action.token_shapes),
                tokens=self.action.tokens,
                condition_mask=list(self.action.condition_mask),
                noisy_frame_indexes=list(self.action.noisy_frame_indexes),
                domain_id=gen_data_clean.action_domain_id
                if gen_data_clean.action_domain_id is not None
                else [torch.zeros(1, dtype=torch.long)] * len(self.action.token_shapes),
                raw_action_dim=gen_data_clean.raw_action_dim,
            )
        sound: ModalityData | None = None
        if self.sound is not None and len(self.sound.sequence_indexes) > 0:
            sound = ModalityData(
                sequence_indexes=torch.tensor(self.sound.sequence_indexes, dtype=torch.long),
                timesteps=torch.tensor(self.sound.timesteps, dtype=torch.float32),
                mse_loss_indexes=torch.tensor(self.sound.mse_loss_indexes, dtype=torch.long),
                token_shapes=list(self.sound.token_shapes),
                tokens=self.sound.tokens,
                condition_mask=list(self.sound.condition_mask),
                noisy_frame_indexes=list(self.sound.noisy_frame_indexes),
            )
        assert isinstance(self.position_ids, list)
        if len(self.position_ids) > 0:
            position_ids = torch.cat(self.position_ids, dim=1)
        else:
            position_ids = torch.empty((3, 0), dtype=torch.long)
        return PackedSequence(
            sequence_length=sequence_length,
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            is_image_batch=gen_data_clean.is_image_batch,
            text_ids=torch.tensor(self.text_ids, dtype=torch.long),
            text_indexes=torch.tensor(self.text_indexes, dtype=torch.long),
            position_ids=position_ids,
            label_ids=label_ids,
            ce_loss_indexes=ce_loss_indexes,
            ce_loss_weights=ce_loss_weights,
            vision=vision,
            action=action,
            sound=sound,
            null_action_supertokens=self.null_action_supertokens,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
            vision_item_split_lens=list(self.vision_item_split_lens),
            control_weights=gen_data_clean.control_weights,
        )

    def to_cuda(self) -> None:
        """Move all tensor fields to CUDA in-place."""
        if isinstance(self.text_ids, torch.Tensor):
            self.text_ids = self.text_ids.cuda()
        if isinstance(self.text_indexes, torch.Tensor):
            self.text_indexes = self.text_indexes.cuda()
        if isinstance(self.position_ids, torch.Tensor):
            self.position_ids = self.position_ids.cuda()
        if isinstance(self.label_ids, torch.Tensor):
            self.label_ids = self.label_ids.cuda()
        if isinstance(self.ce_loss_indexes, torch.Tensor):
            self.ce_loss_indexes = self.ce_loss_indexes.cuda()
        if isinstance(self.ce_loss_weights, torch.Tensor):
            self.ce_loss_weights = self.ce_loss_weights.cuda()
        if self.vision is not None:
            self.vision.to_cuda()
        if self.action is not None:
            self.action.to_cuda()
        if self.sound is not None:
            self.sound.to_cuda()


@dataclass
class SequencePlan:
    """Plan describing which modalities are present in a sample.

    This dataclass tracks the presence of different modalities (text, vision, action)
    and their conditioning configurations for a dataset sample. Unlike SequencePlan
    which holds the actual tensor data, this class provides a lightweight summary
    of what modalities exist and how they should be conditioned.

    Attributes:
        has_text: Whether text/caption tokens are present for this sample.
            Used for text-conditioned generation (e.g., text-to-image/video).
        has_vision: Whether vision input (image or video latents) is present.
            Defaults to False.
        condition_frame_indexes_vision: Indexes of latent vision frames that are clean/conditioning.
            [] means all frames are noised/supervised.
            All frames specified means all frames are clean (no MSE supervision).
            For multi-item samples (e.g. image editing where each sample has multiple
            separately-encoded images), this applies to each vision item individually.
            The number of items per sample is tracked by
            ``GenerationDataClean.num_vision_items_per_sample``.
        has_action: Whether action input is present for robotics/embodied AI tasks.
            Defaults to False.
        condition_frame_indexes_action: Indexes of action steps that are clean/conditioning.
            [] means all steps are noised/supervised.
            All steps specified means all steps are clean (no MSE supervision).
    """

    has_text: bool
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    share_vision_temporal_positions: bool = False
    has_action: bool = False
    condition_frame_indexes_action: list[int] = field(default_factory=list)
    action_start_frame_offset: int = 1
    has_sound: bool = False
    condition_frame_indexes_sound: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_action": self.has_action,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
            "condition_frame_indexes_action": self.condition_frame_indexes_action,
            "condition_frame_indexes_sound": self.condition_frame_indexes_sound,
            "share_vision_temporal_positions": self.share_vision_temporal_positions,
        }


def build_sequence_plans_from_data_batch(
    data_batch: dict, input_video_key, input_image_key: str
) -> list[SequencePlan]:
    """Build or retrieve sequence plans from a data batch dictionary.

    This function extracts sequence plans from the data batch if they exist,
    otherwise creates default SequencePlan objects for each sample
    in the batch.

    Args:
        data_batch: Dictionary containing the data batch from the dataloader.
            Expected keys include 'video' or other tensors to determine batch size.
            If 'sequence_plan' key exists, those plans are returned directly.

    Returns:
        List of SequencePlan objects, one per sample in the batch.
    """
    if "sequence_plan" in data_batch:
        return data_batch["sequence_plan"]
    assert "action" not in data_batch or data_batch["action"] is None, "Action data SHOULD have sequence_plans!"
    assert "sound" not in data_batch or data_batch["sound"] is None, "Sound data SHOULD have sequence_plans!"
    batch_size = 0
    for key in [input_video_key, input_image_key]:
        if key in data_batch:
            val = data_batch[key]
            if isinstance(val, torch.Tensor):
                batch_size = val.shape[0]
                break
            elif isinstance(val, list):
                batch_size = len(val)
                break
    if batch_size == 0:
        raise ValueError(
            f"Cannot determine batch size from data_batch. Expected {input_video_key}, {input_image_key}, or similar key."
        )
    return [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]) for _ in range(batch_size)]
