# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import math
import torch
from data.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from data.sequence_packing.types import ModalityData, PackedSequence


def pack_supertokens_temporal_causal(
    packed_seq: "PackedSequence",
    input_vision_tokens: torch.Tensor,
    input_action_tokens: torch.Tensor | None,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    latent_patch_size: int,
    temporal_compression_factor: int,
    action_dim: int,
    vision_fps: float | None = None,
    action_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    pack_action_tokens: bool = True,
) -> tuple[int, bool]:
    """Pack vision and (optionally) action tokens in supertoken order for temporal causal attention.

    Buffer layout per frame:
        pack_action_tokens=True:  [action_t (tcf), vision_t (H*W)]  — supertoken size tcf + H*W
        pack_action_tokens=False: [vision_t (H*W)]                  — supertoken size H*W

    Use ``pack_action_tokens=False`` when ``config.action_gen=False``; the resulting
    ``num_action_tokens_per_supertoken=0`` is stamped on the pack and read by the
    attention builder so NATTEN metadata stays in sync automatically.

    mRoPE layout (with actions, unified_3d_mrope only). The layout is inferred from the
    action tensor shape:
        - Whole-clip training (frame 0 is the clean conditioning frame, so
          ``real_actions`` has ``(T-1)*tcf`` rows): null action for supertoken 0, real
          actions for frames 1..T-1 with ``start_frame_offset=1`` so the last action in
          group i co-locates with vision frame i; vision uses ``start_frame_offset=0``.
        - AR generation, single frame OR chunk (every frame carries a real action, so
          ``real_actions`` has ``latent_t*tcf`` rows): vision AND action both use
          ``start_frame_offset=1``, generalizing the single-frame AR supertoken to
          ``latent_t`` frames. The caller (``pack_input_sequence_autoregressive``)
          seeds ``temporal_offset`` one frame-stride back to compensate, so the unit
          lands at the same absolute positions as the whole-clip training pack.
        - Interleaved per frame as cat([action_ids, vision_ids]).

    ``input_timestep`` is float (TF/none) or Tensor(T_max,) (DF, per-frame sigma).
    Conditioning frames are excluded from mse_loss_indexes either way.

    Returns (total_split_len, null_action_flag); null_action_flag is False when
    pack_action_tokens=False.
    """
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"
    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    tcf = temporal_compression_factor
    patches_per_frame = patch_h * patch_w
    supertoken_len = tcf + patches_per_frame if pack_action_tokens else patches_per_frame
    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()
    if pack_action_tokens and packed_seq.action is None:
        packed_seq.action = ModalityData()
    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)
    assert isinstance(packed_seq.vision.condition_mask, list)
    if pack_action_tokens:
        assert isinstance(packed_seq.action.sequence_indexes, list)
        assert isinstance(packed_seq.action.mse_loss_indexes, list)
        assert isinstance(packed_seq.action.timesteps, list)
        assert isinstance(packed_seq.action.tokens, list)
        assert isinstance(packed_seq.action.condition_mask, list)
    device = input_vision_tokens.device
    dtype = input_vision_tokens.dtype
    null_action_flag: bool
    if pack_action_tokens:
        if input_action_tokens is not None:
            if input_action_tokens.dim() == 3:
                real_actions = input_action_tokens.squeeze(0)
            else:
                real_actions = input_action_tokens
            null_tokens = torch.zeros(tcf, action_dim, device=device, dtype=real_actions.dtype)
            if real_actions.shape[0] == latent_t * tcf:
                all_action_tokens = real_actions
                null_action_flag = False
            elif real_actions.shape[0] == (latent_t - 1) * tcf:
                all_action_tokens = torch.cat([null_tokens, real_actions], dim=0)
                null_action_flag = True
            else:
                raise ValueError(
                    f"Temporal-causal action tokens must have either latent_t*tcf rows for AR chunks or (latent_t-1)*tcf rows for whole-clip training; got {real_actions.shape[0]} rows for latent_t={latent_t}, tcf={tcf}."
                )
        else:
            all_action_tokens = torch.zeros(latent_t * tcf, action_dim, device=device, dtype=dtype)
            null_action_flag = True
    else:
        assert input_action_tokens is None, (
            "pack_action_tokens=False requires input_action_tokens=None; got a non-None tensor."
        )
        null_action_flag = False
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)
    condition_set_vision = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)
    for fidx in condition_set_vision:
        vision_condition_mask[fidx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)
    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set_vision], device=device, dtype=torch.long
    )
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)
    if pack_action_tokens:
        packed_seq.action.token_shapes.append((latent_t * tcf,))
        packed_seq.action.tokens.append(all_action_tokens)
        action_condition_mask = torch.ones((latent_t * tcf, 1), device=device, dtype=dtype)
        packed_seq.action.condition_mask.append(action_condition_mask)
    curr = packed_seq.curr
    total_split_len = 0
    temporal_offset = packed_seq._mrope_temporal_offset
    effective_vision_fps = vision_fps if enable_fps_modulation else None
    all_frames_have_real_action = (
        pack_action_tokens and input_action_tokens is not None and (real_actions.shape[0] == latent_t * tcf)
    )
    vision_sfo = 1 if all_frames_have_real_action else 0
    vision_ids_flat, new_offset = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=temporal_offset,
        reset_spatial_indices=packed_seq._mrope_reset_spatial,
        fps=effective_vision_fps,
        base_fps=base_fps,
        temporal_compression_factor=tcf,
        start_frame_offset=vision_sfo,
    )
    if pack_action_tokens:
        effective_action_fps = action_fps if enable_fps_modulation else None
        fps_active = effective_action_fps is not None
        t_dtype = torch.float32 if fps_active else torch.long
        t_offset = float(temporal_offset) if fps_active else int(temporal_offset)
        null_t = torch.full((tcf,), t_offset, dtype=t_dtype)
        null_hw = torch.zeros(tcf, dtype=t_dtype)
        null_ids = torch.stack([null_t, null_hw, null_hw])

        def _real_action_ids(n_frames: int, start_frame_offset: int) -> torch.Tensor:
            flat, _ = get_3d_mrope_ids_vae_tokens(
                grid_t=n_frames * tcf,
                grid_h=1,
                grid_w=1,
                temporal_offset=temporal_offset,
                reset_spatial_indices=packed_seq._mrope_reset_spatial,
                fps=effective_action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=tcf,
                start_frame_offset=start_frame_offset,
            )
            return flat.reshape(3, n_frames, tcf)

        if all_frames_have_real_action:
            action_ids_3d = _real_action_ids(latent_t, start_frame_offset=1)
        elif latent_t > 1:
            null_ids_3d = null_ids.reshape(3, 1, tcf)
            real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)
            action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)
        else:
            action_ids_3d = null_ids.reshape(3, 1, tcf)
        vision_ids_3d = vision_ids_flat.reshape(3, latent_t, patches_per_frame)
        interleaved_ids = torch.cat([action_ids_3d, vision_ids_3d], dim=2).reshape(3, latent_t * supertoken_len)
        packed_seq.position_ids.append(interleaved_ids)
    else:
        packed_seq.position_ids.append(vision_ids_flat)
    packed_seq._mrope_temporal_offset = new_offset
    for frame_t in range(latent_t):
        if pack_action_tokens:
            action_indexes = list(range(curr, curr + tcf))
            packed_seq.action.sequence_indexes.extend(action_indexes)
            curr += tcf
            total_split_len += tcf
        frame_indexes = list(range(curr, curr + patches_per_frame))
        packed_seq.vision.sequence_indexes.extend(frame_indexes)
        curr += patches_per_frame
        total_split_len += patches_per_frame
        if frame_t not in condition_set_vision:
            packed_seq.vision.mse_loss_indexes.extend(frame_indexes)
            frame_ts = input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            packed_seq.vision.timesteps.extend([frame_ts] * patches_per_frame)
    packed_seq.curr = curr
    return (total_split_len, null_action_flag)
