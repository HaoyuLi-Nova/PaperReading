# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
from scripts.utils import log, misc
from scripts.sampling.unipc import UniPCSampler, UniPCSamplerConfig
from scripts.qwen.utils import tokenize_caption
from scripts.runtime.data_and_condition import GenerationDataClean, unwrap_and_densify
from scripts.runtime.memory import MemoryState
from data.sequence_packing import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
    pack_input_sequence,
)
from data.resolution_utils import get_vision_data_resolution
import torch
from scripts.sampling.unipc import UniPCSampler, UniPCSamplerConfig


class InferenceCore:
    def __init__(self, net, vae, tokenizer, special_tokens, config):
        self.net = net
        self.tokenizer_vision_gen = vae
        self.vlm_tokenizer = tokenizer
        self.llm_special_tokens = special_tokens
        self.config = config
        self.vlm_config = config.vlm_config
        self.input_caption_key = "ai_caption"
        self.input_video_key = "video"
        self.input_image_key = "images"
        self.precision = torch.bfloat16
        self.tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}
        self.tensor_kwargs_fp32 = {"device": "cuda", "dtype": torch.float32}
        self.sampler = UniPCSampler(UniPCSamplerConfig(), self.tensor_kwargs)

    def _derive_include_end_of_generation_token(self) -> bool:
        impl = self.config.joint_attn_implementation
        assert impl in ("two_way", "three_way"), (
            f"Invalid joint_attn_implementation: {impl}. Must be 'two_way' or 'three_way'."
        )
        return False

    def _pack_input_sequence(
        self,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        gen_data_clean: GenerationDataClean,
        input_timesteps: torch.Tensor,
        include_end_of_generation_token: bool = False,
        skip_text_tokens: bool = False,
        initial_mrope_temporal_offset: int | float = 0,
    ) -> PackedSequence:
        """Wrap ``pack_input_sequence`` with all config-derived args pre-filled.

        Centralises the config-derived positional/embedding args so callers only
        supply the four per-call arguments (sequence_plans, text tokens, data, timesteps)
        plus three optional flags.
        """
        assert self.tokenizer_vision_gen is not None
        return pack_input_sequence(
            sequence_plans=sequence_plans,
            input_text_indexes=input_text_indexes,
            gen_data_clean=gen_data_clean,
            input_timesteps=input_timesteps,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
            skip_text_tokens=skip_text_tokens,
            include_end_of_generation_token=include_end_of_generation_token,
            unified_3d_mrope_reset_spatial_ids=self.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=float(self.config.diffusion_expert_config.base_fps),
            sound_base_temporal_compression_factor=self.config.diffusion_expert_config.sound_base_temporal_compression_factor,
            temporal_compression_factor=self.tokenizer_vision_gen.temporal_compression_factor,
            vision_temporal_position_mode=self.config.diffusion_expert_config.vision_temporal_position_mode,
            video_temporal_causal=self.config.video_temporal_causal,
            action_dim=self.config.max_action_dim,
            initial_mrope_temporal_offset=initial_mrope_temporal_offset,
        )

    def _get_temporal_positions_vision(
        self, raw_state_vision: list[torch.Tensor], x0_tokens_vision: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        """Return optional per-latent temporal coordinates for vision tokens."""
        mode = self.config.diffusion_expert_config.vision_temporal_position_mode
        if mode == "latent_index":
            return None
        if mode != "uniae_source_right_edge":
            raise ValueError(
                f"Unsupported vision_temporal_position_mode: {mode}. Expected 'latent_index' or 'uniae_source_right_edge'."
            )
        assert self.tokenizer_vision_gen is not None
        temporal_positions_vision: list[torch.Tensor] = []
        for raw_state_vision_i, x0_tokens_vision_i in zip(raw_state_vision, x0_tokens_vision, strict=True):
            if raw_state_vision_i.dim() == 5:
                num_pixel_frames = int(raw_state_vision_i.shape[2])
            elif raw_state_vision_i.dim() == 4:
                num_pixel_frames = int(raw_state_vision_i.shape[1])
            else:
                raise ValueError(
                    f"raw_state_vision items must have shape [B,C,T,H,W] or [C,T,H,W], got shape {tuple(raw_state_vision_i.shape)}."
                )
            num_latent_frames = int(x0_tokens_vision_i.shape[2])
            frame_h = int(raw_state_vision_i.shape[-2])
            frame_w = int(raw_state_vision_i.shape[-1])
            resolution = get_vision_data_resolution((frame_h, frame_w))
            temporal_positions = self.tokenizer_vision_gen.get_latent_temporal_positions(
                num_pixel_frames=num_pixel_frames, resolution=resolution, num_latent_frames=num_latent_frames
            )
            if temporal_positions is None:
                raise ValueError(
                    f"{type(self.tokenizer_vision_gen).__name__} does not support vision_temporal_position_mode={mode}."
                )
            if temporal_positions.shape[0] != num_latent_frames:
                raise ValueError(
                    f"Vision temporal position count must match latent frames: got {temporal_positions.shape[0]} positions for {num_latent_frames} latent frames."
                )
            temporal_positions = temporal_positions.to(device=x0_tokens_vision_i.device, dtype=torch.float32)
            temporal_positions_vision.append(temporal_positions)
        return temporal_positions_vision

    def _get_inference_text_tokens(
        self, data_batch: dict, has_negative_prompt: bool
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Tokenize conditional and unconditional captions for inference.

        Delegates the per-caption chat-template tokenization to
        :meth:`_tokenize_captions` (the same helper backing the public
        :meth:`tokenize_text`) so there is a single source of truth for
        how raw captions become token ids.
        """
        use_system_prompt = self.vlm_config.use_system_prompt
        system_prompt: str | None = data_batch.get("system_prompt")
        cond_tokens = self._tokenize_captions(
            data_batch[self.input_caption_key],
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            is_video=False,
        )
        if has_negative_prompt:
            neg_key = "neg_" + self.input_caption_key
            if neg_key not in data_batch:
                raise ValueError(f"Negative prompt ({neg_key}) not found")
            uncond_captions = data_batch[neg_key]
        else:
            uncond_captions = [""] * len(cond_tokens)
        uncond_tokens = self._tokenize_captions(
            uncond_captions, use_system_prompt=use_system_prompt, system_prompt=system_prompt, is_video=False
        )
        return (cond_tokens, uncond_tokens)

    def _prepare_inference_data(
        self, data_batch: dict, seed: list[int], has_negative_prompt: bool = False
    ) -> tuple[
        list[SequencePlan],
        GenerationDataClean,
        list[list[int]],
        list[list[int]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Prepare all data needed for inference sampling.
        Mirrors training_step's data preparation flow.

        This method:
        1. Builds sequence plans (conditioning information)
        2. Gets data and condition (encodes vision)
        3. Tokenizes text (conditional and unconditional for CFG)
        4. Builds a packed sequence to fetch conditioning masks
        5. Initializes noise with conditioning applied (as lists for variable shapes)
        6. If action_gen is True, concatenates action noise with vision noise

        Args:
            data_batch: Raw data batch from dataloader.
            seed: Random seed(s) for noise generation.
            has_negative_prompt: If True, use negative prompt for unconditional branch.

        Returns:
            Tuple of:
                - sequence_plans: List of SequencePlan objects
                - gen_data_clean: GenerationDataClean with encoded tokens
                - cond_text_tokens: Conditional text tokens
                - uncond_text_tokens: Unconditional text tokens (for CFG)
                - initial_noise: List of noise tensors (one per sample), each containing
                  flattened vision (and optionally action) noise concatenated
                - condition_reference: List of clean reference tensors flattened
                  in the same order as initial_noise
                - condition_mask: List of masks flattened in the same order as
                  initial_noise, where 1 keeps condition_reference values fixed
        """
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch, input_video_key=self.input_video_key, input_image_key=self.input_image_key
        )
        gen_data_clean = self.get_data_and_condition(data_batch)
        num_items_per_sample = gen_data_clean.num_vision_items_per_sample
        cond_text_tokens, uncond_text_tokens = self._get_inference_text_tokens(data_batch, has_negative_prompt)
        mask_timesteps = torch.zeros((gen_data_clean.batch_size,), dtype=torch.float32)
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            cond_text_tokens,
            gen_data_clean,
            mask_timesteps,
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
        )
        assert packed_sequence.vision is not None, "Packed vision data required for inference noise"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for inference noise"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for inference noise"
        )
        assert gen_data_clean.x0_tokens_vision is not None, "Vision data required for inference noise"
        n_sample = (
            len(gen_data_clean.x0_tokens_vision)
            if gen_data_clean.num_vision_items_per_sample is None
            else len(gen_data_clean.num_vision_items_per_sample)
        )
        assert len(seed) == n_sample, (
            f"Seed list length {len(seed)} must have the same length as the number of samples {n_sample}"
        )
        seed_dict = {"vision": [], "action": [], "sound": []}
        for sample_idx in range(n_sample):
            num_vision_items = num_items_per_sample[sample_idx] if num_items_per_sample is not None else 1
            seed_dict["vision"].extend([seed[sample_idx]] * num_vision_items)
            seed_dict["action"].append(seed[sample_idx])
            seed_dict["sound"].append(seed[sample_idx])
        noise_vision_list: list[torch.Tensor] = []
        for i, (x0_token, cond_mask) in enumerate(
            zip(gen_data_clean.x0_tokens_vision, packed_sequence.vision.condition_mask, strict=True)
        ):
            pure_noise_i = misc.arch_invariant_rand(
                tuple(x0_token.shape),
                self.tensor_kwargs_fp32["dtype"],
                self.tensor_kwargs_fp32["device"],
                seed_dict["vision"][i],
            )
            noise_i = cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i
            noise_vision_list.append(noise_i)
        has_action = True and any((plan.has_action for plan in sequence_plans))
        noise_action_list: list[torch.Tensor] | None = None
        if has_action:
            assert gen_data_clean.x0_tokens_action is not None, "Action data required when sequence plan has action"
            assert packed_sequence.action is not None, "Packed action data required when action_gen is True"
            assert packed_sequence.action.condition_mask is not None, "Action condition mask required"
            assert isinstance(packed_sequence.action.condition_mask, list), (
                "Action condition mask must be a list of tensors for inference noise"
            )
            noise_action_list = []
            for i, (x0_action, cond_mask_action) in enumerate(
                zip(gen_data_clean.x0_tokens_action, packed_sequence.action.condition_mask, strict=True)
            ):
                pure_noise_action_i = misc.arch_invariant_rand(
                    tuple(x0_action.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["action"][i],
                )
                noise_action_i = (
                    cond_mask_action * x0_action.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_action) * pure_noise_action_i
                )
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    noise_action_i[:, gen_data_clean.raw_action_dim[i] :] = 0
                noise_action_list.append(noise_action_i)
        has_sound = False
        noise_sound_list: list[torch.Tensor] | None = None
        if has_sound:
            assert gen_data_clean.x0_tokens_sound is not None, "Sound data required when sequence plan has sound"
            assert packed_sequence.sound is not None, "Packed sound data required when sound_gen is True"
            assert packed_sequence.sound.condition_mask is not None, "Sound condition mask required"
            assert isinstance(packed_sequence.sound.condition_mask, list), (
                "Sound condition mask must be a list of tensors for inference noise"
            )
            noise_sound_list = []
            for i, (x0_sound, cond_mask_sound) in enumerate(
                zip(gen_data_clean.x0_tokens_sound, packed_sequence.sound.condition_mask, strict=True)
            ):
                pure_noise_sound_i = misc.arch_invariant_rand(
                    tuple(x0_sound.shape),
                    self.tensor_kwargs["dtype"],
                    self.tensor_kwargs["device"],
                    seed_dict["sound"][i],
                )
                noise_sound_i = (
                    cond_mask_sound.T * x0_sound.to(**self.tensor_kwargs)
                    + (1.0 - cond_mask_sound.T) * pure_noise_sound_i
                )
                noise_sound_list.append(noise_sound_i)
        initial_noise: list[torch.Tensor] = []
        condition_reference: list[torch.Tensor] = []
        condition_mask: list[torch.Tensor] = []
        idx_vision = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_sample):
            parts = []
            condition_reference_parts = []
            condition_mask_parts = []
            num_vis = num_items_per_sample[i] if num_items_per_sample is not None else 1
            for _ in range(num_vis):
                parts.append(noise_vision_list[idx_vision].reshape(-1))
                x0_vision = gen_data_clean.x0_tokens_vision[idx_vision]
                mask_vision = packed_sequence.vision.condition_mask[idx_vision].to(
                    dtype=x0_vision.dtype, device=x0_vision.device
                )
                condition_reference_parts.append(x0_vision.reshape(-1))
                condition_mask_parts.append((mask_vision * torch.ones_like(x0_vision)).reshape(-1))
                idx_vision += 1
            if noise_action_list is not None and sequence_plans[i].has_action:
                assert packed_sequence.action is not None
                assert packed_sequence.action.condition_mask is not None
                assert gen_data_clean.x0_tokens_action is not None
                parts.append(noise_action_list[idx_action].reshape(-1))
                x0_action = gen_data_clean.x0_tokens_action[idx_action].to(
                    dtype=noise_action_list[idx_action].dtype, device=noise_action_list[idx_action].device
                )
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[idx_action] is not None:
                    x0_action = x0_action.clone()
                    x0_action[:, gen_data_clean.raw_action_dim[idx_action] :] = 0
                mask_action = packed_sequence.action.condition_mask[idx_action].to(
                    dtype=x0_action.dtype, device=x0_action.device
                )
                condition_reference_parts.append(x0_action.reshape(-1))
                condition_mask_parts.append((mask_action * torch.ones_like(x0_action)).reshape(-1))
                idx_action += 1
            if noise_sound_list is not None and sequence_plans[i].has_sound:
                assert packed_sequence.sound is not None
                assert packed_sequence.sound.condition_mask is not None
                assert gen_data_clean.x0_tokens_sound is not None
                parts.append(noise_sound_list[idx_sound].reshape(-1))
                x0_sound = gen_data_clean.x0_tokens_sound[idx_sound].to(
                    dtype=noise_sound_list[idx_sound].dtype, device=noise_sound_list[idx_sound].device
                )
                mask_sound = packed_sequence.sound.condition_mask[idx_sound].T.to(
                    dtype=x0_sound.dtype, device=x0_sound.device
                )
                condition_reference_parts.append(x0_sound.reshape(-1))
                condition_mask_parts.append((mask_sound * torch.ones_like(x0_sound)).reshape(-1))
                idx_sound += 1
            initial_noise.append(torch.cat(parts, dim=0))
            condition_reference.append(torch.cat(condition_reference_parts, dim=0))
            condition_mask.append(torch.cat(condition_mask_parts, dim=0))
        return (
            sequence_plans,
            gen_data_clean,
            cond_text_tokens,
            uncond_text_tokens,
            initial_noise,
            condition_reference,
            condition_mask,
        )

    def _get_velocity(
        self,
        *,
        net: torch.nn.Module | None = None,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        text_tokens: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        skip_text_tokens: bool = False,
        timestep_action: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Compute velocity prediction for a single sampling step.

        This method handles the full pipeline for one denoising step:
        1. Splits flattened noise_x into vision (and action) parts per sample
        2. Packs the input sequence with current noisy latents
        3. Runs the network via self.denoise()
        4. Applies velocity masks (zeroes out conditioned parts)
        5. Returns flattened velocities (concatenated vision + action per sample)

        Args:
            noise_x: List of noisy latents, each containing concatenated
                     vision (and optionally action) noise.
                     len(noise_x) == B, noise_x[i] is shape (D)
            timestep: Current timestep for each sample
            text_tokens: Tokenized text for each sample
            sequence_plans: Pre-computed sequence plans (from _prepare_inference_data)
            gen_data_clean: Pre-computed clean data (from _prepare_inference_data)
            skip_text_tokens: If True, skip text tokens (for CFG unconditional branch)
            timestep_action: Optional per-sample ``[B,1]`` timestep override for the ACTION
                segment (dual-schedule / modality-forcing inference). When set, the packed
                action timesteps injected by the packer (vision timestep) are overwritten so
                the denoiser's action timestep embedding matches the action segment's own
                sigma trajectory — mirroring the training-side override under
                ``independent_action_schedule=True``.

        Returns:
            Stacked flattened velocity tensors (one per sample), each containing
            concatenated vision (and optionally action) velocity
        """
        n_samples = len(noise_x)
        is_image_batch = gen_data_clean.is_image_batch
        has_action = True and any((plan.has_action for plan in sequence_plans))
        num_items = gen_data_clean.num_vision_items_per_sample
        has_sound = False
        noise_x_vision: list[torch.Tensor] = []
        noise_x_action: list[torch.Tensor] | None = [] if has_action else None
        noise_x_sound: list[torch.Tensor] | None = [] if has_sound else None
        vision_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            n_vis = num_items[i] if num_items is not None else 1
            offset = 0
            for j in range(n_vis):
                vision_shape = gen_data_clean.x0_tokens_vision[vision_offset + j].shape
                vision_dim = int(torch.prod(torch.tensor(vision_shape)))
                noise_vision_ij = noise_x[i][offset : offset + vision_dim].reshape(vision_shape)
                noise_x_vision.append(noise_vision_ij)
                offset += vision_dim
            vision_offset += n_vis
            if has_action and noise_x_action is not None:
                assert gen_data_clean.x0_tokens_action is not None
                action_shape = gen_data_clean.x0_tokens_action[idx_action].shape
                action_dim = int(torch.prod(torch.tensor(action_shape)))
                noise_x_action.append(noise_x[i][offset : offset + action_dim].reshape(action_shape))
                offset += action_dim
                idx_action += 1
            if has_sound and noise_x_sound is not None and sequence_plans[i].has_sound:
                assert gen_data_clean.x0_tokens_sound is not None
                sound_shape = gen_data_clean.x0_tokens_sound[idx_sound].shape
                sound_dim = int(torch.prod(torch.tensor(sound_shape)))
                noise_x_sound.append(noise_x[i][offset : offset + sound_dim].reshape(sound_shape))
                offset += sound_dim
                idx_sound += 1
        gen_data_for_packing = GenerationDataClean(
            batch_size=n_samples,
            is_image_batch=is_image_batch,
            raw_state_vision=gen_data_clean.raw_state_vision,
            x0_tokens_vision=noise_x_vision,
            fps_vision=gen_data_clean.fps_vision,
            temporal_positions_vision=gen_data_clean.temporal_positions_vision,
            raw_state_action=gen_data_clean.raw_state_action if has_action else None,
            x0_tokens_action=noise_x_action if has_action else None,
            action_domain_id=gen_data_clean.action_domain_id if has_action else None,
            fps_action=gen_data_clean.fps_action if has_action else None,
            raw_action_dim=gen_data_clean.raw_action_dim if has_action else None,
            raw_state_sound=gen_data_clean.raw_state_sound if has_sound else None,
            x0_tokens_sound=noise_x_sound if has_sound else None,
            fps_sound=gen_data_clean.fps_sound if has_sound else None,
            num_vision_items_per_sample=num_items,
            control_weights=gen_data_clean.control_weights,
        )
        packed_sequence = self._pack_input_sequence(
            sequence_plans,
            text_tokens,
            gen_data_for_packing,
            timestep.cpu(),
            include_end_of_generation_token=self._derive_include_end_of_generation_token(),
            skip_text_tokens=skip_text_tokens,
        )
        if packed_sequence.vision is not None:
            packed_sequence.vision.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_vision]
        if has_action and noise_x_action is not None:
            assert packed_sequence.action is not None, "packed_sequence.action must exist when has_action is True"
            packed_sequence.action.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_action]
            packed_sequence.action.domain_id = gen_data_clean.action_domain_id
        if timestep_action is not None and has_action and (packed_sequence.action is not None):
            dense_ts = [timestep_action[i, 0] for i, plan in enumerate(sequence_plans) if plan.has_action]
            if any((nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes)):
                packed_sequence.action.timesteps = torch.cat(
                    [
                        dense_ts[j].reshape(1).cpu().expand(nfi.numel())
                        for j, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
                    ]
                ).to(dtype=torch.float32)
        if has_sound and noise_x_sound is not None:
            assert packed_sequence.sound is not None, "packed_sequence.sound must exist when has_sound is True"
            packed_sequence.sound.tokens = [x.to(**self.tensor_kwargs) for x in noise_x_sound]
        packed_sequence.to_cuda()
        out = self.denoise(net=net, data_batch_packed=packed_sequence)
        assert packed_sequence.vision is not None, "packed_sequence.vision must exist for velocity masking"
        assert packed_sequence.vision.condition_mask is not None, "Vision condition mask required for masking"
        assert isinstance(packed_sequence.vision.condition_mask, list), (
            "Vision condition mask must be a list of tensors for masking"
        )
        noisy_mask_vision = [1.0 - cond_mask for cond_mask in packed_sequence.vision.condition_mask]
        velocity_vision: list[torch.Tensor] = []
        for i, (pred, noisy_mask) in enumerate(zip(out["preds_vision"], noisy_mask_vision)):
            has_noisy_tokens_i = noisy_mask.sum() > 0
            if has_noisy_tokens_i:
                velocity_vision.append(pred * noisy_mask.to(dtype=pred.dtype, device=pred.device))
            else:
                velocity_vision.append(torch.zeros_like(pred))
        velocity_action: list[torch.Tensor] | None = None
        if (
            has_action
            and packed_sequence.action is not None
            and (packed_sequence.action.condition_mask is not None)
            and isinstance(packed_sequence.action.condition_mask, list)
        ):
            noisy_mask_action = [1.0 - cond_mask for cond_mask in packed_sequence.action.condition_mask]
            velocity_action = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_action"], noisy_mask_action)):
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    v = pred * noisy_mask.to(dtype=pred.dtype, device=pred.device)
                else:
                    v = torch.zeros_like(pred)
                if gen_data_clean.raw_action_dim is not None and gen_data_clean.raw_action_dim[i] is not None:
                    v[:, gen_data_clean.raw_action_dim[i] :] = 0
                velocity_action.append(v)
        velocity_sound: list[torch.Tensor] | None = None
        if (
            has_sound
            and packed_sequence.sound is not None
            and (packed_sequence.sound.condition_mask is not None)
            and isinstance(packed_sequence.sound.condition_mask, list)
        ):
            noisy_mask_sound = [1.0 - cond_mask for cond_mask in packed_sequence.sound.condition_mask]
            velocity_sound = []
            for i, (pred, noisy_mask) in enumerate(zip(out["preds_sound"], noisy_mask_sound)):
                has_noisy_tokens_i = noisy_mask.sum() > 0
                if has_noisy_tokens_i:
                    velocity_sound.append(pred * noisy_mask.T.to(dtype=pred.dtype, device=pred.device))
                else:
                    velocity_sound.append(torch.zeros_like(pred))
        velocity_output: list[torch.Tensor] = []
        vis_offset = 0
        idx_action = 0
        idx_sound = 0
        for i in range(n_samples):
            parts = []
            n_vis = num_items[i] if num_items is not None else 1
            for _ in range(n_vis):
                parts.append(velocity_vision[vis_offset].reshape(-1))
                vis_offset += 1
            if velocity_action is not None and sequence_plans[i].has_action:
                parts.append(velocity_action[idx_action].reshape(-1))
                idx_action += 1
            if velocity_sound is not None and sequence_plans[i].has_sound:
                parts.append(velocity_sound[idx_sound].reshape(-1))
                idx_sound += 1
            velocity_output.append(torch.cat(parts, dim=0))
        return velocity_output

    def _remove_padding_from_latent(
        self, x0_tokens_vision: list[torch.Tensor], frame_size: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Remove reflection padding from encoded latent vision tokens.

        Each sample in the batch may have different original dimensions, so we process
        each sample individually and return a list of latents with varying spatial sizes.

        The padding coordinates are scaled down by the spatial compression factor since
        we're operating in latent space.

        Args:
            x0_tokens_vision (list[torch.Tensor]): List of encoded latent tensors,
                each of shape (1, C, T, H_latent, W_latent)
                where H_latent, W_latent include scaled padding.
            frame_size (list[torch.Tensor]): List of tensors, each of shape (1,4) or (4,) containing
                [target_h, target_w, orig_h, orig_w] for each sample (in pixel space).

        Returns:
            list[torch.Tensor]: List of cropped latent tokens, each of shape (1, C, T, H_latent_cropped, W_latent_cropped).
                Each element may have different spatial sizes based on original image dimensions.
        """
        batch_size = len(x0_tokens_vision)
        spatial_factor = self.tokenizer_vision_gen.spatial_compression_factor
        cropped_latents = []
        for i in range(batch_size):
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())
            if orig_h // spatial_factor == 0 or orig_w // spatial_factor == 0:
                log.warning(
                    f"Zero-sized latent found: orig_h: {orig_h}, orig_w: {orig_w}, spatial_factor: {spatial_factor}"
                )
            orig_h_latent = max(orig_h // spatial_factor, 1)
            orig_w_latent = max(orig_w // spatial_factor, 1)
            cropped_latent = x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous()
            cropped_latents.append(cropped_latent)
        return cropped_latents

    def get_data_and_condition(self, data_batch):
        self._normalize_video_databatch_inplace(data_batch)
        vision = data_batch["video"]
        latent = [self.encode(v).contiguous().float() for v in vision]
        latent = self._remove_padding_from_latent(latent, data_batch["image_size"])
        actions, domain_ids = self._normalize_action_databatch(data_batch)
        fps = torch.stack(data_batch["conditioning_fps"]).flatten().to(**self.tensor_kwargs)
        return GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            raw_state_vision=vision,
            x0_tokens_vision=latent,
            temporal_positions_vision=self._get_temporal_positions_vision(vision, latent),
            raw_state_action=actions,
            x0_tokens_action=actions,
            action_domain_id=domain_ids,
            raw_action_dim=data_batch["raw_action_dim"],
            fps_vision=fps,
            fps_action=fps,
        )

    def _normalize_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str | None = None
    ) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.
            input_key (str | None): The key for the video tensor in the data_batch. Defaults to
                `self.input_video_key` if not provided.

        Side Effects:
            Modifies the tensor at `input_key` within `data_batch` in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        IS_PREPROCESSED_KEY = "is_preprocessed"
        input_key = self.input_video_key if input_key is None else input_key
        if input_key in data_batch:
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                for i in range(len(data_batch[input_key])):
                    assert torch.is_floating_point(data_batch[input_key][i]), "Video data is not in float format."
                    assert torch.all((data_batch[input_key][i] >= -1.0001) & (data_batch[input_key][i] <= 1.0001)), (
                        f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key][i].min()}, {data_batch[input_key][i].max()}]"
                    )
            else:
                for i in range(len(data_batch[input_key])):
                    item = data_batch[input_key][i]
                    if isinstance(item, torch.Tensor):
                        item = [item]
                    assert item[0].dtype == torch.uint8, "Video data is not in uint8 format."
                    data_batch[input_key][i] = torch.stack(item).to(**self.tensor_kwargs_fp32) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _normalize_action_databatch(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Extract dense action and domain_id lists from the data batch.

        The joint dataloader produces action and domain_id data as
        ``[[tensor], [None], [tensor], ...]`` (each sample wrapped in a
        single-element list).  This method unwraps inner lists and filters
        out ``None`` entries to produce dense lists suitable for the model,
        **without mutating** ``data_batch``.

        Returns:
            (dense_action, dense_domain_id): Each is a list of device tensors
            containing only non-None entries, or ``None`` if all entries are
            ``None`` / the key is absent.
        """
        dense_action = unwrap_and_densify(data_batch.get("action", None), self.tensor_kwargs)
        dense_domain_id = unwrap_and_densify(
            data_batch.get("domain_id", None), {"device": self.tensor_kwargs["device"]}
        )
        return (dense_action, dense_domain_id)

    def denoise(
        self,
        net: torch.nn.Module | None = None,
        data_batch_packed: PackedSequence | None = None,
        memory: MemoryState | None = None,
    ) -> dict:
        """
        Runs the MoT network on a packed multi-modal sequence to predict velocity (v) targets.

        Args:
            data_batch_packed: PackedSequence from `pack_input_sequence(...)`.
            memory: Optional pre-built MemoryState for autoregressive generation
                or KV-cache training.

        Returns:
            dict containing:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_action": Velocity prediction for action modality (if action_gen enabled).
                - "preds_sound": Velocity prediction for sound modality (if sound_gen enabled).
                - "lbl_metadata_und": Load balancing metadata for understanding pathway (if present).
                - "lbl_metadata_gen": Load balancing metadata for generation pathway (if present).
        """
        net = net or self.net
        out_net = net(packed_seq=data_batch_packed, memory=memory)
        output_dict = dict()
        output_dict["preds_vision"] = out_net["preds_vision"]
        if True and "preds_action" in out_net:
            output_dict["preds_action"] = out_net["preds_action"]
        for key, value in out_net.items():
            if "lbl_metadata_" in key:
                output_dict[key] = value
        return output_dict

    def _tokenize_captions(
        self, captions: list[str], *, use_system_prompt: bool, system_prompt: str | None, is_video: bool
    ) -> list[list[int]]:
        """Per-caption chat-template tokenization (ragged, no padding).

        Single source of truth for the per-caption loop shared by
        :meth:`tokenize_text` (which adds padding + tensor-wrapping) and
        :meth:`_get_inference_text_tokens` (which feeds the diffusion
        sampling pipeline that natively consumes ``list[list[int]]``).

        Args:
            captions: Raw text prompts.
            use_system_prompt: Already-resolved boolean (callers handle any
                ``None`` fallback against ``self.vlm_config``).
            system_prompt: Explicit system prompt that overrides
                ``use_system_prompt`` / ``is_video`` when supplied.
            is_video: Selects the video vs image default system prompt
                when ``use_system_prompt=True`` and no explicit
                ``system_prompt`` is given.
        """
        return [
            tokenize_caption(
                c,
                self.vlm_tokenizer,
                is_video=is_video,
                use_system_prompt=use_system_prompt,
                system_prompt=system_prompt,
            )
            for c in captions
        ]

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer_vision_gen.decode(latent)

    @torch.no_grad()
    def generate(self, batch, seed=42, num_steps=50, guidance=6.0, guidance_action=1.0, shift=10.0, shift_action=1.0):
        """Joint video/motion flow matching with the released modality-forcing schedule."""
        plans, clean, cond, uncond, noise, reference, mask = self._prepare_inference_data(batch, [seed], True)
        vision_shape = clean.x0_tokens_vision[0].shape
        vision_size = clean.x0_tokens_vision[0].numel()
        action_shape = clean.x0_tokens_action[0].shape
        action_size = clean.x0_tokens_action[0].numel()
        weights = torch.full_like(noise[0], guidance)
        weights[vision_size : vision_size + action_size] = guidance_action

        def velocity(noise_x, timestep, timestep_action=None):
            kwargs = dict(
                noise_x=noise_x,
                timestep=timestep,
                timestep_action=timestep_action,
                sequence_plans=plans,
                gen_data_clean=clean,
            )
            positive = self._get_velocity(text_tokens=cond, **kwargs)
            if guidance == 1.0 and guidance_action == 1.0:
                return positive
            negative = self._get_velocity(text_tokens=uncond, **kwargs)
            return [negative[0] + weights * (positive[0] - negative[0])]

        latent = self.sampler(
            velocity,
            noise,
            num_steps=num_steps,
            shift=shift,
            seed=[seed],
            shift_action=shift_action,
            action_spans=[(vision_size, action_size)],
        )[0]
        video_latent = latent[:vision_size].reshape(vision_shape)
        motion = latent[vision_size : vision_size + action_size].reshape(action_shape)[:, :135]
        video = self.decode(video_latent).squeeze(0)
        return (video, motion)
