# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
from scripts.attention import attention
from scripts.attention.masks import CausalType
from scripts.runtime.memory import KVToStore, MemoryValue


class SplitInfo:
    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which may
        be padded to max sequence length by upstream packers.
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )
        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)
        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)
        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens
        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens
        self.control_stream_token_ranges: list[tuple[int, int]] | None = None
        self.noisy_token_range: tuple[int, int] | None = None
        self.control_weights: list[float] | None = None


AttentionMaskType = SplitInfo
from data.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_all_seq,
    get_causal_seq,
    get_full_only_seq,
    sequence_pack_from_packed_sequence,
)


def two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    packed_key_states_normalized: SequencePack | None = None,
):
    """
    Performs two-way attention with causal and full attention.

    ``packed_key_states_normalized``: optional alternative K pack used for the generator's
    full attention (gen→all).  When provided, the generator attends to these keys
    instead of ``packed_key_states``, allowing the und K tokens to be normalised for
    the gen cross-attention path while keeping raw K tokens for the reasoner's own
    causal self-attention.  If ``None``, ``packed_key_states`` is used for both paths.
    """
    packed_key_normalized = (
        packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
    )
    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    sample_offsets = packed_query_states["sample_offsets"]
    use_dont_care_mask = causal_q_offsets is causal_k_offsets
    causal_res = attention(
        causal_q.unsqueeze(0),
        causal_k.unsqueeze(0),
        causal_v.unsqueeze(0),
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=packed_query_states["max_causal_len"],
        max_seqlen_KV=packed_query_states["max_causal_len"],
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )
    causal_out = causal_res.squeeze(0).flatten(-2, -1)
    full_res = attention(
        full_q.unsqueeze(0),
        get_all_seq(packed_key_normalized).unsqueeze(0),
        get_all_seq(packed_value_states).unsqueeze(0),
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=sample_offsets,
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_sample_len"],
    )
    full_out = full_res.squeeze(0).flatten(-2, -1)
    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def dispatch_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    assert memory_value is None
    assert isinstance(attention_mask, SplitInfo) and (not attention_mask.is_three_way)
    return (
        two_way_attention(packed_query_states, packed_key_states, packed_value_states, packed_key_states_normalized),
        None,
    )


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: list[tuple[int, int, int]] | None = None,
    natten_parameter_list: list | None = None,
    block_size: int = 128,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    skip_natten_metadata: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    pad_for_cuda_graphs: bool = False,
) -> tuple[SequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta).
    """
    device = packed_sequence.device
    natten_metadata_list = None
    attention_meta = SplitInfo(
        split_lens=split_lens, attn_modes=attn_modes, sample_lens=sample_lens, actual_len=int(packed_sequence.shape[0])
    )
    make_pack = sequence_pack_from_packed_sequence
    input_pack = make_pack(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
    )
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return (input_pack, attention_meta, natten_metadata_list)
