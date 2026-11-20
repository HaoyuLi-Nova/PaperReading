# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from data.sequence_packing.runtime import get_all_seq


def get_context_parallel_sharded_sequence(attn_implementation, input_pack, position_ids, parallel_dims):
    assert parallel_dims is None
    return (input_pack, position_ids)


def get_context_parallel_last_hidden_state(packed_outputs, parallel_dims):
    assert parallel_dims is None
    return get_all_seq(packed_outputs)
