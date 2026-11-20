# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

import torch
from torch import Tensor
from scripts.attention.utils import is_torch_compiling


def generate_varlen_parameters(
    query: Tensor, key: Tensor, value: Tensor, seqlens_Q: Tensor | None = None, seqlens_KV: Tensor | None = None
) -> tuple[None, None, int, int] | tuple[Tensor, Tensor, int, int]:
    if is_torch_compiling():
        raise RuntimeError(
            "Running 'generate_varlen_parameters' in a torch-compiled region is disallowed as it results in graph breaks. Please consider calling ahead of time and pass 'cumulative_seqlen_{Q,KV}' and 'max_seqlen_{Q,KV}' instead of 'seqlens_{Q,KV}' to 'attention'. "
        )
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            f"Q, K, and V must match in batch size, got query.shape[0]={query.shape[0]!r}, key.shape[0]={key.shape[0]!r}, value.shape[0]={value.shape[0]!r}."
        )
    if (seqlens_Q is None) ^ (seqlens_KV is None):
        raise ValueError(
            f"Variable length Attention requires both of seqlens_Q and seqlens_KV to be set, got seqlens_Q={seqlens_Q!r}, seqlens_KV={seqlens_KV!r}."
        )
    if seqlens_Q is None and seqlens_KV is None:
        return (None, None, 0, 0)
    assert seqlens_Q is not None
    assert seqlens_KV is not None
    if not isinstance(seqlens_Q, Tensor) or not isinstance(seqlens_KV, Tensor):
        raise ValueError("seqlens_Q and seqlens_KV must both be tensors.")
    if seqlens_Q.device != query.device or seqlens_KV.device != query.device:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must be on the same device as QKV, but seqlens_Q.device={seqlens_Q.device!r}, seqlens_KV.device={seqlens_KV.device!r}, query.device={query.device!r}."
        )
    if seqlens_Q.dtype != torch.int32 or seqlens_KV.dtype != torch.int32:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must both be torch.int32 tensors, got seqlens_Q.dtype={seqlens_Q.dtype!r}, seqlens_KV.dtype={seqlens_KV.dtype!r}."
        )
    if seqlens_Q.dim() != 1 or seqlens_KV.dim() != 1:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must both be 1-D tensors, got seqlens_Q.dim()={seqlens_Q.dim()!r}, seqlens_KV.dim()={seqlens_KV.dim()!r}."
        )
    if seqlens_Q.shape[0] != seqlens_KV.shape[0]:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must match in size, got seqlens_Q.shape={seqlens_Q.shape!r}, seqlens_KV.shape={seqlens_KV.shape!r}."
        )
    if seqlens_Q.shape[0] < 1:
        raise ValueError(
            f"seqlens_Q and seqlens_KV must contain at least one element, got seqlens_Q.shape={seqlens_Q.shape!r}, seqlens_KV.shape={seqlens_KV.shape!r}."
        )
    if query.shape[0] != 1:
        raise ValueError(
            f"Variable length attention only supports sequence-packed memory layout (batch = 1), got query.shape[0]={query.shape[0]!r}."
        )
    assert seqlens_Q.dim() == seqlens_KV.dim() == 1
    assert seqlens_Q.shape[0] == seqlens_KV.shape[0] >= 1
    assert seqlens_Q.dtype == seqlens_KV.dtype == torch.int32
    max_seqlen_Q = seqlens_Q.max().item()
    max_seqlen_KV = seqlens_KV.max().item()
    if max_seqlen_Q < 0 or max_seqlen_KV < 0:
        raise ValueError(
            f"max_seqlen_Q and max_seqlen_KV cannot be negative, got max_seqlen_Q={max_seqlen_Q!r}, max_seqlen_KV={max_seqlen_KV!r}."
        )
    if (max_seqlen_Q == 0) != (max_seqlen_KV == 0):
        raise ValueError(
            f"max_seqlen_Q and max_seqlen_KV must either both be 0 or both be positive, but computed max_seqlen_Q={max_seqlen_Q!r}, max_seqlen_KV={max_seqlen_KV!r} from provided seqlens."
        )
    z = torch.tensor([0], dtype=torch.int32, device=seqlens_Q.device)
    cumulative_seqlen_Q = torch.cat([z, seqlens_Q.cumsum(0).to(torch.int32)], dim=0)
    cumulative_seqlen_KV = torch.cat([z, seqlens_KV.cumsum(0).to(torch.int32)], dim=0)
    assert isinstance(max_seqlen_Q, int)
    assert isinstance(max_seqlen_KV, int)
    return (cumulative_seqlen_Q, cumulative_seqlen_KV, max_seqlen_Q, max_seqlen_KV)
