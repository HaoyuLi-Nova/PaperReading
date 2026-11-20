# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from torch import Tensor
from scripts.attention.backends import choose_backend
from scripts.attention.checks import (
    attention_param_checks,
    attention_tensor_checks,
    universal_tensor_checks,
    varlen_tensor_checks,
)
from scripts.attention.flash2 import flash2_attention
from scripts.attention.flash3 import flash3_attention
from scripts.attention.masks import CausalType
from scripts.attention.natten import natten_attention
from scripts.attention.utils.safe_ops import log

BACKEND_MAP = {"natten": natten_attention, "flash2": flash2_attention, "flash3": flash3_attention}


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    seqlens_Q: Tensor | None = None,
    seqlens_KV: Tensor | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    backend: str | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Attention on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Varlen Attention is only supported for the sequence-packed layout: QKV tensors have batch size
    1, and tokens from different batches are concatenated without any padding along the sequence
    dimension. Sequence lengths for different batches can be provided in two ways:
        1. `seqlens_Q` and `seqlens_KV` (less efficient): only provide the sequence lengths as
            integer tensors (must be on the same device as QKV), and cumulative and maximum sequence
            lengths are recomputed on each call.
        2. `cumulative_seqlen_{Q,KV}` and `max_seqlen_{Q,KV}` (more efficient):
            compute cumulative and maximum sequence lengths. `cumulative_seqlen_{Q,KV}` are integer
            tensors on the same device as QKV containing the cumulative sum of `seqlens_{Q,KV}`,
            with an additional `0` element in the beginning, therefore sized `batch+1`.
            `max_seqlen_{Q,KV}` are integers (not Tensors) that represent the maximum sequence
            lengths for Q and KV among all sequence batches.
            You can use `generate_varlen_parameters` to generate these
            parameters:
                ```python3
                from cosmos_framework.model.attention.varlen import generate_varlen_parameters
                (
                    cumulative_seqlen_Q,
                    cumulative_seqlen_KV,
                    max_seqlen_Q,
                    max_seqlen_KV,
                ) = generate_varlen_parameters(q, k, v, seqlens_Q, seqlens_KV)
                ```

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`, `CausalType.DontCare` (only valid when seqlen_q == seqlen_kv).
            Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        seqlens_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of query tokens in each batch. Must be passed together with
            `seqlens_KV`.

        seqlens_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of key/value tokens in each batch. Must be passed together with
            `seqlens_Q`.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        backend (str | None): Backend to run with. If unspecified (default), it will try to
            select the best available.

        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to the backend's
            attention operator, if any. Only valid when a specific backend is selected (backend is
            not None).

        deterministic (bool): Whether to enforce deterministic backward pass. Default is False.
            When True, backends are selected based on deterministic support.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen_q, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not guaranteed to be contiguous with some backends and it should
            not be made contiguous unless we can guarantee its results aren't merged via
            `merge_attentions`.
    """
    assert universal_tensor_checks(query=query, key=key, value=value, raise_error=True)
    assert attention_tensor_checks(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        raise_error=True,
    )
    attention_param_checks(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )
    cumulative_seqlen_Q, cumulative_seqlen_KV, max_seqlen_Q, max_seqlen_KV = varlen_tensor_checks(
        query=query,
        key=key,
        value=value,
        seqlens_Q=seqlens_Q,
        seqlens_KV=seqlens_KV,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
    )
    is_varlen = cumulative_seqlen_Q is not None
    scale = scale if scale is not None else query.shape[-1] ** (-0.5)
    if backend is None and backend_kwargs is not None:
        backend_kwargs = None
        log.debug("A backend was not specified, but got backend_kwargs. Ignoring... ")
    if backend is not None and backend not in BACKEND_MAP:
        raise ValueError(f"Selected backend={backend!r}, but available choices are {BACKEND_MAP.keys()}. ")
    compatible_backend = choose_backend(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        backend=backend,
        raise_error=False,
    )
    if compatible_backend is None and backend is None:
        raise ValueError(
            "Could not find a compatible Attention backend for this use case / device. Try running with debug logs to find out why."
        )
    elif compatible_backend is None:
        raise ValueError(
            f"Selected Attention backend {backend} is incompatible with this use case / device. Try running with debug logs to find out why."
        )
    assert compatible_backend in BACKEND_MAP
    return BACKEND_MAP[compatible_backend](
        query=query,
        key=key,
        value=value,
        is_causal=is_causal,
        causal_type=causal_type,
        scale=scale,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
        return_lse=return_lse,
        backend_kwargs=backend_kwargs,
        deterministic=deterministic,
    )
