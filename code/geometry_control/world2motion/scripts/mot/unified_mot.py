# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Any
import json
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from torch.distributed import ProcessGroup
from scripts.mot.attention import AttentionMaskType, dispatch_attention
from scripts.qwen.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from scripts.qwen.qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)
from scripts.qwen.qwen3_vl import apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb
from scripts.runtime.memory import KVToStore, MemoryState, MemoryValue
from data.sequence_packing.runtime import (
    SequencePack,
    from_all_seq,
    from_und_gen_splits,
    get_device_and_dtype,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)


class LayerTypes:
    """Architecture-family dispatch table for the shared MoT layers.

    A single ``LayerTypes(variant)`` instance bundles the four sub-layer
    classes / functions that the MoT decoder layers swap in based on the
    text-model family: the MLP block, the RMSNorm class, the rotary
    embedding module, and the ``apply_rotary_pos_emb`` function.  Passed
    through ``_impl_init`` to ``MoTDecoderLayer`` so a single decoder
    implementation works across all three families.

    Supported variants: ``"qwen3_vl_dense"``, ``"qwen3_vl_moe"``,
    ``"nemotron_dense"``.
    """

    @property
    def is_moe(self) -> bool:
        return self.variant == "qwen3_vl_moe"

    def __init__(self, variant):
        assert variant == "qwen3_vl_dense"
        self.variant = variant
        self.mlp = Qwen3VLTextMLP
        self.rms_norm = Qwen3VLTextRMSNorm
        self.rotary_embedding = Qwen3VLTextRotaryEmbedding
        self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb


class _MoTConfigBase(object):
    """Shared MoT wrapper logic for all three architecture families.

    Concrete subclasses: :class:`Qwen3VLMoTConfig` (dense Qwen3-VL),
    :class:`Qwen3VLMoeMoTConfig` (Qwen3-VL MoE), and
    :class:`Nemotron3DenseVLMoTConfig`.

    Stores the **JSON config dict** (as loaded from the per-family
    config JSON) plus a small fixed set of MoT-specific fields, and
    materializes fresh HF configs on demand via :pyattr:`full_config`,
    :pyattr:`text_config`, and :pyattr:`vision_config`.  Three concerns
    are kept separate:

    1. The HF *text* config that ``*TextModel`` / decoder layers
       actually consume — built from the nested ``text_config``
       sub-section of the JSON (or the flat dict for LLM-only configs),
       optionally massaged by :meth:`_transform_text_dict`, then splatted
       into :pyattr:`_text_config_cls`.  MoT-specific fields are *not*
       merged in here; see point 3.
    2. The HF *vision* sub-section (``config_dict["vision_config"]``)
       that the visual tower needs but the text model must never see —
       surfaced via :pyattr:`vision_config`.
    3. MoT-specific knobs (``qk_norm_for_text`` / ``qk_norm_for_diffusion``,
       ``include_visual``) that are read directly off the wrapper:
       ``qk_norm_for_*`` are passed as constructor kwargs to the
       ``*TextModel`` (and from there to the decoder layers / packed
       attention), and ``include_visual`` gates whether the ViT is built
       at all by the ``*TextForCausalLM`` __init__.

    Constructor surface:

    - ``config_dict``: full JSON dict (nested-VLM with top-level
      ``text_config`` / ``vision_config`` / ``image_token_id``, or flat
      LLM-only).  Stored verbatim under ``self.config_dict`` and never
      mutated; the materialization properties build fresh dicts.
    - MoT kwargs (``qk_norm_for_text``, ``qk_norm_for_diffusion``,
      ``include_visual``): stored as plain attributes.  These do *not*
      flow into :pyattr:`text_config` — the ``*TextForCausalLM.__init__``
      flow reads them directly off the wrapper.
    - ``text_config_overrides``: optional ``{field: value}`` mapping
      that the :pyattr:`text_config` property merges into the text
      dict *after* :meth:`_transform_text_dict` (so SMOKE-style
      shrinks like ``num_hidden_layers=2`` win over both the JSON
      defaults and the Nemotron ``56 -> 28`` fold).  Stored as a
      plain attribute so the ``create_vlm_config`` ``setattr`` flow
      can drop a fresh mapping in post-construction; the next
      ``text_config`` access picks it up.

    Post-construction overrides via plain ``setattr`` (the
    ``create_vlm_config`` flow in ``configs/base/defaults/reasoner.py``)
    just update the same plain attributes, so the next property access
    picks up the latest values.  No cache, no ``__setattr__``
    interception, no override bucket — the property rebuild is cheap
    relative to model init and consumers (``*TextForCausalLM.__init__``)
    read each property once and store the result locally.

    Subclasses customize four things:

    - :pyattr:`_full_config_cls`: HF full-config class (e.g.
      ``Qwen3VLConfig``); used by :pyattr:`full_config` to build the
      base ``PreTrainedModel.config`` for the ``*ForCausalLM`` wrapper.
    - :pyattr:`_text_config_cls`: HF text-config class (e.g.
      ``Qwen3VLTextConfig``); used by :pyattr:`text_config`.
    - :pyattr:`_vision_config_cls`: HF vision-config class for the
      visual tower (only relevant when ``include_visual=True``).
    - :meth:`_transform_text_dict`: optional hook for architecture-
      specific massaging of the text-config dict (e.g. the Nemotron 3
      Dense VL ``56 -> 28`` layer fold).  Default is identity.
    """

    _full_config_cls: type = type(None)
    _text_config_cls: type = type(None)
    _vision_config_cls: type = type(None)

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        include_visual: bool = False,
        gen_noisy_gating: bool = False,
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        self.config_dict = dict(config_dict)
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.include_visual = include_visual
        self.gen_noisy_gating = gen_noisy_gating
        self.text_config_overrides: dict[str, Any] = dict(text_config_overrides) if text_config_overrides else {}

    @property
    def full_config(self) -> Any:
        """Materialize the full HF LLM/VLM config from the stored dict.

        Splats :pyattr:`config_dict` directly into :pyattr:`_full_config_cls`.
        Used by ``*TextForCausalLM.__init__`` to build the
        ``PreTrainedModel.config`` that the HF base class stores under
        ``self.config``.
        """
        if self._full_config_cls is type(None):
            raise ValueError(f"No _full_config_cls defined for {self.__class__.__name__}")
        return self._full_config_cls(**self.config_dict)

    @property
    def text_config(self) -> Any:
        """Materialize a fresh HF text config from the stored dict.

        Built from scratch on every access — cheap relative to model
        init and naturally correct under the ``create_vlm_config``
        post-init ``setattr`` flow.  Construction steps:

        1. Pick the source dict: the ``text_config`` sub-section of
           :pyattr:`config_dict` when nested, or the dict itself for
           flat / LLM-only configs.
        2. Run :meth:`_transform_text_dict` on it (default identity;
           the Nemotron variant uses it for the ``56 -> 28`` layer
           fold).
        3. Merge :pyattr:`text_config_overrides` on top (last write
           wins) so caller-supplied per-field overrides (SMOKE
           shrinks like ``num_hidden_layers=2``) beat both the JSON
           defaults and the Nemotron transform.  Applied as a
           post-transform overlay rather than a pre-transform merge
           so transform-aware overrides (final layer counts) don't
           get re-halved by the Nemotron fold.
        4. Splat into :pyattr:`_text_config_cls`.

        MoT-specific fields (``qk_norm_for_*``, ``include_visual``) are
        deliberately *not* injected into the returned HF text config —
        they're read directly off the wrapper by
        ``*TextForCausalLM.__init__`` and forwarded as ``*TextModel``
        constructor kwargs.  Likewise ``tie_word_embeddings`` is only
        consumed via :pyattr:`full_config` (it lives at the JSON
        top-level for nested VLM configs and is not duplicated into the
        text sub-section).
        """
        nested = self.config_dict.get("text_config")
        text_dict = nested if isinstance(nested, dict) else self.config_dict
        text_dict = self._transform_text_dict(text_dict)
        overrides = getattr(self, "text_config_overrides", None) or {}
        if overrides:
            text_dict = {**text_dict, **overrides}
        if self._text_config_cls is type(None):
            raise ValueError(f"No _text_config_cls defined for {self.__class__.__name__}")
        return self._text_config_cls(**text_dict)

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Subclass hook: transform the text-config dict before MoT fields are merged.

        Default implementation is identity.  Override in subclasses to
        apply architecture-specific massaging (e.g. the Nemotron 3
        Dense VL ``56 -> 28`` layer fold).  The hook receives the dict
        already split out of the nested ``text_config`` sub-section
        (or the flat dict for LLM-only configs) and must return a
        dict-like object suitable for splatting into
        :pyattr:`_text_config_cls`.
        """
        return text_dict

    @property
    def vision_config(self) -> Any | None:
        """Materialize the HF vision config for the visual tower, or None.

        Mirrors :pyattr:`text_config`'s "build on demand" pattern:
        gates on :pyattr:`include_visual`, splats the JSON's
        ``vision_config`` sub-section into :pyattr:`_vision_config_cls`,
        and returns the built HF config (caller plugs it into
        ``Qwen3VL{,Moe}VisionModel._from_config``).

        Returns ``None`` when ``include_visual=False``.  When
        ``include_visual=True`` but the JSON has no ``vision_config``
        sub-section (e.g. an LLM-only checkpoint mistakenly used as a
        VLM), raises ``ValueError`` eagerly rather than silently
        building an empty ViT.
        """
        if not self.include_visual:
            return None
        vision_dict = self.config_dict.get("vision_config")
        if vision_dict is None:
            raise ValueError(
                "include_visual=True requires a vision_config sub-section in the language-model JSON config."
            )
        if self._vision_config_cls is type(None):
            raise ValueError(f"No _vision_config_cls defined for {self.__class__.__name__}")
        return self._vision_config_cls(**vision_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "_MoTConfigBase":
        """Load the JSON file verbatim and wrap it.

        The full dict is stored under ``self.config_dict``; sibling
        ``vision_config`` / ``image_token_id`` / ``vision_*_token_id``
        fields (when present) are surfaced lazily via
        :pyattr:`vision_config` and by HF downstream consumers reading
        the dict directly.
        """
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls(config_dict=config_dict)


class Qwen3VLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the dense Qwen3-VL family."""

    _full_config_cls = Qwen3VLConfig
    _text_config_cls = Qwen3VLTextConfig
    _vision_config_cls = Qwen3VLVisionConfig


def _apply_head_sharded_o_proj(
    local_attn_output: torch.Tensor, projection: nn.Linear, feature_slice: slice, cp_group: ProcessGroup
) -> torch.Tensor:
    """Apply one local input-column slice of ``projection`` and sum partial outputs."""
    local_weight = projection.weight[:, feature_slice]
    out = torch.nn.functional.linear(local_attn_output, local_weight, bias=None)
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=cp_group)
    if projection.bias is not None:
        out = out + projection.bias
    return out


class PackedAttentionMoT(nn.Module):
    """
    Dual-pathway packed attention for MoT architectures.
    Implements understanding and generation pathways with separate projections.

    Used for Qwen3VL (Dense), Qwen3VL-MoE, and Nemotron 3 Dense VL variants.
    QK normalisation and RoPE function are selected via ``layer_types`` and the
    ``qk_norm_for_text`` config attribute (the generation pathway always installs
    QK norm).
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Any | Any,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        use_und_k_norm_for_gen: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        eps = config.rms_norm_eps
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        if qk_norm_for_text:
            self.q_norm = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        if qk_norm_for_diffusion:
            self.q_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()
        if use_und_k_norm_for_gen and qk_norm_for_diffusion and (not qk_norm_for_text):
            self.k_norm_und_for_gen: nn.Module | None = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.k_norm_und_for_gen = None
        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self._apply_rotary_pos_emb = layer_types.apply_rotary_pos_emb
        self.dispatch_attention_fn = dispatch_attention
        self.replicated_attention_io_local_head_o_proj = False
        self.replicated_attention_io_cp_mesh: Any | None = None

    def _replicated_attention_io_q_feature_slice(self) -> slice:
        cp_mesh = self.replicated_attention_io_cp_mesh
        assert cp_mesh is not None, "replicated attention I/O requires a CP mesh"
        cp_group = cp_mesh.get_group()
        cp_rank = torch.distributed.get_rank(cp_group)
        cp_size = torch.distributed.get_world_size(cp_group)
        assert self.num_key_value_heads % cp_size == 0, (
            f"cp_size({cp_size}) must divide num_key_value_heads({self.num_key_value_heads})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads({self.num_attention_heads}) must be divisible by num_key_value_heads({self.num_key_value_heads})"
        )
        kv_heads_per_rank = self.num_key_value_heads // cp_size
        q_heads_per_kv_head = self.num_attention_heads // self.num_key_value_heads
        q_heads_per_rank = kv_heads_per_rank * q_heads_per_kv_head
        q_start = cp_rank * q_heads_per_rank
        q_end = q_start + q_heads_per_rank
        return slice(q_start * self.head_dim, q_end * self.head_dim)

    def _uses_replicated_attention_io_local_head_o_proj(self, memory_value: MemoryValue | None) -> bool:
        return (
            self.replicated_attention_io_local_head_o_proj
            and memory_value is not None
            and (getattr(memory_value, "frame_idx", 0) > 0)
            and (not getattr(memory_value, "for_cuda_graphs", False))
        )

    def forward(
        self,
        pack: SequencePack,
        attention_mask: AttentionMaskType,
        packed_position_embeddings: tuple[SequencePack, SequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
    ) -> tuple[SequencePack, KVToStore | None]:
        """Forward pass with optional memory-augmented attention.

        When ``memory_value`` is provided, ``dispatch_attention_fn`` routes to
        the appropriate attention kernel (e.g. three-way KV-cache attention
        for training, or AR inference concat + dense attention).

        ``kv_to_store`` is produced when ``memory_value`` is present:
        ``(gen_k, gen_v, und_k, und_v)`` for the caller to write back via
        ``MemoryState.write_for_layer()``.  The tensors are passed with
        gradients attached; each ``MemoryState`` decides whether to detach
        (e.g. for truncated BPTT) or keep gradients (e.g. teacher forcing).

        Args:
            pack: Packed sequence with und/gen tokens
            attention_mask: Attention metadata (SplitInfo).
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Optional read-only tensor container for memory-augmented attention.
        """
        q_und_in = self.q_proj(get_und_seq(pack))
        q_gen_in = self.q_proj_moe_gen(get_gen_seq(pack))
        k_und_in = self.k_proj(get_und_seq(pack))
        k_gen_in = self.k_proj_moe_gen(get_gen_seq(pack))
        v_und_in = self.v_proj(get_und_seq(pack))
        v_gen_in = self.v_proj_moe_gen(get_gen_seq(pack))
        q_und = q_und_in.view(-1, self.num_attention_heads, self.head_dim)
        k_und = k_und_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_und = v_und_in.view(-1, self.num_key_value_heads, self.head_dim)
        q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)
        k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)
        v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)
        q_und = self.q_norm(q_und)
        k_und = self.k_norm(k_und)
        q_gen = self.q_norm_moe_gen(q_gen)
        k_gen = self.k_norm_moe_gen(k_gen)
        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]
        q_und_, k_und_ = self._apply_rotary_pos_emb(
            q_und, k_und, get_und_seq(packed_cos), get_und_seq(packed_sin), unsqueeze_dim=1
        )
        q_gen_, k_gen_ = self._apply_rotary_pos_emb(
            q_gen, k_gen, get_gen_seq(packed_cos), get_gen_seq(packed_sin), unsqueeze_dim=1
        )
        packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)
        packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)
        packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)
        if self.k_norm_und_for_gen is not None:
            k_und_normalized = self.k_norm_und_for_gen(k_und)
            _, k_und_for_gen_ = self._apply_rotary_pos_emb(
                q_und_, k_und_normalized, get_und_seq(packed_cos), get_und_seq(packed_sin), unsqueeze_dim=1
            )
            packed_key_states_normalized_: SequencePack | None = from_und_gen_splits(k_und_for_gen_, k_gen_, pack)
        else:
            packed_key_states_normalized_ = None
        packed_attn_output, kv_to_store = self.dispatch_attention_fn(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=packed_key_states_normalized_,
        )
        if memory_value is not None and kv_to_store is None:
            und_len = pack["_num_causal_tokens"]
            gen_len = pack["_num_full_tokens"]
            k_und_to_store = k_und_for_gen_ if self.k_norm_und_for_gen is not None else k_und_
            kv_to_store = (
                k_gen_[:gen_len].unsqueeze(0),
                v_gen[:gen_len].unsqueeze(0),
                k_und_to_store[:und_len].unsqueeze(0),
                v_und[:und_len].unsqueeze(0),
            )
        if self._uses_replicated_attention_io_local_head_o_proj(memory_value):
            local_und_attn = get_und_seq(packed_attn_output)
            local_gen_attn = get_gen_seq(packed_attn_output)
            assert local_und_attn.shape[0] == 0, "replicated attention I/O only supports gen-only frame 1+ attention"
            feature_slice = self._replicated_attention_io_q_feature_slice()
            assert feature_slice.start is not None and feature_slice.stop is not None
            expected_local_features = feature_slice.stop - feature_slice.start
            assert local_gen_attn.shape[-1] == expected_local_features, (
                f"Expected local attention features {expected_local_features}, got {local_gen_attn.shape[-1]}"
            )
            cp_mesh = self.replicated_attention_io_cp_mesh
            assert cp_mesh is not None, "replicated attention I/O requires a CP mesh"
            cp_group = cp_mesh.get_group()
            und_seq = local_gen_attn.new_empty((0, self.hidden_size))
            gen_seq = _apply_head_sharded_o_proj(local_gen_attn, self.o_proj_moe_gen, feature_slice, cp_group)
        else:
            und_seq = self.o_proj(get_und_seq(packed_attn_output))
            gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))
        return (from_und_gen_splits(und_seq, gen_seq, pack), kv_to_store)


def _impl_init(
    self,
    config: Qwen3VLTextConfig | Any | Any,
    *,
    layer_types: LayerTypes,
    qk_norm_for_text: bool,
    qk_norm_for_diffusion: bool,
    gen_noisy_gating: bool = False,
    use_und_k_norm_for_gen: bool = False,
):
    """Shared ``__init__`` body for the three MoT text-model variants.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.  Sub-layer classes (MLP, RMSNorm,
    rotary embedding) are dispatched through ``layer_types``.
    """
    self.padding_idx = getattr(config, "pad_token_id", None)
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList()
    for layer_idx in range(config.num_hidden_layers):
        self.layers.append(
            MoTDecoderLayer(
                config=config,
                layer_types=layer_types,
                layer_idx=layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
                gen_noisy_gating=gen_noisy_gating,
                use_und_k_norm_for_gen=use_und_k_norm_for_gen,
            )
        )
    self.norm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    self.norm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = layer_types.rotary_embedding(config)
    self.post_init()


def _impl_forward(
    self,
    pack: SequencePack,
    attention_mask,
    position_ids: torch.Tensor,
    natten_metadata_list: list | None = None,
    memory: MemoryState | None = None,
) -> tuple[SequencePack, dict[str, Any]]:
    """Shared training forward pass for the three MoT text models.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.

    Args:
        pack: Packed sequence with und/gen tokens.
        attention_mask: Attention metadata (SplitInfo).
        position_ids: Position IDs (1D ``[N]`` for standard RoPE or 2D
            ``[3, N]`` for mrope).
        natten_metadata_list: Optional per-layer NATTEN metadata.
        memory: Optional ``MemoryState`` for persistent memory across
            forward passes.
    """
    device, dtype = get_device_and_dtype(pack)
    _meta_tensor = torch.tensor([], dtype=dtype, device=device)
    cos, sin = self.rotary_emb(
        _meta_tensor, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)
    position_embeddings = (from_all_seq(cos, pack), from_all_seq(sin, pack))
    lbl_metadata_all = dict(und=[], gen=[])
    hidden_states = pack
    if memory is not None:
        memory.init(hidden_states, device)
    memory_gen_only = memory.is_gen_only() if memory is not None else False
    for i, decoder_layer in enumerate(self.layers):
        memory_value = memory.read_for_layer(i) if memory is not None else None
        hidden_states, lbl_metadata_dict, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            position_embeddings,
            natten_metadata=None if natten_metadata_list is None else natten_metadata_list[i],
            memory_value=memory_value,
            gen_only=memory_gen_only,
        )
        if kv_to_store is not None and memory is not None:
            memory.write_for_layer(i, kv_to_store)
        for pathway, lbl_metadata in lbl_metadata_dict.items():
            lbl_metadata_all[pathway].append(lbl_metadata)
    final_lbl_metadata: dict[str, Any] = dict()
    hidden_states_out = zeros_like(hidden_states)
    set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))
    set_gen_seq(hidden_states_out, self.norm_moe_gen(get_gen_seq(hidden_states)))
    return (hidden_states_out, final_lbl_metadata)


def _run_mlp(mlp: torch.nn.Module, input: torch.Tensor) -> tuple[torch.Tensor, Any | None]:
    return (mlp(input), None)


class MoTDecoderLayer(nn.Module):
    """
    Unified MoT (Mixture of Transformers) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Any | Any,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        gen_noisy_gating: bool = False,
        use_und_k_norm_for_gen: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionMoT(
            config,
            layer_types=layer_types,
            layer_idx=layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )
        self.mlp = layer_types.mlp(config)
        self.mlp_moe_gen = layer_types.mlp(config)
        self.input_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input: SequencePack,
        attention_mask,
        packed_position_embeddings: tuple[SequencePack, SequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        gen_only: bool = False,
    ) -> tuple[SequencePack, dict[str, Any], KVToStore | None]:
        """Forward pass with MoT routing and optional memory-augmented attention.

        Returns a 3-tuple: ``(hidden_states, lbl_metadata_dict, kv_to_store)``.
        ``kv_to_store`` is non-None when ``memory_value`` is provided,
        containing ``(gen_k, gen_v, und_k, und_v)`` to be written back by
        ``MemoryState.write_for_layer()`` outside the ``torch.compile``
        boundary.

        Args:
            input: Packed sequence with und/gen tokens
            attention_mask: Attention mask
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Read-only tensor container from MemoryState.read_for_layer().
            gen_only: When True, skip the understanding pathway (und K/V come from cache).
        """
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(input)), self.input_layernorm_moe_gen(get_gen_seq(input)), input
        )
        kv_to_store: KVToStore | None = None
        if gen_only:
            assert natten_metadata is None
            _gen_norm = get_gen_seq(pack_norm_out)
            gen_pack = from_und_gen_splits(_gen_norm.new_empty(0, _gen_norm.shape[-1]), _gen_norm, pack_norm_out)
            _cos, _sin = packed_position_embeddings
            _empty_cos_und = get_und_seq(_cos)[:0]
            _empty_sin_und = get_und_seq(_sin)[:0]
            gen_position_embeddings = (
                from_und_gen_splits(_empty_cos_und, get_gen_seq(_cos), _cos),
                from_und_gen_splits(_empty_sin_und, get_gen_seq(_sin), _sin),
            )
            pack_attn_out, kv_to_store = self.self_attn(
                gen_pack,
                attention_mask,
                gen_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            gen_attn_out = get_gen_seq(pack_attn_out)
            residual_gen = get_gen_seq(input) + gen_attn_out
        else:
            pack_attn_out, kv_to_store = self.self_attn(
                pack_norm_out,
                attention_mask,
                packed_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            residual_und = get_und_seq(input) + get_und_seq(pack_attn_out)
            residual_gen = get_gen_seq(input) + get_gen_seq(pack_attn_out)
        lbl_metadata_dict: dict[str, Any] = dict()
        if gen_only:
            ln_out_und = residual_gen.new_empty(0, residual_gen.shape[-1])
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)
            gen_len = pack_attn_out["_num_full_tokens"]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen
            mlp_out_und_seq = residual_gen.new_empty(0, residual_gen.shape[-1])
            mlp_out_gen_seq = residual_gen + mlp_out_gen
        else:
            ln_out_und = self.post_attention_layernorm(residual_und)
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)
            gen_len = pack_attn_out["_num_full_tokens"]
            und_len = pack_attn_out["_num_causal_tokens"]
            ln_out_und_unpadded = ln_out_und[:und_len]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]
            mlp_out_und_unpadded, lbl_metadata_und = _run_mlp(self.mlp, ln_out_und_unpadded)
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            mlp_out_und = torch.cat([mlp_out_und_unpadded, ln_out_und[und_len:]], dim=0)
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)
            if lbl_metadata_und is not None:
                lbl_metadata_dict["und"] = lbl_metadata_und
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen
            mlp_out_und_seq = residual_und + mlp_out_und
            mlp_out_gen_seq = residual_gen + mlp_out_gen
        return (from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict, kv_to_store)


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text model for MoT with dense MLPs.
    This is a wrapper around the _impl_forward defined above,
    specialized for dense models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        use_und_k_norm_for_gen: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)


class Qwen3VLTextForCausalLM(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text causal language model for MoT.
    This variant is used for dense-only MLP models.
    """

    _tied_weights_keys: list[str] = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoTConfig):
        super().__init__(config.full_config)
        text_config = config.text_config
        self.model = Qwen3VLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
            use_und_k_norm_for_gen=getattr(config, "use_und_k_norm_for_gen", False),
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        vision_config = config.vision_config
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: SequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[SequencePack, dict[str, Any]]:
        """Training forward pass — delegates to the dense text model."""
        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        return outputs
