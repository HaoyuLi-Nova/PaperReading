# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Optional, cast
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_SYSTEM_PROMPT_IMAGE = "You are a helpful assistant who will generate images from a give prompt."
_SYSTEM_PROMPT_VIDEO = "You are a helpful assistant who will generate videos from a give prompt."


def tokenize_caption(
    caption: str,
    tokenizer: PreTrainedTokenizerBase,
    is_video: bool = False,
    use_system_prompt: bool = False,
    system_prompt: Optional[str] = None,
) -> list[int]:
    """Tokenize a text caption into token IDs using the Qwen2 chat template.

    Wraps the caption in a chat-style conversation (with a "user" role) and applies
    the tokenizer's chat template to produce the final token ID sequence, including
    any special tokens (e.g., BOS, role markers, generation prompt).

    Args:
        caption: The text caption to tokenize.
        tokenizer: A HuggingFace ``PreTrainedTokenizerBase`` (e.g. Qwen2Tokenizer or Fast tokenizer).
        is_video: If True (and use_system_prompt=True), uses the video system prompt;
            otherwise uses the image system prompt. Ignored when ``system_prompt`` is
            provided.
        use_system_prompt: If True, prepends a system prompt message to the conversation
            before the user caption. Ignored when ``system_prompt`` is provided.
        system_prompt: When supplied, this exact string is used as the system prompt,
            overriding both ``is_video`` and ``use_system_prompt``.

    Returns:
        List of token IDs representing the full chat-formatted caption.
    """
    conversations = []
    if system_prompt is not None:
        conversations.append({"role": "system", "content": system_prompt})
    elif use_system_prompt:
        _system_prompt = _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE
        conversations.append({"role": "system", "content": _system_prompt})
    conversations.append({"role": "user", "content": caption})
    tokenizer_output = tokenizer.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, add_vision_id=False, return_dict=False
    )
    return cast(list[int], tokenizer_output)
