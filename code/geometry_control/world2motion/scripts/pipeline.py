"""Load the released World2Motion MoT, tokenizer, and video VAE for inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from data.sequence_packing.modalities import add_special_tokens
from scripts.core import InferenceCore
from scripts.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig
from scripts.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from scripts.qwen.qwen3_vl import Qwen3VLTextRotaryEmbedding
from scripts.vae.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


def load_pipeline(checkpoint_dir: Path, vae_path: Path | None = None) -> InferenceCore:
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires an NVIDIA CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Inference requires a GPU with BF16 support.")
    checkpoint_dir = Path(checkpoint_dir)
    vae_path = Path(vae_path) if vae_path else checkpoint_dir / "Wan2.2_VAE.pth"
    weight_path = checkpoint_dir / "model.safetensors"
    for path in (weight_path, vae_path, checkpoint_dir / "qwen_config.json", checkpoint_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required model asset is missing: {path}")
    tokenizer_dir = checkpoint_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer files are missing: {tokenizer_dir}")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = namespace(json.loads((checkpoint_dir / "config.json").read_text()))
    qwen_config = Qwen3VLMoTConfig(json.loads((checkpoint_dir / "qwen_config.json").read_text()))
    qwen_config.freeze_und = False
    qwen_config.layer_module = "MoTDecoderLayer"
    qwen_config.qk_norm_for_text = True
    qwen_config.tie_word_embeddings = True

    logging.info("Building the World2Motion MoT architecture")
    with torch.device("meta"):
        language_model = Qwen3VLTextForCausalLM(qwen_config)
        net_config = Cosmos3VFMNetworkConfig(
            vlm_config=language_model.config,
            latent_patch_size=2,
            latent_downsample_factor=16,
            latent_channel_size=48,
            max_latent_h=40,
            max_latent_w=40,
            max_latent_t=300,
            enable_fps_modulation=True,
            base_fps=24,
            vision_gen=True,
            action_gen=True,
            sound_gen=False,
            joint_attn_implementation="two_way",
            timestep_scale=0.001,
            action_dim=160,
            num_embodiment_domains=32,
            temporal_compression_factor_vision=4,
            natten_parameter_list=None,
            video_temporal_causal=False,
        )
        net_config._attn_implementation_internal = "eager"
        net = Cosmos3VFMNetwork(language_model, net_config)
    net = net.to(dtype=torch.bfloat16)
    net.to_empty(device="cuda")
    # RoPE frequencies must stay FP32, including after materializing a meta model.
    for module in net.modules():
        if isinstance(module, Qwen3VLTextRotaryEmbedding):
            module.init_weights(buffer_device=torch.device("cuda"))
    expected = net.state_dict()
    logging.info("Loading the final EMA BF16 checkpoint")
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        required = {"model.net." + key for key in expected}
        if keys != required:
            raise ValueError(
                f"Checkpoint/architecture mismatch: missing={sorted(required - keys)}, unexpected={sorted(keys - required)}"
            )
        with torch.no_grad():
            for name, destination in expected.items():
                source = weights.get_tensor("model.net." + name)
                if source.shape != destination.shape or source.dtype != destination.dtype:
                    raise ValueError(f"Invalid tensor shape/dtype for {name}: {source.shape}, {source.dtype}")
                destination.copy_(source)
    del expected
    net.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer, tokens = add_special_tokens(tokenizer)
    tokens["eos_token_id"] = tokenizer.eos_token_id
    logging.info("Loading the Wan 2.2 video VAE")
    vae = Wan2pt2VAEInterface(vae_path=str(vae_path), encode_chunk_frames={"256": 68, "480": 24, "720": 12})
    return InferenceCore(net, vae, tokenizer, tokens, config)
