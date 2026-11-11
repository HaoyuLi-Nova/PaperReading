import os
import torch
from easydict import EasyDict


# ------------------------ environment ------------------------ #
cfg = EasyDict(__name__='Config: Wanx T2V RGB with Motion')

# environment variables
os.environ.update({
    'TOKENIZERS_PARALLELISM': 'false',
    'TORCH_CUDNN_V8_API_ENABLED': '1',
    'PYTHONHASHSEED': '2024', 
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})

# --------------------------- data --------------------------- #
H, W, T = 1280, 704, 121
t = (T - 1) // 4 + 1

cfg.update({'H': H, 'W': W, 'T': T, 't': t})

num_motion_latent = 51
cfg.update({
    'num_motion_latent_per_frame': num_motion_latent,
    'seq_len_video': t * H * W // 32 // 32,
    'seq_len_motion': T * num_motion_latent,
    'fps': 24,
    'seed': 42,
})


# ------------------------- model ------------------------- #
# T5
cfg.update({
    't5_model': 'umt5_xxl',
    't5_dtype': torch.bfloat16,
    'text_len': 512,
    't5_checkpoint': 'ckpt/models_t5_umt5-xxl-enc-bf16.pth',
    't5_tokenizer': 'ckpt/google/umt5-xxl/',
})

# VAE
cfg.update({
    'vae_checkpoint': 'ckpt/Wan2.2_VAE.pth',
    'downsample': (4, 32, 32),
})

# Transformer-5B
cfg.update({
    'patch_size': (1, 2, 2),
    'dim': 3072,
    'ffn_dim': 14336,
    'freq_dim': 256,
    'num_heads': 24,
    'num_layers': 30,
    'window_size': (-1, -1),
    'qk_norm': True,
    'cross_attn_norm': True,
    'eps': 1e-6,
    'checkpoint_path': 'ckpt/dit/echomotion_dit.pth', # TODO
})

cfg.update({
    'MoT_layers': list(range(0, cfg.num_layers - 2, 2)) + [cfg.num_layers - 1],
})


# ------------------------ motion ------------------------ #
cfg.update({
    'inject_motion_type': 'motion_patch_enc',
    'motion_statics': 'dataset/echomotion_local_data.npz',
    'motion_dim': 232,
    'use_task_embedding': True,
    'fetch_local_motion': True,
    'use_shape': True,
})

# ------------------------ diffusion ------------------------ #
cfg.update({
    'schedule': 'cosine',
    'prediction_type': 'fm_scheduler',
    'solver': 'dpmpp_2m_sde',
    'neg_prompt': '右上角有一个小的白色圆形标志，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，色调艳丽',
    'sample_shift': 6.0,
    'sample_shift_motion': 6.0,
    'num_train_timesteps': 1000,
    'sample_steps': 50,
})

# ----------------------- guidance ----------------------- #
cfg.update({
    'sample_guide_scale': 5.0,
    'motion_sample_guide_scale': 3.0,
    'video_guide_motion_scale': 2.0,
    'motion_guide_video_scale': 3.0,
    'use_prompt_extend': False,
})

# ----------------------- inference ----------------------- #
cfg.update({
    'param_dtype': torch.bfloat16,
    'sample_fps': 24,
    'out_dir': 'outputs/',
    'max_seq_len': cfg.t * W * H // 32 // 32,
    'target_shape': (48, cfg.t, H // 16, W // 16),
})

