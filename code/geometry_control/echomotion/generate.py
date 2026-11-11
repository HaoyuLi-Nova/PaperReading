import argparse
import logging
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.cuda.amp as amp
from einops import rearrange
from tqdm import tqdm

from dataset.utils import (
    load_motion,
    recover_from_velocity,
    denormalize_motion,
    normalize_motion,
)

import models
from models import Wan2_2_VAE, EchoMotion_DiT
from utils import HuggingfaceTokenizer
from utils.fm_solvers import FlowDPMSolverMultistepScheduler    
from utils.prompt_extend import call_with_messages
from utils.utils import TensorList, cache_video, randn_like, to_
from vis_motion import motion_vis_during_validation
from utils.sample_utils import load_video, get_video_dimensions

from configs.echomotion_wan5b_config import cfg

class T5Encoder:

    def __init__(
        self,
        name,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
    ):
        self.name = name
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = getattr(models, name)(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device
        ).eval().requires_grad_(False)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean='whitespace'
        )
    
    def __call__(self, texts):
        ids, mask = to_(self.tokenizer(
            texts,
            return_mask=True,
            add_special_tokens=True
        ), self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps+1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))
    return sigma

def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device= None,
    timesteps= None,
    sigmas = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _parse_args():
    parser = argparse.ArgumentParser(description='Training script with config selection')
    parser.add_argument(
        '--sample_mode',
        default='joint',
        help='joint or guide each orther, choose from [joint, motion_to_video, video_to_motion]'
    )
    parser.add_argument(
        '--prompt',
        default='A man is running.',
        help='text input for in-context guidance'
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--input_motion',
        default=None,
        help='only for motion to video'
    )
    parser.add_argument(
        '--input_video',
        default=None,
        help='only for video to motion'
    )
    parser.add_argument(
        '--use_prompt_extend',
        default=False,
        action='store_true',
        help='whether to use prompt extension'
    )
    args, _ = parser.parse_known_args()
    return args

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)
        
if __name__ == "__main__":

    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _init_logging(0)

    # load motion mean and std
    if cfg.motion_statics:
        motion_mean, motion_var = np.load(cfg.motion_statics, allow_pickle=True)['mean'], np.load(cfg.motion_statics, allow_pickle=True)['var']
        motion_std = np.sqrt(motion_var)
        motion_mean, motion_std = torch.from_numpy(motion_mean).to(device), torch.from_numpy(motion_std).to(device)
        motion_mean, motion_std = motion_mean.squeeze(0), motion_std.squeeze(0)
    
    # loading input conditional signal for cross-modal completion
    if args.sample_mode == 'motion_to_video':
        assert args.input_motion is not None, "input_motion is None"
        motion_gt_dict, motion_gt = load_motion(motion_path=args.input_motion, fetch_local_motion=True, init_canonical=False, use_shape=True, max_frame_len=cfg.T)
        input_len = len(motion_gt)

        if input_len >= cfg.T:
            adjusted_T = cfg.T
        else:
            adjusted_T = ((input_len - 1) // 4) * 4 + 1
            if adjusted_T >= input_len:
                adjusted_T = ((input_len - 1 - 4) // 4) * 4 + 1
                if adjusted_T <= 0:
                    adjusted_T = 1
            if adjusted_T == 0:
                adjusted_T = 4 

        cfg.T = adjusted_T
        motion_gt = motion_gt[:cfg.T]
        motion_gt = normalize_motion(motion_gt, motion_mean, motion_std)
        motion_gt = motion_gt.to(device)

    elif args.sample_mode == 'video_to_motion':
        assert args.input_video is not None, "input_video is None"
        
        orig_H, orig_W, orig_T = get_video_dimensions(args.input_video)
        
        if orig_T >= cfg.T:
            adjusted_T = cfg.T
        else:
            adjusted_T = ((orig_T - 1) // 4) * 4 + 1
            if adjusted_T >= orig_T:
                adjusted_T = ((orig_T - 1 - 4) // 4) * 4 + 1
                if adjusted_T <= 0:
                    adjusted_T = 1
            if adjusted_T == 0:
                adjusted_T = 4

        cfg.T = adjusted_T

        aspect_ratio = orig_H / orig_W
        target_area = cfg.H * cfg.W
        
        target_W = int((target_area / aspect_ratio) ** 0.5)
        target_H = int(target_area / target_W)
        
        target_H = (target_H + 31) // 32 * 32
        target_W = (target_W + 31) // 32 * 32

        cfg.H, cfg.W = target_H, target_W
        
        video = load_video(video_path=args.input_video, ow=target_W, oh=target_H, video_len=cfg.T)
        logging.info(f"Original: {orig_H}x{orig_W}x{orig_T} -> Loaded: {video.shape[2]}x{video.shape[3]}x{video.shape[1]}")

    # update seq_len_video, seq_len_motion and target_shape
    if args.sample_mode == 'video_to_motion' or args.sample_mode == 'motion_to_video':
        cfg.t = (cfg.T - 1) // 4 + 1
        cfg.seq_len_video = cfg.t * cfg.H * cfg.W // 32 // 32
        cfg.seq_len_motion = cfg.T * cfg.num_motion_latent_per_frame
        cfg.target_shape = (48, cfg.t, cfg.H//16, cfg.W//16)

    # initialize models
    # [model] t5
    t5 = T5Encoder(
        name=cfg.t5_model,
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=cfg.t5_checkpoint,
        tokenizer_path=cfg.t5_tokenizer
    )

    # [model] vae
    vae = Wan2_2_VAE(
        vae_pth=cfg.vae_checkpoint,
        device=device
    )

    # [model] transformer
    model = EchoMotion_DiT(
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            in_dim=vae.model.z_dim,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            text_dim=t5.model.dim,
            out_dim=vae.model.z_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
            inject_motion_type=cfg.inject_motion_type,
            motion_dim=cfg.motion_dim,
            use_task_embedding=cfg.use_task_embedding,
            MoT_layers=cfg.MoT_layers,
            use_shape=cfg.use_shape
        )

    logging.info(f"Loading checkpoint from {cfg.checkpoint_path}")
    state_dict = torch.load(cfg.checkpoint_path, map_location='cpu')
    status = model.load_state_dict(state_dict, strict=True)

    model = model.to(device).eval()

    # clear cache
    torch.cuda.empty_cache()
        
    logging.info(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        args.prompt = call_with_messages(
            args.prompt,
            input_video_path=args.input_video,
            system_prompt=None,
            model_name=None,
            task_name=args.sample_mode,
            seed=cfg.seed)
        logging.info(f"Extended prompt: {args.prompt}")

    out_name = os.path.join(cfg.out_dir, str(args.sample_mode), f'_seed{cfg.seed}_{args.sample_mode}')
    logging.info(f'out_name: {out_name}')
    
    # preprocess text prompt 
    context = t5([args.prompt])
    context_null = t5([cfg.neg_prompt])
    context_null_motion = t5([cfg.neg_prompt])
                
    if args.sample_mode == 'joint':

        seed_g = torch.Generator(device=device)
        if not args.seed:
            current_seed = cfg.seed
        else:
            current_seed = args.seed

        seed_g.manual_seed(current_seed)
        logging.info(f"Sampling joint with seed {current_seed}")

        noise = [torch.randn(cfg.target_shape[0], cfg.target_shape[1], cfg.target_shape[2], cfg.target_shape[3], dtype=torch.float32, device=device, generator=seed_g)]
        noise_motion = [torch.randn(cfg.T, cfg.motion_dim, dtype=torch.float32, device=device, generator=seed_g)]
        
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sample_scheduler_motion = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)
        sampling_sigmas_motion = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift_motion)

        # sample videos
        latents = noise
        motion = noise_motion
        for j in range(len(latents)):
            with (
                amp.autocast(dtype=cfg.param_dtype),
                torch.no_grad(),
            ):

                timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
                timesteps_motion, _ = retrieve_timesteps(sample_scheduler_motion, device=device, sigmas=sampling_sigmas_motion)

                arg_c = {'context': context[j:j+1], 'context_motion': context[j:j+1], 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['video_with_motion']}
                arg_null = {'context': context_null[j:j+1], 'context_motion': context_null_motion[j:j+1], 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['video_with_motion']}

                for i, t in enumerate(tqdm(timesteps)):
                    
                    latent_model_input = latents[j:j+1]
                    motion_input = motion[j:j+1]

                    timestep = [t]
                    timestep_motion = [timesteps_motion[i]]

                    timestep = torch.stack(timestep)
                    timestep_motion = torch.stack(timestep_motion)

                    noise_pred_cond, motion_pred_cond = model(x=TensorList(latent_model_input), motion=TensorList(motion_input), t=timestep, t_motion=timestep_motion,  **arg_c)
                    noise_pred_cond, motion_pred_cond = TensorList(noise_pred_cond), TensorList(motion_pred_cond)

                    noise_pred_uncond, motion_pred_uncond = model(x=TensorList(latent_model_input), motion=TensorList(motion_input), t=timestep, t_motion=timestep_motion, **arg_null)
                    noise_pred_uncond, motion_pred_uncond = TensorList(noise_pred_uncond), TensorList(motion_pred_uncond) 
                    
                    noise_pred = noise_pred_uncond + cfg.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
                    motion_pred = motion_pred_uncond + cfg.motion_sample_guide_scale * (motion_pred_cond - motion_pred_uncond)

                    temp_x0 = sample_scheduler.step(noise_pred[0].unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                    temp_x0_motion = sample_scheduler_motion.step_motion(motion_pred[0].unsqueeze(0), timesteps_motion[i], motion[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]

                    latents[j] = temp_x0.squeeze(0)
                    motion[j] = temp_x0_motion.squeeze(0)
                    
                x0 = latents[j]
                motion0 = motion[j]

            motion0 = denormalize_motion(motion0, motion_mean, motion_std)
            motion0 = recover_from_velocity(motion0, use_shape=cfg.use_shape)

            save_file = out_name + f'{current_seed}_{args.prompt[:60].replace(" ", "_").replace(",", "_")}.mp4'
            fake_video = vae.decode([x0]) 

            logging.info(f"Saving video to {save_file}")
            cache_video(
                tensor=fake_video[0].unsqueeze(0),
                save_file=save_file,
                fps=24,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )  

            save_file_motion = out_name + f'_pred_motion_{current_seed}_{args.prompt[:60].replace(" ", "_").replace(",", "_")}.mp4'
            motion_vis_during_validation(motion_data=motion0, 
                                            output_dir=None, 
                                            H=cfg.H, W=cfg.W, 
                                            fps=24, 
                                            motion_name=None, 
                                            device=motion0.device, 
                                            verbose=True,
                                            output_video_path= save_file_motion,
                                            zero_trans=cfg.fetch_local_motion,
                                            is_6d=True,
                                            use_shape=cfg.use_shape)
            
    elif args.sample_mode == 'motion_to_video':
        seed_g = torch.Generator(device=device)
        if not args.seed:
            current_seed = cfg.seed
        else:
            current_seed = args.seed
        seed_g.manual_seed(current_seed)
        logging.info(f"Sampling motion to video with seed {current_seed}")

        if args.input_video is not None:
            logging.info(f"Adjusting H and W based on input video aspect ratio...")
            orig_H, orig_W, orig_T = get_video_dimensions(args.input_video)

            aspect_ratio = orig_H / orig_W
            target_area = cfg.H * cfg.W

            target_W = int((target_area / aspect_ratio) ** 0.5)
            target_H = int(target_area / target_W)

            target_H = (target_H + 31) // 32 * 32
            target_W = (target_W + 31) // 32 * 32

            cfg.H, cfg.W = target_H, target_W
            logging.info(f"Original: {orig_H}x{orig_W} -> Adjusted: {cfg.H}x{cfg.W}")

            cfg.t = (cfg.T - 1) // 4 + 1
            cfg.seq_len_video = cfg.t * cfg.H * cfg.W // 32 // 32
            cfg.target_shape = (48, cfg.t, cfg.H//16, cfg.W//16)

        noise = [torch.randn(cfg.target_shape[0], cfg.target_shape[1], cfg.target_shape[2], cfg.target_shape[3], dtype=torch.float32, device=device, generator=seed_g)]
        
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)

        # sample videos
        latents = noise
        motion = [motion_gt]
        motino_null = [torch.zeros_like(motion[0], device=motion[0].device)]

        for j in range(len(latents)):
            with (
                amp.autocast(dtype=cfg.param_dtype),
                torch.no_grad(),
            ):

                timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

                arg_c = {'context': context[j:j+1], 'context_motion': context[j:j+1], 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['motion_to_video']}
                arg_null = {'context': context_null[j:j+1], 'context_motion': context_null_motion[j:j+1], 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['motion_to_video']}
                arg_wo_text = {'context': context[j:j+1], 'context_motion': context[j:j+1], 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['motion_to_video']}

                for i, t in enumerate(tqdm(timesteps)):
                    
                    latent_model_input = latents[j:j+1]
                    motion_input = motion[j:j+1]

                    timestep = [t]

                    timestep = torch.stack(timestep)

                    # sample null
                    noise_pred_null, motion_pred_null = model(TensorList(latent_model_input), motion=TensorList(motino_null), t=timestep, **arg_null)
                    noise_pred_null, motion_pred_null = TensorList(noise_pred_null), TensorList(motion_pred_null) 

                    # sample without text
                    noise_pred_wo_text, motion_pred_wo_text = model(TensorList(latent_model_input), motion=TensorList(motion_input), t=timestep, **arg_wo_text)
                    noise_pred_wo_text, motion_pred_wo_text = TensorList(noise_pred_wo_text), TensorList(motion_pred_wo_text)          

                    # full condition sample
                    noise_pred_cond, motion_pred_cond = model(TensorList(latent_model_input), motion=TensorList(motion_input), t=timestep, **arg_c)
                    noise_pred_cond, motion_pred_cond = TensorList(noise_pred_cond), TensorList(motion_pred_cond) 

                    # in-context classifier free guidance
                    noise_pred = noise_pred_null \
                    + cfg.motion_guide_video_scale * (noise_pred_wo_text - noise_pred_null) \
                    + cfg.sample_guide_scale  * (noise_pred_cond - noise_pred_wo_text)

                    temp_x0 = sample_scheduler.step(noise_pred[0].unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                    latents[j] = temp_x0.squeeze(0)
                    
                x0 = latents[j]
                motion0 = motion[j]

            motion0 = denormalize_motion(motion0, motion_mean, motion_std)
            motion0 = recover_from_velocity(motion0, use_shape=cfg.use_shape)

            motion_basename = os.path.splitext(os.path.basename(args.input_motion))[0]
            save_file = out_name + f'{current_seed}_{motion_basename}_{args.prompt[:60].replace(" ", "_").replace(",", "_")}.mp4'
            fake_video = vae.decode([x0]) 
            logging.info(f"Saving video to {save_file}")
            cache_video(
                tensor=fake_video[0].unsqueeze(0),
                save_file=save_file,
                fps=24,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )  

            save_file_motion = out_name + f'_conditional_motion_{current_seed}_{motion_basename}_{args.prompt[:60].replace(" ", "_").replace(",", "_")}.mp4'
            motion_vis_during_validation(motion_data=motion0, 
                                            output_dir=None, 
                                            H=cfg.H, W=cfg.W,
                                            fps=24,
                                            motion_name=None,
                                            device=motion0.device,
                                            verbose=True,
                                            output_video_path= save_file_motion,
                                            zero_trans=cfg.fetch_local_motion,
                                            is_6d=True,
                                            use_shape=cfg.use_shape)

    elif args.sample_mode == 'video_to_motion':
        seed_g = torch.Generator(device=device)
        if not args.seed:
            current_seed = cfg.seed
        else:
            current_seed = args.seed
        seed_g.manual_seed(current_seed)

        logging.info(f"Sampling video to motion with seed {current_seed}")

        noise_motion = [torch.randn(cfg.T, cfg.motion_dim, dtype=torch.float32, device=device, generator=seed_g)]
        
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sample_scheduler_motion = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)
        sampling_sigmas_motion = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift_motion)

        # encode video to latent
        video = video.to(device=device, dtype=torch.float32)
        latents = vae.encode([video])
        
        # sample videos
        latents = latents
        latent_model_input_null = [torch.zeros_like(latents[0], device=latents[0].device)]
        motion = noise_motion

        for j in range(len(latents)):
            with (
                amp.autocast(dtype=cfg.param_dtype),
                torch.no_grad(),
            ):

                timesteps_motion, _ = retrieve_timesteps(sample_scheduler_motion, device=device, sigmas=sampling_sigmas_motion)

                arg_c = {'context': t5(['']), 'context_motion': t5(['']), 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['video_to_motion']}
                arg_null = {'context': t5(['']), 'context_motion': t5(['']), 'seq_len_video': cfg.seq_len_video, 'seq_len_motion': cfg.seq_len_motion, 'pred_task': ['video_to_motion']}

                for i, t in enumerate(tqdm(timesteps_motion)):
                    
                    latent_model_input = latents[j:j+1]
                    motion_input = motion[j:j+1]

                    timestep = [t]
                    timestep_motion = [timesteps_motion[i]]

                    timestep = torch.stack(timestep)

                    noise_pred_cond, motion_pred_cond = model(x=TensorList(latent_model_input), motion=TensorList(motion_input), t=timestep, t_motion=timestep_motion,  **arg_c)
                    _, motion_pred_cond = TensorList(noise_pred_cond), TensorList(motion_pred_cond)

                    noise_pred_uncond, motion_pred_uncond = model(x=TensorList(latent_model_input_null), motion=TensorList(motion_input), t=timestep, t_motion=timestep_motion, **arg_null)
                    _, motion_pred_uncond = TensorList(noise_pred_uncond), TensorList(motion_pred_uncond) 
                    
                    motion_pred = motion_pred_uncond + cfg.video_guide_motion_scale * (motion_pred_cond - motion_pred_uncond)

                    temp_x0_motion = sample_scheduler_motion.step_motion(motion_pred[0].unsqueeze(0), timesteps_motion[i], motion[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                    motion[j] = temp_x0_motion.squeeze(0)

                x0 = latents[j]
                motion0 = motion[j]

            motion0 = denormalize_motion(motion0, motion_mean, motion_std)
            motion0 = recover_from_velocity(motion0, use_shape=cfg.use_shape)

            video_basename = os.path.splitext(os.path.basename(args.input_video))[0]
            save_file = out_name + f'{current_seed}_{video_basename}.mp4'
            fake_video = vae.decode([x0]) 
            logging.info(f"Saving video to {save_file}")
            cache_video(
                tensor=fake_video[0].unsqueeze(0),
                save_file=save_file,
                fps=24,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )  

            save_file_motion = out_name + f'_pred_motion_{current_seed}_{video_basename}.mp4'
            motion_vis_during_validation(motion_data=motion0,
                                            output_dir=None,
                                            H=cfg.H, W=cfg.W, 
                                            fps=24, 
                                            motion_name=None, 
                                            device=motion0.device, 
                                            verbose=True,
                                            output_video_path= save_file_motion,
                                            zero_trans=cfg.fetch_local_motion,
                                            is_6d=True,
                                            use_shape=cfg.use_shape) 

    logging.info("finished sampling")
