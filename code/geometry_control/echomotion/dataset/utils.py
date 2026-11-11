import numpy as np
import torch
from common.motion_rep import motion_rep_to_SMPL
from common.constants import JOINT_NUM

def load_motion(motion_path, fetch_local_motion=False, init_canonical=True, use_shape=False, max_frame_len=300):
    motion = torch.load(motion_path)
    motion = motion["motion"]
    motion = torch.tensor(motion).float()
    if len(motion) > max_frame_len:
        motion = motion[:max_frame_len]
    smpl_params, joints, transl_vel, joints_vel = motion_rep_to_SMPL(motion, recover_from_velocity=False, fetch_local_motion=fetch_local_motion, init_canonical=init_canonical, use_shape=use_shape)
    motion_dict = {
                'global_orient': smpl_params['global_orient'],  # controls the global root orientation
                'body_pose': smpl_params['body_pose'],          # controls the body
                'transl': smpl_params['transl'],                # controls the finger articulation
                'transl_vel': transl_vel,                       # (seq_len-1, 3)
                'joints': joints,                               # controls the global body position, dim = 66
                'joints_vel': joints_vel,
                'betas': smpl_params['betas'] if use_shape else None
            } 
    motion = motion_dict_to_tensor(motion_dict, use_shape=use_shape)
    return motion_dict, motion

def motion_dict_to_tensor(motion_dict, use_shape=False):
    if not use_shape:
        motion = torch.cat([
                            motion_dict['global_orient'], 
                            motion_dict['body_pose'], 
                            motion_dict['transl'],
                            motion_dict['transl_vel'],
                            motion_dict['joints'],
                            ], 
                            dim=1)
    else:
        motion = torch.cat([
                            motion_dict['global_orient'], 
                            motion_dict['body_pose'], 
                            motion_dict['transl'],
                            motion_dict['transl_vel'],
                            motion_dict['joints'],
                            motion_dict['betas']
                            ], 
                            dim=1)
    return motion

def motion_tensor_to_dict(motion, use_shape=False):
    if not use_shape:
        motion_dict = {
                    'global_orient': motion[:, :6],  # controls the global root orientation
                    'body_pose': motion[:, 6:6+(JOINT_NUM-1)*6],  # controls the body
                    'transl': motion[:, 6+(JOINT_NUM-1)*6:9+(JOINT_NUM-1)*6],    # controls the finger articulation
                    'transl_vel': motion[:, 9+(JOINT_NUM-1)*6:12+(JOINT_NUM-1)*6],
                    'joints': motion[:, 12+(JOINT_NUM-1)*6: 12+(JOINT_NUM-1)*6+JOINT_NUM*3],        # controls the global body position, dim = 66
        }
    else:
        motion_dict = {
                    'global_orient': motion[:, :6],  # controls the global root orientation
                    'body_pose': motion[:, 6:6+(JOINT_NUM-1)*6],  # controls the body
                    'transl': motion[:, 6+(JOINT_NUM-1)*6:9+(JOINT_NUM-1)*6],    # controls the finger articulation
                    'transl_vel': motion[:, 9+(JOINT_NUM-1)*6:12+(JOINT_NUM-1)*6],
                    'joints': motion[:, 12+(JOINT_NUM-1)*6: 12+(JOINT_NUM-1)*6+JOINT_NUM*3],        # controls the global body position, dim = 66
                    'betas': motion[:, -10:]  # shape parameters
        }

    return motion_dict

def get_safe_std(std, device):
    safe_std = torch.maximum(std, torch.tensor(1e-5, device=device))
    return safe_std

def normalize_motion(motion_gt, mean, std, debug=False):
    mean = mean.to(motion_gt.device)
    std = std.to(motion_gt.device)
    std = get_safe_std(std, motion_gt.device)
    normed = motion_gt.clone()
    # print('before norme', normed[0])
    normed = (normed - mean) / std
    if not torch.all(torch.isfinite(normed)):
        print("!!!!!!!! FATAL: inf/NaN detected in smart_normalize !!!!!!!!!!")
        print(f"Shape of input motion_gt: {motion_gt.shape}")
        problematic_indices = torch.where(~torch.isfinite(normed))
        print(f"Problematic indices: {problematic_indices}")
        print(f"Original values at problematic indices: {motion_gt[problematic_indices]}")
        print(f"Mean at problematic indices: {mean[problematic_indices[1]]}")
        print(f"Std at problematic indices: {std[problematic_indices[1]]}")
        raise ValueError("Normalization resulted in inf/NaN values.")
    return normed

def denormalize_motion(motion_tensor, mean, std, debug=False):

    mean = mean.to(motion_tensor.device)
    std = std.to(motion_tensor.device)
    std = get_safe_std(std, motion_tensor.device)
    
    denormed = motion_tensor.clone()
    denormed = denormed * std + mean
    
    return denormed

def recover_from_velocity(motion_tensor, use_shape=False):
    seq_len = motion_tensor.shape[0]
    motion_dict = motion_tensor_to_dict(motion_tensor, use_shape=use_shape)
    transl = motion_dict['transl']
    transl_vel = motion_dict['transl_vel']

    trans_first_frame = transl[0:1]  # (1,3)
    trans_recovered = [trans_first_frame]
    for i in range(1, seq_len):
        trans_recovered.append(trans_recovered[i-1] + transl_vel[i-1:i])
    trans_recovered = torch.cat(trans_recovered, dim=0)
    
    motion_dict['transl'] = trans_recovered

    return motion_dict_to_tensor(motion_dict, use_shape=use_shape)