import torch
import numpy as np
from .constants import JOINT_NUM
from .rotation_transform import axis_angle_to_rot6d, rot6d_to_axis_angle, axis_angle_to_mat3x3, mat3x3_to_rot6d, rot6d_to_mat3x3, mat3x3_to_axis_angle
from scipy.spatial.transform import Rotation as R

smplx_root = torch.tensor([[ 0.0031, -0.3514,  0.0120]], dtype=torch.float32)

def apply_inverse_rotation(smpl_params_aligned, R, smplx_root=smplx_root):
    """
    Applies the inverse rotation to bring SMPL parameters back to the original coordinate system.

    Args:
        smpl_params_aligned (dict): Dictionary containing aligned SMPL parameters.
        R (torch.Tensor): The ORIGINAL rotation matrix of shape (3, 3) used for alignment.
        smplx_root (torch.Tensor): The same smplx_root used in the forward pass.

    Returns:
        smpl_params_original (dict): SMPL parameters in the original coordinate system.
    """
    device = smpl_params_aligned['global_orient'].device

    # 1. Calculate inverse rotation matrix
    R_inv = R.T.to(device)

    # 2. Inverse transform translation
    transl_aligned = smpl_params_aligned['transl'] + smplx_root.to(device)
    transl_original = torch.matmul(R_inv[None,], transl_aligned[..., None]).squeeze(-1)
    smpl_params_aligned['transl'] = transl_original - smplx_root.to(device)

    # 3. Inverse transform global orientation
    global_orient_mat_aligned = axis_angle_to_mat3x3(smpl_params_aligned['global_orient'].view(-1, 3))
    global_orient_mat_original = torch.matmul(R_inv[None,], global_orient_mat_aligned)
    smpl_params_aligned['global_orient'] = global_orient_mat_original

    return smpl_params_aligned

# def motion_rep_to_SMPL(motion, recover_from_velocity=False, equal_length=False, fetch_local_motion=False, init_canonical = False, use_shape=False, inverse_rot=True):
#     """
#     Convert motion representation to SMPL parameters.
    
#     Args:
#         motion: Motion tensor of shape (seq_len, motion_dim)
#         recover_from_velocity: Whether to recover motion from velocity
#         equal_length: Whether to maintain equal length
#         fetch_local_motion: Whether to use local motion representation
#         init_canonical: Whether to initialize in canonical pose
#         use_shape: Whether to include shape parameters (betas)
#         inverse_rot: Whether to apply inverse rotation transform
        
#     Returns:
#         smpl_data: Dictionary containing SMPL parameters
#         joints: Joint positions
#     """
#     if motion.shape[1] == JOINT_NUM*12+12:    # global motion
#         idx_bias = 0
#     elif motion.shape[1] == JOINT_NUM*18+30:    # local motion
#         if fetch_local_motion:
#             idx_bias = JOINT_NUM*6+18 # 22*6 + 18 = 150
#         else:
#             idx_bias = 0
#     elif motion.shape[1] == JOINT_NUM*18 + 30 + 10 and use_shape: # local motion with betas
#         if fetch_local_motion:
#             idx_bias = JOINT_NUM*6+18 # 22*6 + 18 + 10 = 170
#         else:
#             idx_bias = 0
#     else:
#         raise ValueError(f"Unexpected motion shape: {motion.shape}")
    
#     seq_len = motion.shape[0]

#     body_poses = motion[:, :(JOINT_NUM-1)*6]    # (seq_len, (joints_num-1)*6)
#     body_poses = rot6d_to_axis_angle(body_poses.reshape(-1, 6)).reshape(seq_len, -1)    # (seq_len, (joints_num-1)*3)

#     joints = motion[:, (idx_bias+JOINT_NUM*6-6):(idx_bias+JOINT_NUM*9-6)].reshape(seq_len, -1, 3)    # (seq_len, joints_num, 3)
#     joints_vel = motion[:, (idx_bias+JOINT_NUM*9-6):(idx_bias+JOINT_NUM*12-6)].reshape(seq_len, -1, 3)    # (seq_len, joints_num, 3)

#     global_orient = motion[:, (idx_bias+JOINT_NUM*12-6):(idx_bias+JOINT_NUM*12)]    # (seq_len, 6)
#     global_orient = rot6d_to_axis_angle(global_orient)    # (seq_len, 3)

#     trans = motion[:, (idx_bias+JOINT_NUM*12+6):(idx_bias+JOINT_NUM*12+9)]    # (seq_len, 3)
#     trans_vel = motion[:, (idx_bias+JOINT_NUM*12+9):(idx_bias+JOINT_NUM*12+12)]

#     if use_shape:
#         betas = motion[:,-10:]
#         # Calculate mean shape
#         betas = betas.mean(dim=0).repeat(trans.shape[0], 1)
#     else:
#         betas = None

#     smpl_data = {
#         'global_orient': global_orient,   # (seq_len, 3)
#         'body_pose': body_poses,          # (seq_len, (joints_num-1)*3)
#         'transl': trans,                   # (seq_len, 3)
#         'betas': betas,
#     }

#     R_motionx_to_amass = torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=torch.float32, device=body_poses.device)
#     if inverse_rot:
#         smpl_data = apply_inverse_rotation(smpl_data, R_motionx_to_amass)

#     return smpl_data, joints

def motion_rep_to_SMPL(motion, recover_from_velocity=False, equal_length=False, fetch_local_motion=False, init_canonical = False, verbose=True, max_transl=10, use_shape=False):
    if motion.shape[1] == JOINT_NUM*12+12:    # global motion
        idx_bias = 0
    elif motion.shape[1] == JOINT_NUM*18+30:    # local motion
        if fetch_local_motion:
            idx_bias = JOINT_NUM*6+18 # 22*6 + 18 = 150
        else:
            idx_bias = 0
    elif motion.shape[1] == JOINT_NUM*12+12+10: # global motion with shape
        idx_bias = 0
    elif motion.shape[1] == JOINT_NUM*18 + 30 + 10 and use_shape:
        if fetch_local_motion:
            idx_bias = JOINT_NUM*6+18
        else:
            idx_bias = 0
    else:
        raise ValueError(f"get unexpected motion shape: {motion.shape}")
    
    seq_len = motion.shape[0]

    body_poses = motion[:, :(JOINT_NUM-1)*6]    # (seq_len-1, (joints_num-1)*6) = (seq_len-1, 126)
    # body_poses = rot6d_to_axis_angle(body_poses.reshape(-1, 6)).reshape(seq_len, -1)    # (seq_len-1, (joints_num-1)*3) = (seq_len-1, 63)

     # (seq_len-1, 66) just for training
    joints = motion[:, (idx_bias+JOINT_NUM*6-6):(idx_bias+JOINT_NUM*9-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3) = （seq_len-1, 22, 3) <- (seq_len-1, 66) 
    joints_vel = motion[:, (idx_bias+JOINT_NUM*9-6):(idx_bias+JOINT_NUM*12-6)].reshape(seq_len, -1, 3)    # (seq_len-1, joints_num, 3)  

    # fetch the global or local motion
    global_orient = motion[:, (idx_bias+JOINT_NUM*12-6):(idx_bias+JOINT_NUM*12)]    # (seq_len-1, 6)
    # global_orient = rot6d_to_axis_angle(global_orient)    # (seq_len-1, 3)

    trans = motion[:, (idx_bias+JOINT_NUM*12+6):(idx_bias+JOINT_NUM*12+9)]    # (seq_len-1, 3)
    trans_vel = motion[:, (idx_bias+JOINT_NUM*12+9):(idx_bias+JOINT_NUM*12+12)]

    if use_shape:
        # assume that the shape of human do not change during the video
        betas = motion[:, -10:].mean(dim=0).repeat(trans.shape[0], 1)
    else:
        betas = None
    if use_shape and motion.shape[1] == JOINT_NUM*18+30 or motion.shape[1] == JOINT_NUM*12+12:
        betas = torch.zeros(trans.shape[0], 10, device=motion.device)

    smpl_data = {
        'global_orient': global_orient, 
        'body_pose': body_poses,
        'transl': trans,
        'betas': betas,
    }
    # if use_shape and motion.shape[1] == JOINT_NUM*18+30:
    #     print('betas when output', smpl_data['betas'].shape, smpl_data['betas'][0])

    joints = joints.reshape(seq_len, -1)  
    joints_vel = joints_vel.reshape(seq_len, -1)

    return smpl_data, joints, trans_vel, joints_vel
