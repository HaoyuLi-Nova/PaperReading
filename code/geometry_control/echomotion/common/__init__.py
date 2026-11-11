"""
Common utilities for EchoMotion project.
Shared code for motion representation, constants, and transformations.
"""

from .constants import JOINT_NUM
from .motion_rep import motion_rep_to_SMPL, apply_inverse_rotation
from .rotation_transform import (
    axis_angle_to_rot6d, 
    rot6d_to_axis_angle, 
    axis_angle_to_mat3x3, 
    mat3x3_to_rot6d, 
    rot6d_to_mat3x3, 
    mat3x3_to_axis_angle
)

__all__ = [
    'JOINT_NUM',
    'motion_rep_to_SMPL',
    'apply_inverse_rotation',
    'axis_angle_to_rot6d',
    'rot6d_to_axis_angle',
    'axis_angle_to_mat3x3',
    'mat3x3_to_rot6d',
    'rot6d_to_mat3x3',
    'mat3x3_to_axis_angle',
]

