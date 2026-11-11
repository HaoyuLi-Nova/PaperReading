# ParticleSfM
# Copyright (C) 2022  ByteDance Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
from os.path import dirname
import numpy as np
import argparse
import tqdm

import numpy as np
import imageio
import os
from glob import glob
from multiprocessing import Pool
import json
import math
from scipy.spatial.transform import Rotation as R

def batch_rotation_matrix_angle_error(R1_batch, R2_batch):

    assert R1_batch.shape == R2_batch.shape
    assert R1_batch.shape[1:] == (3, 3)
    
    B = R1_batch.shape[0]
    angle_errors = np.zeros(B)  
    
    for i in range(B):
        R_relative = np.dot(R1_batch[i].T, R2_batch[i])
        
        r = R.from_matrix(R_relative)
        angle_error = r.magnitude()
        
        # angle_errors[i] = np.degrees(angle_error)
        angle_errors[i] = angle_error
    
    return angle_errors

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def matrix_to_euler_angles(matrix):
    sy = math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(matrix[2][1], matrix[2][2])
        y = math.atan2(-matrix[2][0], sy)
        z = math.atan2(matrix[1][0], matrix[0][0])
    else:
        x = math.atan2(-matrix[1][2], matrix[1][1])
        y = math.atan2(-matrix[2][0], sy)
        z = 0

    return math.degrees(x), math.degrees(y), math.degrees(z)

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R.T

def extract_location_rotation(data):
    results = {}
    for key, value in data.items():
        matrix = parse_matrix(value)
        location = np.array([matrix[3][0], matrix[3][1], matrix[3][2]])
        rotation = eul2rot(matrix_to_euler_angles(matrix))
        transofmed_matrix = np.identity(4)
        transofmed_matrix[:3,3] = location
        transofmed_matrix[:3,:3] = rotation
        results[key] = transofmed_matrix
    return results

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def batch_axis_angle_to_rotation_matrix(r_batch):
    batch_size = r_batch.shape[0]
    rotation_matrices = []
    
    for i in range(batch_size):
        r = r_batch[i]
        theta = np.linalg.norm(r)
        if theta == 0:
            rotation_matrices.append(np.eye(3))
        else:
            k = r / theta 
            kx, ky, kz = k
            
            K = np.array([
                [0, -kz, ky],
                [kz, 0, -kx],
                [-ky, kx, 0]
            ])
            
            # Rodrigues formulation
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
            rotation_matrices.append(R)
    
    return np.array(rotation_matrices)


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    args = parser.parse_args()

    folder_path = args.folder
    video_files = os.listdir(folder_path)
    for video_file in video_files:
        if video_file.endswith('txt'):
            video_files.remove(video_file)
    num_video_files = len(video_files)

    with open(('eval_gt_poses.json'), 'r') as f: eval_gt_poses = json.load(f)
    transl_err_all, rotat_err_all = 0, 0

    for video_file in tqdm.tqdm(sorted(video_files)):

        obj_poses = np.array(eval_gt_poses[video_file])
            
        start_frame_ind = 10
        sample_n_frames = 77
        frame_indices = np.linspace(start_frame_ind, start_frame_ind + sample_n_frames - 1, sample_n_frames, dtype=int)
        obj_poses = obj_poses[frame_indices]
        
        # load smpl pose
        video_path = os.path.join(folder_path, video_file)
        smpl_poses = np.zeros_like(obj_poses)
        smpl_poses[:,3,3] = 1.
        obj_rotats = np.load(os.path.join(video_path, 'smpl_orient.npy'))
        smpl_poses[:,:3,:3] = batch_axis_angle_to_rotation_matrix(obj_rotats)
        smpl_poses[:,:3,3] = np.load(os.path.join(video_path, 'smpl_transl.npy'))

        # align y-axis orientation
        smpl_poses[:,:3,3][:,1] *= -1.
        smpl_poses[:,:,:2] *= -1.

        # align pose translation
        translation_bias = smpl_poses[0,:3,3] - obj_poses[0,:3,3]
        smpl_poses[:,:3,3] -= translation_bias

        # evaluation
        transl_err = np.linalg.norm(smpl_poses[:,:3,3] - obj_poses[:,:3,3],ord=2,axis=1).mean()
        rotat_err = batch_rotation_matrix_angle_error(obj_poses[:,:3,:3],smpl_poses[:,:3,:3]).mean()

        transl_err_all += transl_err
        rotat_err_all += rotat_err

        print(video_path)
        print('transl_err:{:.3f}'.format(transl_err))
        print('rotat_err:{:.3f}'.format(rotat_err))

    print('transl_err_all:{:.3f}'.format(transl_err_all/num_video_files))
    print('rotat_err_all:{:.3f}'.format(rotat_err_all/num_video_files))
