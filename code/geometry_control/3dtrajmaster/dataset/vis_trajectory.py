import trimesh
import numpy as np
import imageio
import copy
import cv2
import os
from glob import glob
import open3d
from multiprocessing import Pool
import json
from utils import *

if __name__ == '__main__' :

    H = 480
    W = 720
    intrinsics = np.array([[1060.606,0.],
                           [0., 1060.606]])
    
    cam_path = "traj_vis/Hemi12_transforms.json"
    location_path = "traj_vis/location_data_desert.json"
    video_name = "D_loc1_61_t3n13_003d_Hemi12_1.json"

    with open(location_path, 'r') as f: locations = json.load(f)
    locations_info = {locations[idx]['name']:locations[idx] for idx in range(len(locations))}
    location_name = video_name.split('_')[1]
    location_info = locations_info[location_name]
    translation = location_info['coordinates']['CameraTarget']['position']
    vis_all = []
    
    # vis cam
    with open(cam_path, 'r') as file:
        data = json.load(file)
    cam_poses = []
    for i, key in enumerate(data.keys()):
        if "C_" in key:
            cam_poses.append(parse_matrix(data[key]))

    cam_poses = np.stack(cam_poses)
    cam_poses = np.transpose(cam_poses, (0,2,1))
    cam_poses[:,:3,3] /= 100.
    cam_poses = cam_poses[:,:,[1,2,0,3]]
    relative_pose = np.linalg.inv(cam_poses[0])
    cam_poses = relative_pose @ cam_poses
    # convert right-hand coord to left-hand coord
    cam_poses[:,:3,3][:,1] *= -1.
    cam_poses[:,:,:2] *= -1.

    cam_num = len(cam_poses)
    for idx in range(cam_num):
        cam_pose = cam_poses[idx]
        cam_points_vis = get_cam_points_vis(W, H, intrinsics, cam_pose, [0.4, 0.4, 0.4], frustum_length=1.)
        vis_all.append(cam_points_vis)

    # vis gt obj poses
    start_frame_ind = 10
    sample_n_frames = 77
    frame_indices = np.linspace(start_frame_ind, start_frame_ind + sample_n_frames - 1, sample_n_frames, dtype=int)
    
    with open('traj_vis/'+video_name, 'r') as file:
        data = json.load(file)
    obj_poses = []
    for i, key in enumerate(data.keys()):
        obj_poses.append(parse_matrix(data[key][0]['matrix']))
    obj_poses = np.stack(obj_poses)
    obj_poses = np.transpose(obj_poses, (0,2,1))
    obj_poses[:,:3,3] -= translation
    obj_poses[:,:3,3] /= 100.
    obj_poses = obj_poses[:, :, [1,2,0,3]]
    obj_poses = relative_pose @ obj_poses
    obj_poses = obj_poses[frame_indices]
    # convert right-hand coord to left-hand coord
    obj_poses[:,:3,3][:,1] *= -1.
    obj_poses[:,:,:2] *= -1.

    obj_num = len(obj_poses)
    for idx in range(obj_num):
        obj_pose = obj_poses[idx]
        if idx % 5 == 0:
            cam_points_vis = get_cam_points_vis(W, H, intrinsics, obj_pose, [0.8, 0., 0.], frustum_length=0.5)
            vis_all.append(cam_points_vis)

    if len(data[key])>=2:
        with open('traj_vis/'+video_name, 'r') as file:
            data = json.load(file)
        obj_poses = []
        for i, key in enumerate(data.keys()):
            obj_poses.append(parse_matrix(data[key][1]['matrix']))
        obj_poses = np.stack(obj_poses)
        obj_poses = np.transpose(obj_poses, (0,2,1))
        obj_poses[:,:3,3] -= translation
        obj_poses[:,:3,3] /= 100.
        obj_poses = obj_poses[:, :, [1,2,0,3]]
        obj_poses = relative_pose @ obj_poses
        obj_poses = obj_poses[frame_indices]
        # convert right-hand coord to left-hand coord
        obj_poses[:,:3,3][:,1] *= -1.
        obj_poses[:,:,:2] *= -1.
        obj_num = len(obj_poses)
        for idx in range(obj_num):
            obj_pose = obj_poses[idx]
            if (idx % 5 == 0) :
                cam_points_vis = get_cam_points_vis(W, H, intrinsics, obj_pose, [0., 0.8,0.], frustum_length=0.5)
                vis_all.append(cam_points_vis)

    if len(data[key])>=3:
        with open('traj_vis/'+video_name, 'r') as file:
            data = json.load(file)
        obj_poses = []
        for i, key in enumerate(data.keys()):
            obj_poses.append(parse_matrix(data[key][2]['matrix']))
        obj_poses = np.stack(obj_poses)
        obj_poses = np.transpose(obj_poses, (0,2,1))
        obj_poses[:,:3,3] -= translation
        obj_poses[:,:3,3] /= 100.
        obj_poses = obj_poses[:, :, [1,2,0,3]]
        obj_poses = relative_pose @ obj_poses
        obj_poses = obj_poses[frame_indices]
        # convert right-hand coord to left-hand coord
        obj_poses[:,:3,3][:,1] *= -1.
        obj_poses[:,:,:2] *= -1.
        obj_num = len(obj_poses)
        for idx in range(obj_num):
            obj_pose = obj_poses[idx]
            if (idx % 5 == 0):
                cam_points_vis = get_cam_points_vis(W, H, intrinsics, obj_pose, [0., 0., 0.8], frustum_length=0.5)
                vis_all.append(cam_points_vis)

    # vis coordinates
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0,0,0])
    
    open3d.visualization.draw_geometries(vis_all)
