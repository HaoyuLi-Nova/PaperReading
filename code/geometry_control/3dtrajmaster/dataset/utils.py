# Copyright 2024 Xiao Fu, CUHK, Kuaishou Tech. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# More information about the method can be found at http://fuxiao0719.github.io/projects/3dtrajmaster
# --------------------------------------------------------------------------
import os
import numpy as np
from io import BytesIO
import imageio.v2 as imageio
import open3d as o3d
import math
import trimesh
import json


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def load_sceneposes(objs_file, obj_idx, obj_transl):
    ext_poses = []
    for i, key in enumerate(objs_file.keys()):
        ext_poses.append(parse_matrix(objs_file[key][obj_idx]['matrix']))
    ext_poses = np.stack(ext_poses)
    ext_poses = np.transpose(ext_poses, (0,2,1))
    ext_poses[:,:3,3] -= obj_transl
    ext_poses[:,:3,3] /= 100.
    ext_poses = ext_poses[:, :, [1,2,0,3]]
    return ext_poses

def save_images2video(images, video_name, fps):
    fps = fps
    format = "mp4"  
    codec = "libx264"  
    ffmpeg_params = ["-crf", str(12)]
    pixelformat = "yuv420p" 
    video_stream = BytesIO()

    with imageio.get_writer(
        video_stream,
        fps=fps,
        format=format,
        codec=codec,
        ffmpeg_params=ffmpeg_params,
        pixelformat=pixelformat,
    ) as writer:
        for idx in range(len(images)):
            writer.append_data(images[idx])

    video_data = video_stream.getvalue()

    output_path = os.path.join(video_name + ".mp4")
    with open(output_path, "wb") as f:
        f.write(video_data)

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

def get_cam_points_vis(W, H, intrinsics, ext_pose, color,frustum_length):
    cam = get_camera_frustum((W, H), intrinsics, np.linalg.inv(ext_pose), frustum_length=frustum_length,  color=[0., 0., 1.])
    cam_points = cam[0]
    for item in cam[1]:
        cam_points = np.concatenate((cam_points, np.linspace(cam[0][item[0]], cam[0][item[1]], num=1000, endpoint=True, retstep=False, dtype=None)))
    cam_points[:,0]*=-1
    cam_points = trimesh.points.PointCloud(vertices = cam_points, colors=[0, 255, 0, 255])
    cam_points_vis = o3d.geometry.PointCloud()
    cam_points_vis.points = o3d.utility.Vector3dVector(cam_points)
    cam_points_vis.paint_uniform_color(color)
    return cam_points_vis

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
            
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
            rotation_matrices.append(R)
    
    return np.array(rotation_matrices)