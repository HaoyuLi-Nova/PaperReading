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
import json
import torch
import random
import cv2
import decord
from einops import rearrange
from utils import *

# --------------------------------------------------------------------------
# 1. Load scenes infomation
# --------------------------------------------------------------------------
dataset_root = 'root_path/360Motion-Dataset'
video_res = '480_720'
video_names = []
scenes = ['Desert', 'HDRI']
scene_location_pair = {
    'Desert' : 'desert',
    'HDRI' : 
    {
        'loc1' : 'snowy street',
        'loc2' : 'park',
        'loc3' : 'indoor open space',
        'loc11' : 'gymnastics room',
        'loc13' : 'autumn forest',
    }
}
for scene in scenes:
    video_path = os.path.join(dataset_root, video_res, scene)
    locations_path = os.path.join(video_path, "location_data.json")
    with open(locations_path, 'r') as f: locations = json.load(f)
    locations_info = {locations[idx]['name']:locations[idx] for idx in range(len(locations))}
    for video_name in os.listdir(video_path):
        if video_name.endswith('Hemi12_1') == True:
            if scene != 'HDRI':
                location = scene_location_pair[scene]
            else:
                location = scene_location_pair['HDRI'][video_name.split('_')[1]]
            video_names.append((video_res, scene, video_name, location, locations_info))

# --------------------------------------------------------------------------
# 2. Load 12 surrounding cameras
# --------------------------------------------------------------------------
cam_num = 12
max_objs_num = 3
length = len(video_names)
captions_path = os.path.join(dataset_root, "CharacterInfo.json")
with open(captions_path, 'r') as f: captions = json.load(f)['CharacterInfo']
captions_info = {int(captions[idx]['index']):captions[idx]['eng'] for idx in range(len(captions))}
cams_path = os.path.join(dataset_root, "Hemi12_transforms.json")
with open(cams_path, 'r') as f: cams_info = json.load(f)
cam_poses = []
for i, key in enumerate(cams_info.keys()):
    if "C_" in key:
        cam_poses.append(parse_matrix(cams_info[key]))
cam_poses = np.stack(cam_poses)
cam_poses = np.transpose(cam_poses, (0,2,1))
cam_poses = cam_poses[:,:,[1,2,0,3]]
cam_poses[:,:3,3] /= 100.
cam_poses = cam_poses
sample_n_frames = 49

# --------------------------------------------------------------------------
# 3. Load a sample of video & object poses
# --------------------------------------------------------------------------
(video_res, scene, video_name, location, locations_info) = video_names[20]

with open(os.path.join(dataset_root, video_res, scene, video_name, video_name+'.json'), 'r') as f: objs_file = json.load(f)
objs_num = len(objs_file['0'])
video_index = random.randint(1, cam_num-1)

location_name = video_name.split('_')[1]
location_info = locations_info[location_name]
cam_pose = cam_poses[video_index-1]
obj_transl = location_info['coordinates']['CameraTarget']['position']

prompt = ''
video_caption_list = []
obj_poses_list = []

for obj_idx in range(objs_num):

    obj_name_index = objs_file['0'][obj_idx]['index'] 
    video_caption = captions_info[obj_name_index]

    if video_caption.startswith(" "):
        video_caption = video_caption[1:]
    if video_caption.endswith("."):
        video_caption = video_caption[:-1]
    video_caption = video_caption.lower()
    video_caption_list.append(video_caption)
    
    obj_poses = load_sceneposes(objs_file, obj_idx, obj_transl)
    obj_poses = np.linalg.inv(cam_pose) @ obj_poses
    obj_poses_list.append(obj_poses)

for obj_idx in range(objs_num):
    video_caption = video_caption_list[obj_idx]
    if obj_idx == objs_num - 1:
        if objs_num == 1:
            prompt += video_caption + ' is moving in the ' + location
        else:
            prompt += video_caption + ' are moving in the ' + location
    else:
        prompt += video_caption + ' and '

obj_poses_all = torch.from_numpy(np.array(obj_poses_list))

total_frames = 99
current_sample_stride = 1.75
cropped_length = int(sample_n_frames * current_sample_stride)
start_frame_ind = random.randint(10, max(10, total_frames - cropped_length - 1))
end_frame_ind = min(start_frame_ind + cropped_length, total_frames)
frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, sample_n_frames, dtype=int)

video_frames_path = os.path.join(dataset_root, video_res, scene, video_name, 'videos', video_name+ f'_C_{video_index:02d}_35mm.mp4')
cap = cv2.VideoCapture(video_frames_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# get local rank
ctx = decord.cpu(0)
reader = decord.VideoReader(video_frames_path, ctx=ctx, height=height, width=width)
assert len(reader) == total_frames or len(reader) == total_frames+1
frame_indexes = [frame_idx for frame_idx in range(total_frames)]
try:
    video_chunk = reader.get_batch(frame_indexes).asnumpy()    
except:
    video_chunk = reader.get_batch(frame_indexes).numpy()

pixel_values = np.array([video_chunk[indice] for indice in frame_indices])
pixel_values = rearrange(torch.from_numpy(pixel_values) / 255.0, "f h w c -> f c h w")

save_video = True
if save_video:
    video_data = (pixel_values.cpu().to(torch.float32).numpy() * 255).astype(np.uint8)
    video_data = rearrange(video_data, "f c h w -> f h w c")
    save_images2video(video_data, video_name, 12)
