import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import argparse
import os
import cv2
import decord
import numpy as np
import tqdm
import glob
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument("-v1_f", "--videos1_folder", type=str)
parser.add_argument("-v2_f", "--videos2_folder", type=str)
args = parser.parse_args()

videos1_folder_path = args.videos1_folder
videos2_folder_path = args.videos2_folder

sub_folders = os.listdir(videos1_folder_path)
videos_name = []
for sub_folder in sub_folders:
    files = os.listdir(os.path.join(videos1_folder_path, sub_folder))
    for file in files:
        if file.endswith('.mp4'):
            video_name = os.path.join(sub_folder, file)
            videos_name.append(video_name)

base_dir = os.path.dirname(videos2_folder_path)
base_name = os.path.basename(videos2_folder_path)

os.makedirs(f'{base_dir}/eval_1', exist_ok=True)
os.makedirs(f'{base_dir}/eval_2', exist_ok=True)
# ps: pixel value should be in [0, 1]!
NUMBER_OF_VIDEOS = len(videos_name)
VIDEO_LENGTH = 77
CHANNEL = 3
H_SIZE = 384
W_SIZE = 672
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H_SIZE, W_SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H_SIZE, W_SIZE, requires_grad=False)

for video_idx, video_name in tqdm.tqdm(enumerate(videos_name)):

    print(video_name)
    video_frames_path = os.path.join(videos1_folder_path, video_name)
    cap = cv2.VideoCapture(video_frames_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ctx = decord.cpu(0)
    reader = decord.VideoReader(video_frames_path, ctx=ctx, height=height, width=width)
    frame_indexes = [frame_idx for frame_idx in range(VIDEO_LENGTH)]
    try:
        video_chunk = reader.get_batch(frame_indexes).asnumpy()    
    except:
        video_chunk = reader.get_batch(frame_indexes).numpy()
    for frame_idx in range(VIDEO_LENGTH):
        cv2.imwrite(f'{base_dir}/eval_1/{video_idx:03d}_{frame_idx:02d}.png', video_chunk[frame_idx][:,:,::-1])
    video_chunk = video_chunk.transpose(0,3,1,2)/255.

    videos1[video_idx] = torch.from_numpy(video_chunk)

    video_frames_path = os.path.join(videos2_folder_path, video_name)    
    cap = cv2.VideoCapture(video_frames_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ctx = decord.cpu(0)
    reader = decord.VideoReader(video_frames_path, ctx=ctx, height=height, width=width)
    frame_indexes = [frame_idx for frame_idx in range(VIDEO_LENGTH)]
    try:
        video_chunk = reader.get_batch(frame_indexes).asnumpy()    
    except:
        video_chunk = reader.get_batch(frame_indexes).numpy()
    for frame_idx in range(VIDEO_LENGTH):
        cv2.imwrite(f'{base_dir}/eval_2/{video_idx:03d}_{frame_idx:02d}.png', video_chunk[frame_idx][:,:,::-1])
    
    video_chunk = video_chunk.transpose(0,3,1,2)/255.
    videos2[video_idx] = torch.from_numpy(video_chunk)

if NUMBER_OF_VIDEOS == 1:
    videos1 = videos1.repeat(2,1,1,1,1)
    videos2 = videos2.repeat(2,1,1,1,1)

print('load videos done')
device = torch.device("cuda")

import json
result = {}
result['fvd_styleganv'] = calculate_fvd(videos1, videos2, device, method='styleganv')
# result['fvd_videogpt'] = calculate_fvd(videos1, videos2, device, method='videogpt')

fvd_value = result['fvd_styleganv']['value']
print(f'FVD: {fvd_value}')