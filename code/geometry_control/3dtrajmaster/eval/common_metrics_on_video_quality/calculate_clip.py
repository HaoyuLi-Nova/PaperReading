import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json
import os
from tqdm import tqdm
import torch
import clip
from PIL import Image
import cv2
import numpy as np
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_video_scores(video_path, prompt):
    video = cv2.VideoCapture(video_path)
    texts = [prompt]
    clip_score_list = []
    while True:
        ret, frame = video.read()

        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=texts, images=[image], return_tensors="pt", padding=True, truncation=True).to(device)
            logits_per_image = model(**inputs).logits_per_image
            clip_score = logits_per_image.item()
            clip_score_list.append(clip_score)
        else:
            break

    video.release()
    return sum(clip_score_list) / len(clip_score_list)


parser = argparse.ArgumentParser()
parser.add_argument("-v_f", "--videos_folder", type=str)
args = parser.parse_args()

videos_folder_path = args.videos_folder
prompts_path = '/ytech_m2v2_hdd/fuxiao/scenectrl/common_metrics_on_video_quality/eval_prompts.json'
with open(prompts_path, "r", encoding="utf-8") as f: prompts_dict = json.load(f)

sub_folders = os.listdir(videos_folder_path)
videos_name = []
for sub_folder in sub_folders:
    files = os.listdir(os.path.join(videos_folder_path, sub_folder))
    for file in files:
        if file.endswith('.mp4'):
            video_name = os.path.join(sub_folder, file)
            videos_name.append(video_name)

num_videos = len(videos_name)

prompts = []
video_paths = []
for video_name in videos_name:
    prompt = prompts_dict[video_name.split('/')[0]]
    video_path = os.path.join(videos_folder_path, video_name)
    prompts.append(prompt)
    video_paths.append(video_path)

import csv
CLIP_T = True
if CLIP_T:
    scores = []
    for i in tqdm(range(num_videos)):
        # 加载图片
        video_path = video_paths[i]
        
        # 准备文本
        texts = prompts[i]
        score = get_video_scores(video_path, texts)
        scores.append(score)

    print(f"CLIP-SIM: {sum(scores)/len(scores)/100.}")
    #### CLIP-T ####
    # basemodel: 33.44