import decord
from decord import VideoReader, cpu
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import numpy as np

def get_video_dimensions(video_path):
    """
    Get the dimensions of a video file without loading the full video.
    Returns:
        tuple: (height, width, num_frames)
    """
    reader = VideoReader(video_path)
    num_frames = len(reader)
    # Get first frame to determine dimensions
    frame = reader[0]
    h, w = frame.shape[0], frame.shape[1]
    return h, w, num_frames

def _preprocess_video(video, oh, ow):
    """
    Resize, center crop, convert to tensor, and normalize.
    [t, h, w, c] -> [c, t, h, w]
    """
    # Permute to [t, c, h, w]
    video = video.permute(0, 3, 1, 2)

    # Resize and crop
    ih, iw = video.shape[2:]
    if ih != oh or iw != ow:
        # Calculate scale factor
        scale = max(oh / ih, ow / iw)
        
        # Resize
        video = F.interpolate(
            video,
            size=(round(scale * ih), round(scale * iw)),
            mode='bicubic',
            antialias=True
        )
        
        # Center crop
        x1 = (video.size(3) - ow) // 2
        y1 = (video.size(2) - oh) // 2
        video = video[:, :, y1:y1+oh, x1:x1+ow]

    # Normalize and permute to [c, t, h, w]
    video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
    return video

def load_video(video_path, ow=842, oh=480, video_len=121):
    """
    Load and preprocess video with flexible frame selection
    Args:
        video_path: path to video file
        ow: output width (after preprocessing)
        oh: output height (after preprocessing)
        frame_indices: optional list of frame indices to select
    Returns:
        torch.Tensor: preprocessed video tensor [C, T, H, W]
    """
    # Initialize video reader
    decord.bridge.set_bridge('torch')
    reader = VideoReader(video_path)
    
    # Get all frames if indices not specified
    actual_total_frames = len(reader)
    selected_frames_count = min(actual_total_frames, video_len)

    frame_indices = list(range(selected_frames_count))
    
    try:
        # Read selected frames
        video = reader.get_batch(frame_indices)
    except decord.DECORDError as e:
        print(f"Error loading video {video_path}: {str(e)}")
        raise
    # Preprocess video
    return _preprocess_video(video, oh, ow)