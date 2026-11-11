import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Config:
    image_size, in_channels = 32, 3
    patch_size, D, depth, n_heads, mlp_ratio = 4, 128, 4, 4, 4
    t_sample       = "uniform"      # "uniform" | "logit_normal" | "cosine"
    logit_shift    = 0.05
    batch_size, lr, num_steps = 64, 2e-4, 2000
    ema_decay      = 0.999
    device         = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

def timestep_embedding(sigma, dim, device):
    """t: [B] -> [B, dim]"""
    half = dim // 2 
    index = torch.arange(half, devive=device)
    freqs = torch.exp((-math.log(10000) * index))
    args = torch.outer(sigma, freqs) # [B,dim/2] [[],[],[]...]
    return torch.cat([torch.sin(args),torch.cos(args)], dim=-1) # [B,dim]

def sample_t(B, device):
    return torch.sigmoid(torch.randn(B, device=device))

class SyntheticDataset(Dataset):
    def __init__(self, n=20000, size=32):
        self.n = n
        self.size = size
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.aragne(size,dtype=torch.float32),
            indexinf = 'ij'
            )

