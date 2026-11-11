import torch
from wan.vae import WanVAE  # Wan2.1 VAE, z_dim=16, 空间8x, 时间4x

B = 1
T = 81                  # 4n+1
H, W = 480, 832
H_lat, W_lat = H // 8, W // 8   # 60, 104
T_lat = (T - 1) // 4 + 1        # 21

# 画布：首帧=参考图，其余全零
canvas = torch.zeros(B, 3, T, H, W)
canvas[:, :, 0] = reference_image   # [B,3,H,W]

z_video = vae.encode(canvas)        # [B, 16, 21, 60, 104]
image_latent = vae.encode(reference_image.unsqueeze(2))  
# [B, 16, 1, 60, 104]

mask_1ch = torch.ones(B, 1, T_lat, H_lat, W_lat)
mask_1ch[:, :, 0] = 0.0                # 首帧潜位置为条件，不需要生成

# 4 通道：对应 VAE 时间压缩比 4
mask = mask_1ch.repeat_interleave(4, dim=1)  
# [B, 4, 21, 60, 104]

# 当前去噪步的噪声潜变量
noisy_latent = scheduler.add_noise(z_video, noise, timestep)  
# [B, 16, 21, 60, 104]

# 先拼条件块：mask(4) + image_latent(16) = 20ch
condition = torch.cat([mask, image_latent], dim=1)  
# [B, 20, 21, 60, 104]

# 再与噪声拼接：noise(16) + condition(20) = 36ch
z_concat = torch.cat([noisy_latent, condition], dim=1)  
# [B, 36, 21, 60, 104]

# 当前去噪步的噪声潜变量
noisy_latent = scheduler.add_noise(z_video, noise, timestep)  
# [B, 16, 21, 60, 104]

# 先拼条件块：mask(4) + image_latent(16) = 20ch
condition = torch.cat([mask, image_latent], dim=1)  
# [B, 20, 21, 60, 104]

# 再与噪声拼接：noise(16) + condition(20) = 36ch
z_concat = torch.cat([noisy_latent, condition], dim=1)  
# [B, 36, 21, 60, 104]