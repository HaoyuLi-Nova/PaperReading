import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CACHE_T = 2

# ============================================================
# 1. 因果 3D 卷积（支持分块推理缓存）
# ============================================================
class CausalConv3d(nn.Conv3d):
    """
    Wan2.2 风格的因果 3D 卷积。
    时间维只在左侧 padding，右侧不补，保证不看未来。
    支持 cache_x：分块推理时携带上一块末尾的帧。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将 (pad_t, pad_h, pad_w) 转换为 F.pad 需要的顺序
        # F.pad 对 (B,C,D,H,W) 的 padding 顺序是 [W_l, W_r, H_t, H_b, D_l, D_r]
        self._padding = (
            self.padding[2], self.padding[2],   # W 左/右
            self.padding[1], self.padding[1],   # H 上/下
            2 * self.padding[0], 0,             # D 左/右：左侧补 2*pad_t（因果）
        )
        self.padding = (0, 0, 0)  # 交给 F.pad 显式处理，Conv3d 本身不再 padding

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # 将缓存帧拼到时间维前面
            x = torch.cat([cache_x, x], dim=2)
            # 已经拼了真实历史，左侧零填充相应减少
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

# ============================================================
# 2. RMSNorm（Wan VAE 中取代部分 LayerNorm）
# ============================================================
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

# ============================================================
# 3. 上/下采样模块（2D 空间 + 可选的 3D 时间）
# ============================================================
class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d",
                        "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            # 时间维 2× 上采样：3D 因果卷积
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, cache_x=None):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # 空间 2D 上/下采样需要先合并 T 到 batch 维
        x_2d = rearrange(x, "b c t h w -> (b t) c h w")
        x_2d = self.resample(x_2d)
        x = rearrange(x_2d, "(b t) c h w -> b c t h w", t=T)

        if self.mode == "upsample3d":
            # 时间 2× 上采样：先重复再因果卷积
            x = x.repeat_interleave(2, dim=2)
            x = self.time_conv(x, cache_x)
        elif self.mode == "downsample3d":
            # 时间 2× 下采样：因果卷积 + 步长 2
            x = self.time_conv(x, cache_x)
            x = x[:, :, ::2, :, :]

        return x

# ============================================================
# 4. 残差块（Wan VAE 的基础构件）
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm1 = RMS_norm(in_dim)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)

        self.norm2 = RMS_norm(out_dim)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)

        if in_dim != out_dim:
            self.shortcut = CausalConv3d(in_dim, out_dim, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cache_x=None):
        h = self.conv1(F.silu(self.norm1(x)), cache_x)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

# ============================================================
# 5. 编码器
# ============================================================
class Wan22VAEEncoder(nn.Module):
    def __init__(self, z_dim=48, dim=160, dim_mult=(1, 2, 4, 4),
                 temperal_downsample=(False, True, True)):
        super().__init__()
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.temperal_downsample = temperal_downsample
        self.num_stages = len(dim_mult)

        dims = [dim * u for u in dim_mult]

        # 输入 patchify：将空间 2×2 折叠到通道
        self.patchify_conv = CausalConv3d(3 * 4, dims[0], 3, padding=1)

        # 每个 stage：Resample（下采样）+ 两个 ResidualBlock
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            in_dim = dims[i - 1] if i > 0 else dims[0]
            out_dim = dims[i]

            # 第一个 stage 不做空间下采样（已经在 patchify 中压缩了 2×）
            if i == 0:
                resample_mode = "none"
            else:
                resample_mode = "downsample3d" if temperal_downsample[i - 1] else "downsample2d"

            blocks = nn.ModuleList()
            blocks.append(ResidualBlock(in_dim, out_dim))
            blocks.append(ResidualBlock(out_dim, out_dim))
            self.stages.append(nn.ModuleDict({
                "resample": Resample(out_dim, mode=resample_mode) if resample_mode != "none" else nn.Identity(),
                "blocks": blocks,
            }))

        # 最终投影到 latent 分布参数（均值 + 对数方差）
        self.norm_out = RMS_norm(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], z_dim * 2, 3, padding=1)

    def forward(self, x, cache=None):
        # x: (B, 3, T, H, W)
        B, _, T, H, W = x.shape

        # 输入 patchify：空间 2×2 折叠到通道
        # (B, 3, T, H, W) -> (B, 3*4, T, H/2, W/2)
        x = rearrange(x, "b c t (h p1) (w p2) -> b (c p1 p2) t h w", p1=2, p2=2)
        x = self.patchify_conv(x)

        for stage in self.stages:
            if not isinstance(stage["resample"], nn.Identity):
                x = stage["resample"](x)
            for block in stage["blocks"]:
                x = block(x)

        x = self.conv_out(F.silu(self.norm_out(x)))
        # 分成均值和对数方差
        mean, logvar = torch.chunk(x, 2, dim=1)
        return mean, logvar

# ============================================================
# 6. 解码器（与编码器镜像，但更宽）
# ============================================================
class Wan22VAEDecoder(nn.Module):
    def __init__(self, z_dim=48, dim=160, dec_dim=256, dim_mult=(1, 2, 4, 4),
                 temperal_downsample=(False, True, True)):
        super().__init__()
        self.z_dim = z_dim
        self.dec_dim = dec_dim
        # 解码器 stage 顺序与编码器相反
        self.dim_mult = dim_mult[::-1]
        self.temperal_upsample = temperal_downsample[::-1]

        dims = [dec_dim * u for u in self.dim_mult]

        # 从 latent 投影到解码器宽度
        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.stages = nn.ModuleList()
        for i in range(len(self.dim_mult)):
            in_dim = dims[i - 1] if i > 0 else dims[0]
            out_dim = dims[i]

            # 对称地，最后一个 stage 不做空间上采样（对应编码器第一个 stage）
            if i == len(self.dim_mult) - 1:
                resample_mode = "none"
            else:
                resample_mode = "upsample3d" if self.temperal_upsample[i] else "upsample2d"

            blocks = nn.ModuleList()
            blocks.append(ResidualBlock(in_dim, out_dim))
            blocks.append(ResidualBlock(out_dim, out_dim))
            self.stages.append(nn.ModuleDict({
                "blocks": blocks,
                "resample": Resample(out_dim, mode=resample_mode) if resample_mode != "none" else nn.Identity(),
            }))

        # 最终上采样回像素空间（空间 2×2，对应输入 patchify）
        self.norm_out = RMS_norm(dims[-1])
        self.conv_out = CausalConv3d(dims[-1], 3, 3, padding=1)

    def forward(self, z):
        # z: (B, z_dim, T, H, W)
        x = self.conv_in(z)

        for stage in self.stages:
            for block in stage["blocks"]:
                x = block(x)
            if not isinstance(stage["resample"], nn.Identity):
                x = stage["resample"](x)

        x = self.conv_out(F.silu(self.norm_out(x)))

        # 最终空间 2×2 上采样（unpatchify）
        x = rearrange(x, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", p1=2, p2=2)
        return torch.tanh(x)  # 输出范围 [-1, 1]

# ============================================================
# 7. 完整 VAE
# ============================================================
class Wan22VAE(nn.Module):
    def __init__(self, z_dim=48, c_dim=160, dec_dim=256, dim_mult=(1, 2, 4, 4),
                 temperal_downsample=(False, True, True)):
        super().__init__()
        self.encoder = Wan22VAEEncoder(
            z_dim=z_dim, dim=c_dim, dim_mult=dim_mult,
            temperal_downsample=temperal_downsample,
        )
        self.decoder = Wan22VAEDecoder(
            z_dim=z_dim, dim=c_dim, dec_dim=dec_dim, dim_mult=dim_mult,
            temperal_downsample=temperal_downsample,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return self.decode(z), mean, logvar

# ============================================================
# 8. 快速形状验证
# ============================================================
if __name__ == "__main__":
    vae = Wan22VAE(z_dim=48, c_dim=160, dec_dim=256)
    print(f"VAE 参数量: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f} M")

    # 模拟一个 1 帧 64×64 的输入
    x = torch.randn(1, 3, 1, 64, 64)
    mean, logvar = vae.encode(x)
    print(f"输入: {x.shape}")
    print(f"Latent (mean): {mean.shape}")  # 预期: (1, 48, 1, 4, 4)

    recon = vae.decode(mean)
    print(f"重建: {recon.shape}")  # 预期: (1, 3, 1, 64, 64)