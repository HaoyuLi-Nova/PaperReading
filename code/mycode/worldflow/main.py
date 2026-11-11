"""
WorldFlow 核心机制演示：几何嵌入 + 加噪位置 + Token 构造

三句话结论：
  1. 噪声加在【归一化 3D 位移】上，形状 [B, 48, 64, 3]
  2. 几何嵌入是【干净条件】，不参与加噪
  3. Token = Linear(带噪位移) + 几何嵌入 + 条件嵌入
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────
@dataclass
class Config:
    num_points:    int = 64      # 查询点数 N
    num_frames:    int = 49      # 原始视频帧数
    hidden_dim:    int = 128     # 生成器宽度 D
    geo_dim:       int = 64      # 几何中间维度 Dg
    num_freqs:     int = 8       # Fourier 频率数 L
    flow_scale:    float = 0.1   # 域归一化尺度 a_d
    latent_shape:  tuple = (16, 13, 18, 30)  # 模拟 VAE latent


# ─────────────────────────────────────────────────────────────
# 1. 通用工具
# ─────────────────────────────────────────────────────────────
def make_mlp(in_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    """简单的 MLP：in → hidden → out，中间 SiLU"""
    hidden = out_dim
    dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """把坐标 [..., C] 用多频 sin/cos 展开成 [..., C * 2L]。频率固定，不训练。"""

    def __init__(self, num_freqs: int):
        super().__init__()
        freqs = torch.pi * (2.0 ** torch.arange(num_freqs))
        self.register_buffer("freqs", freqs)  # [L]

    def forward(self, coords: Tensor) -> Tensor:
        x = coords.unsqueeze(-1) * self.freqs            # [..., C, L]
        return torch.cat([x.sin(), x.cos()], -1) \
                  .flatten(-2)                            # [..., C*2L]


# ─────────────────────────────────────────────────────────────
# 2. 几何嵌入：绝对 + 相对 → 融合 MLP
# ─────────────────────────────────────────────────────────────
class GeometryEmbedding(nn.Module):
    """
    P0: [B, N, 3]  →  e_3D: [B, N, D]

    绝对位置：这个点在首帧相机坐标系中的位置
    相对位置：这个点相对查询点集质心的位置
    融合 MLP：学习如何组合两路信息
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = 3 * 2 * cfg.num_freqs
        self.fourier  = FourierFeatures(cfg.num_freqs)
        self.mlp_abs  = make_mlp(in_dim,        cfg.geo_dim)   # 绝对 → Dg
        self.mlp_rel  = make_mlp(in_dim,        cfg.geo_dim)   # 相对 → Dg
        self.mlp_fuse = make_mlp(2 * cfg.geo_dim, cfg.hidden_dim)  # 拼接 → D

    def forward(self, P0: Tensor) -> Tensor:
        mu        = P0.mean(dim=1, keepdim=True)                # [B, 1, 3]
        abs_feat  = self.mlp_abs(self.fourier(P0))              # [B, N, Dg]
        rel_feat  = self.mlp_rel(self.fourier(P0 - mu))         # [B, N, Dg]
        fused     = torch.cat([abs_feat, rel_feat], dim=-1)     # [B, N, 2Dg]
        return self.mlp_fuse(fused)                             # [B, N, D]


# ─────────────────────────────────────────────────────────────
# 3. 条件嵌入：噪声级 σ + 模态 id
# ─────────────────────────────────────────────────────────────
class ConditionEmbedding(nn.Module):
    """σ: [B] → e_cond: [B, D]"""

    def __init__(self, cfg: Config, num_modalities: int = 2):
        super().__init__()
        self.cfg      = cfg
        self.modality = nn.Embedding(num_modalities, cfg.hidden_dim)
        self.mlp      = make_mlp(2 * cfg.num_freqs, cfg.hidden_dim)

    def forward(self, sigma: Tensor, modality_id: int) -> Tensor:
        freqs = torch.pi * (2.0 ** torch.arange(self.cfg.num_freqs, device=sigma.device))
        args  = sigma[:, None] * freqs[None, :]                 # [B, L]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)     # [B, 2L]
        mod   = torch.full_like(sigma, modality_id, dtype=torch.long)
        return self.mlp(emb) + self.modality(mod)               # [B, D]


# ─────────────────────────────────────────────────────────────
# 4. 位移与加噪
# ─────────────────────────────────────────────────────────────
def compute_displacement(positions: Tensor, scale: float) -> Tensor:
    """positions: [B, T+1, N, 3] 绝对坐标 → 归一化位移 [B, T, N, 3]"""
    flow = positions[:, 1:] - positions[:, :-1]                 # [B, T, N, 3]
    return flow / scale


def add_noise(x0: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    """在数据空间加噪：x_σ = (1-σ)·x0 + σ·ε。返回 (x_σ, ε)"""
    eps   = torch.randn_like(x0)
    sigma = sigma.view(-1, *([1] * (x0.ndim - 1)))              # [B, 1, ...]
    return (1 - sigma) * x0 + sigma * eps, eps


# ─────────────────────────────────────────────────────────────
# 5. Flow Tokenizer —— 噪声只在这里加在位移上
# ─────────────────────────────────────────────────────────────
class FlowTokenizer(nn.Module):
    """
    flow_clean: [B, T, N, 3]  归一化后的干净位移
    P0:         [B, N, 3]     初始查询点几何
    sigma:      [B]           噪声级
    返回:        [B, T*N, D]  Flow Token
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.geo  = GeometryEmbedding(cfg)
        self.proj = nn.Linear(3, cfg.hidden_dim)                # 3 → D
        self.cond = ConditionEmbedding(cfg)

    def forward(self, flow_clean: Tensor, P0: Tensor, sigma: Tensor) -> Tensor:
        # ① 噪声只加在位移上 ────────────────────────────────
        x_sigma, _ = add_noise(flow_clean, sigma)               # [B, T, N, 3]

        # ② 位移 → Token 表示
        motion = self.proj(x_sigma)                             # [B, T, N, D]

        # ③ 几何嵌入是干净的，沿时间广播 ─────────────────────
        geo = self.geo(P0).unsqueeze(1)                         # [B, 1, N, D]

        # ④ 条件嵌入（σ + 模态），也是干净的
        cond = self.cond(sigma, modality_id=0)                  # [B, D]
        cond = cond[:, None, None, :]                           # [B, 1, 1, D]

        # ⑤ 三部分相加，才成为 Token
        tokens = motion + geo + cond                            # [B, T, N, D]
        return tokens.flatten(1, 2)                             # [B, T*N, D]


# ─────────────────────────────────────────────────────────────
# 6. 视频分支：噪声加在 latent 上，不是 Token 上
# ─────────────────────────────────────────────────────────────
def patchify(z: Tensor, ph: int = 2, pw: int = 2) -> Tensor:
    """z: [B, C, T, H, W] → [B, T, H//ph, W//pw, C*ph*pw]"""
    B, C, T, H, W = z.shape
    z = z.view(B, C, T, H // ph, ph, W // pw, pw)
    z = z.permute(0, 2, 3, 5, 1, 4, 6)                          # [B,T,H',W',C,ph,pw]
    return z.reshape(B, T, H // ph, W // pw, C * ph * pw)


class VideoTokenizer(nn.Module):
    """z0: [B, C, T, H, W] → 视频 Token [B, T·H'·W', D]"""

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.latent_shape[0] * 4                        # C * 2 * 2
        self.proj = nn.Linear(in_dim, cfg.hidden_dim)

    def forward(self, z0: Tensor, sigma: Tensor) -> Tensor:
        z_sigma, _ = add_noise(z0, sigma)                       # 加在 latent 上
        patches    = patchify(z_sigma)                          # [B,T,H',W',C*4]
        B, T, H, W, C = patches.shape
        tokens = self.proj(patches).reshape(B, T * H * W, -1)   # [B, T·H'·W', D]
        return tokens


# ─────────────────────────────────────────────────────────────
# 7. 主流程：把形状变化一次看清
# ─────────────────────────────────────────────────────────────
def main() -> None:
    torch.manual_seed(0)
    cfg = Config()
    B   = 2
    N   = cfg.num_points
    T   = cfg.num_frames - 1                                     # 48

    # 输入：绝对轨迹 [B, 49, 64, 3]、初始几何 [B, 64, 3]、σ [B]
    positions = torch.randn(B, cfg.num_frames, N, 3)
    P0        = positions[:, 0]                                  # [B, 64, 3]
    sigma     = torch.tensor([0.4, 0.7])                         # [B]

    # ── Flow 分支 ─────────────────────────────────────────────
    flow_clean = compute_displacement(positions, cfg.flow_scale) # [B, 48, 64, 3]
    flow_tok   = FlowTokenizer(cfg)(flow_clean, P0, sigma)       # [B, 3072, 128]

    # ── Video 分支 ────────────────────────────────────────────
    z0        = torch.randn(B, *cfg.latent_shape)                # [B,16,13,18,30]
    video_tok = VideoTokenizer(cfg)(z0, sigma)                   # [B, 1755, 128]

    # ── 打印形状 ──────────────────────────────────────────────
    rows = [
        ("positions            ", positions.shape),
        ("P0 (初始几何)        ", P0.shape),
        ("flow_clean (归一化)  ", flow_clean.shape),
        ("x_sigma (加噪位移)   ", add_noise(flow_clean, sigma)[0].shape),
        ("geo_embed (干净)     ", GeometryEmbedding(cfg)(P0).shape),
        ("flow_tokens          ", flow_tok.shape),
        ("video_latent         ", z0.shape),
        ("video_tokens         ", video_tok.shape),
        ("joint tokens         ", flow_tok.shape[1] + video_tok.shape[1]),
    ]
    print(f"{'张量':<24}{'形状'}")
    print("─" * 60)
    for name, shape in rows:
        print(f"{name:<24}{tuple(shape)}")

    # ── 一个小数值例子：看加噪到底作用在哪 ───────────────────
    print("\n加噪示例（单个位移向量）：")
    x0  = torch.tensor([1.0, 0.0, 0.0])
    eps = torch.tensor([-1.0, 1.0, 2.0])
    s   = 0.4
    print(f"  x0      = {x0.tolist()}")
    print(f"  eps     = {eps.tolist()}")
    print(f"  σ       = {s}")
    print(f"  x_sigma = {((1 - s) * x0 + s * eps).tolist()}   ← 3 维，不是 Token")


if __name__ == "__main__":
    main()