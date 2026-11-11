"""
flow_matching.py  (修正版)
- 修复 0 维 tensor expand bug
- t 形状在 model 内部收口
- proj_out 不再 0 初始化
- logit-normal 加 shift
- 合成数据确定性
- 若干风格修复
"""
import math
import copy
import itertools
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


# ---------------- 时间嵌入 ----------------
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / half
    ).to(t.dtype)                              # 显式 dtype 对齐
    args = torch.outer(t, freqs)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------- 时间采样 ----------------
def sample_t(B, device, dist="uniform", shift=0.05):
    if dist == "uniform":
        return torch.rand(B, device=device)
    if dist == "logit_normal":
        u = torch.sigmoid(torch.randn(B, device=device))
        return u * (1 - 2 * shift) + shift      # SD3 的 shift
    if dist == "cosine":
        u = torch.rand(B, device=device)
        return 1 - torch.cos(u * (math.pi / 2))
    raise ValueError(dist)


# ---------------- 确定性合成数据 ----------------
class SyntheticDataset(Dataset):
    def __init__(self, n=20000, size=32):
        self.n, self.size = n, size
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        self.yy, self.xx = yy, xx

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)    # 确定性
        img = torch.zeros(3, self.size, self.size)
        for _ in range(2):
            color = torch.rand(3, 1, 1, generator=g) * 2 - 1
            cx, cy = torch.rand(2, generator=g) * self.size
            r = torch.rand(1, generator=g) * 6 + 4
            mask = ((self.yy - cy) ** 2 + (self.xx - cx) ** 2) < r ** 2
            img = torch.where(mask.unsqueeze(0).expand_as(img), color.expand_as(img), img)
        return img.clamp(-1, 1)


# ---------------- DiT ----------------
def patchify(x, p):
    B, C, H, W = x.shape
    h, w = H // p, W // p
    return x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)


def unpatchify(x, p, C, H, W):
    B, N, _ = x.shape
    h, w = H // p, W // p
    return x.reshape(B, h, w, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


class TimestepEmbedder(nn.Module):
    def __init__(self, D, freqs_dim=256):
        super().__init__()
        self.freqs_dim = freqs_dim
        self.mlp = nn.Sequential(nn.Linear(freqs_dim, D), nn.SiLU(), nn.Linear(D, D))

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return self.mlp(timestep_embedding(t, self.freqs_dim))


class DiTBlock(nn.Module):
    def __init__(self, D, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(D, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_ratio * D), nn.GELU(), nn.Linear(mlp_ratio * D, D))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1.unsqueeze(1)) + s1.unsqueeze(1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + sc2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        p = cfg.patch_size
        in_ch = cfg.in_channels * p * p
        self.x_embed = nn.Linear(in_ch, cfg.D)
        self.t_embed = TimestepEmbedder(cfg.D)
        N = (cfg.image_size // p) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, N, cfg.D))
        self.blocks = nn.ModuleList([DiTBlock(cfg.D, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm_out = nn.LayerNorm(cfg.D, elementwise_affine=False)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(cfg.D, 2 * cfg.D))
        self.proj_out = nn.Linear(cfg.D, in_ch)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[-1].weight)   # 仅 gate 分支置 0
            nn.init.zeros_(blk.adaLN[-1].bias)
        nn.init.zeros_(self.adaLN_out[-1].weight)
        nn.init.zeros_(self.adaLN_out[-1].bias)
        # proj_out 保持默认初始化 —— 不再置 0

    def forward(self, x, t):
        # 形状收口：t 可以是 [] / [1] / [B]
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

        p = self.cfg.patch_size
        h = self.x_embed(patchify(x, p)) + self.pos_embed
        c = self.t_embed(t)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return unpatchify(self.proj_out(h), p, self.cfg.in_channels,
                          self.cfg.image_size, self.cfg.image_size)


# ---------------- 损失 ----------------
def fm_loss(model, x0, t_dist, t_shift):
    B = x0.shape[0]
    t = sample_t(B, x0.device, t_dist, t_shift)
    x1 = torch.randn_like(x0)
    tt = t.reshape(B, 1, 1, 1)
    xt = (1 - tt) * x0 + tt * x1
    return F.mse_loss(model(xt, t), x1 - x0)


# ---------------- 采样器：t 直接传 0 维即可 ----------------
@torch.no_grad()
def euler_sample(model, shape, device, num_steps=50):
    x = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        dt = ts[i + 1] - ts[i]
        x = x + dt * model(x, ts[i])
    return x


@torch.no_grad()
def heun_sample(model, shape, device, num_steps=25):
    x = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t
        k1 = model(x, t)
        k2 = model(x + dt * k1, t_next)
        x = x + 0.5 * dt * (k1 + k2)
    return x


@torch.no_grad()
def rk4_sample(model, shape, device, num_steps=15):
    x = torch.randn(shape, device=device)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t, dt = ts[i], ts[i + 1] - ts[i]
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x + dt * k3, t + dt)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.ema.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for eb, b in zip(self.ema.buffers(), model.buffers()):
            eb.data.copy_(b.data)


# ---------------- 可视化 ----------------
def visualize(samples, path="samples.png"):
    samples = (samples.clamp(-1, 1) + 1) / 2
    B = samples.shape[0]
    n = int(math.isqrt(B))
    m = (B + n - 1) // n
    fig, axes = plt.subplots(n, m, figsize=(m * 1.2, n * 1.2))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, ax in enumerate(axes):
        if i < B:
            ax.imshow(samples[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    print(f"已保存 {path}")


# ---------------- 主程序 ----------------
def main():
    device, torch.manual_seed(0)
    torch.manual_seed(0)
    ds = SyntheticDataset(n=20000, size=CFG.image_size)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, drop_last=True)

    model = DiT(CFG).to(device)
    ema = EMA(model, CFG.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    print(f"参数 {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"device {device} | t_sample {CFG.t_sample}")

    for step, x0 in zip(range(CFG.num_steps), itertools.cycle(dl)):
        x0 = x0.to(device)
        loss = fm_loss(model, x0, CFG.t_sample, CFG.logit_shift)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if step % 200 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    print("采样中 (Heun 25 步)...")
    ema.ema.eval()
    samples = heun_sample(ema.ema,
                          (16, CFG.in_channels, CFG.image_size, CFG.image_size),
                          device, num_steps=25)
    visualize(samples, "samples.png")


if __name__ == "__main__":
    main()