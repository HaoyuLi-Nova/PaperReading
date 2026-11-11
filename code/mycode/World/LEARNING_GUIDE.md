# 从零手写 Video DiT + Flow Matching：逐行练习指南

## 0. 本代码究竟实现了什么？

- **输入**：合成的彩色圆形运动视频，格式 `[B,3,T,32,32]`，归一化到 `[-1,1]`。
- **数据建模空间**：像素空间，不是 Wan 的 VAE latent 空间。
- **网络**：Patchify → Linear → 可学习时空位置向量 → 多个 adaLN-Zero DiT Block → 输出头 → Unpatchify。
- **目标**：Rectified Flow，预测噪声方向速度 `v=eps-x0`。
- **推理**：从 `t=1` 的高斯噪声，用 Euler 或 Heun 沿反向时间积分到 `t=0`。
- **结果**：多个视频帧的拼接 PNG 和 GIF。
- **尚未实现**：VAE、条件图像 I2V、文本 Cross-Attention、3D RoPE、Wan2.2 训练/采样细节。

## 1. 用一组确定的维度跟踪每行代码

默认配置：`B=16,C=3,T=8,H=W=32,pt=2,ps=4,D=128`。

| 步骤                  | Shape              | 意义                                     |
| --------------------- | ------------------ | ---------------------------------------- |
| `clean` / `noise` | `[16,3,8,32,32]` | 视频像素与随机噪声                       |
| `t`                 | `[16]`           | 每个视频独立采样的时刻                   |
| `t_view`            | `[16,1,1,1,1]`   | 用于广播                                 |
| `x_t`               | `[16,3,8,32,32]` | 插值轨迹上的中间状态                     |
| `patchify(x_t)`     | `[16,256,96]`    | 256 个 patch，每个有 `3*2*4*4=96` 个值 |
| `x_embed(...)`      | `[16,256,128]`   | 视频 tokens                              |
| `sincos_t(t)`       | `[16,256]`       | 时间步正余弦特征                         |
| `t_embed(...)`      | `[16,128]`       | 时间条件 `c`                           |
| `adaLN(c)`          | `[16,768]`       | 6 个 128 维调制向量                      |
| DiT block 输出        | `[16,256,128]`   | 保持形状                                 |
| `out_proj`          | `[16,256,96]`    | 每个 patch 预测 96 维速度                |
| `unpatchify`        | `[16,3,8,32,32]` | 像素速度场                               |

检查：`N=(T/pt)*(H/ps)*(W/ps)=4*8*8=256`。T=1 时自动将 `pt=1`，因此可以作为二维模型练习。

## 2. 为什么六个 adaLN 参数？

`(shift1, scale1, gate1, shift2, scale2, gate2) = adaLN(c).chunk(6,-1)`

对每个 token `x`，第一分支：

```
h = norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
x = x + gate1[:, None] * SelfAttention(h)
```

第二分支：

```
h = norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
x = x + gate2[:, None] * MLP(h)
```

Six vectors = each residual branch 3 vectors (shift / scale / gate) × 2 branches.

`adaLN` 末端线性层全部清零，使每个 block 初始是恒等映射（`gate=0`）。**它不意味着全部参数从第一个 batch 就收到非零梯度**，gate 首先学习打开各分支，随后信息逐步传到 Attention 和 MLP。这是正常现象。

## 3. Rectified Flow 的推导

`x_0` 是干净视频，`eps ~ N(0,I)`，训练用的线性路径：

`x_t = (1-t)*x_0 + t*eps`

沿路径求导：`dx_t/dt = eps - x_0`。

所以损失：`MSE(model(x_t,t), eps-x_0)`。

采样从 `x(1)~N(0,I)` 开始向 `t=0` 积分：

`x(t_next) = x(t) + (t_next-t)*v_theta(x(t),t)`。

因为 `t_next < t`，步长为负。Heun 在预测值上再计算一次速度并取均值，通常比相同步数下的 Euler 有更高数值精度，但每步做 2 次模型前向。

这个训练目标称 conditional flow matching / rectified-flow 风格的简单直线路径形式；不要将它等同于 Wan 全部训练策略。

## 4. 时间采样与 shift 的区别

- `uniform`：`U(0,1)`。
- `logit_normal`：`sigmoid(N(0,1))`，集中在时间中间区域。
- `cosine`：对均匀变量做 `1-cos(pi*u/2)` 的非线性变换。
- `time_shift=s`：映射 `s*t/(1+(s-1)*t)`，当 `s=1` 不变；大于 1 将更多概率质量移向高噪声时间。

这是一个明确写出来、便于学习的 rational shift；不同模型/框架对 shift 的参数定义、配置及实现并不必然相同。

## 5. 为什么数据集要确定性？

`__getitem__(idx)` 使用种子 `seed+idx`，同一个 idx 始终生成同一段视频：这样能排查模型、比较修改效果。不同 idx 的圆形颜色、位置和速度不同。

训练仍包含 DataLoader shuffle、每步新噪声和时间采样，因此它不是相同视频、相同噪声的完全固定训练过程。

## 6. 手写练习：推荐按这个顺序

1. **只写 `patchify` / `unpatchify`**：输入 `[2,3,4,16,16]`，证明像素逐项完全恢复。
2. **只写 `sincos_t` / `sample_time`**：计算时间特征，绘制三种时间采样的直方图。
3. **手写 `DiTBlock`**：先使用 `nn.MultiheadAttention`，然后用 `nn.Linear(dim,3*dim)` + `F.scaled_dot_product_attention` 实现 QKV。
4. **手写完整 `VideoDiT`**：不训练，只跑前向并打印每步 shape。
5. **手写 `flow_loss`**：固定模型用随机数据验证梯度。
6. **手写 Euler/Heun**：先用一个已知解析 ODE 验证数值积分，再用生成网络。
7. **手写 MovingCircles 数据集与训练循环**：检查 loss 的整体趋势，查看生成 GIF。
8. **移除学习用简化模块**：加 3D RoPE、Cross-Attention、VAE latent，以及首帧条件。

## 7. 初学时最有价值的 5 个实验

- 保持 seed 不变，比较 `T=1` 与 `T=8`，理解 token 数变化。
- 把 `pt=2` 改为 `pt=1`，计算 token 数与 Attention 复杂度增加多少。
- 将 `adaLN` 的 6D 调制结果逐个打印均值，检查 gate 如何从零开始更新。
- 固定同一随机噪声，用 Euler/Heun 相同 NFE（模型前向次数）比较，而非仅比较步数。
- 将固定时间采样从 uniform 改为 logit-normal，观察 loss 和生成结果差异，注意这不是严格可比实验，最好固定其他超参。

## 8. 通向 Wan2.2 的升级路径

这份文件 **不是** Wan2.2 的复现。要逐步向真实 I2V 系统靠近：

1. 将视频像素 `x_0` 改为 VAE latent；记录 VAE 时间/空间下采样形状。
2. 用时空 3D RoPE 取代固定大小位置参数。需要明确时间、宽、高对应轴。
3. 将纯噪声生成改为首帧条件的 I2V，定义 clean first-frame 和 noisy latents 如何构造。
4. 加入 text embeddings 的 cross-attention，说明文本和视频各自 token 数。
5. 最后才加入独立 trajectory tokens / branch 和联合注意力机制。

**限制**：这是基于小视频的教学工程，不保证 `--steps 2000` 就得到高质量视频；学习目标首先是正确的 shape、数学定义、梯度和完整流水线。
