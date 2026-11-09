# Wan 训练代码阅读路线

先给结论：`PaperReading/code/base_models/Wan2.1/` 和 `Wan2.2/` 是官方推理代码，没有公开的预训练或微调训练循环。这里能直接学习的“训练”有两层：

1. [`wan_i2v_train_step.py`](wan_i2v_train_step.py) 把论文的 Flow Matching 公式和官方 I2V 前向拼成一个单样本训练步骤，用很小的张量验证形状与方向。
2. 工作区里的 `projects/RayLift/v0/src/training/` 是真实的 Wan2.2-TI2V-5B 训练适配，包含 flow schedule、loss、backward、AdamW、EMA 和 checkpoint；但它训练的是 RayLift 的选择性参数面，不是 Wan 原始全量预训练。

因此不要一开始就在官方仓库里找 `train.py`。正确顺序是：先懂一次训练 step，再懂官方 DiT 的一次 forward，最后才看 RayLift 的外层 trainer。

## 1. 本仓库代码地图

本文件内的链接相对 `code/mycode/wan2.1/`；`projects/...` 路径相对工作区根目录。

| 想弄懂什么 | 先读什么 |
| --- | --- |
| Wan2.1 I2V 如何造条件 | `../../base_models/Wan2.1/wan/image2video.py:210-249` |
| Wan2.1 DiT 如何前向 | `../../base_models/Wan2.1/wan/modules/model.py:493-582` |
| 一个 Transformer block 做什么 | `../../base_models/Wan2.1/wan/modules/model.py:238-317` |
| 输出如何还原成 latent | `../../base_models/Wan2.1/wan/modules/model.py:320-347, 584-608` |
| 官方如何采样 | `../../base_models/Wan2.1/wan/text2video.py`、`image2video.py` 中的 `generate` |
| Wan2.2 TI2V-5B 如何锚定首帧 | `../../base_models/Wan2.2/wan/textimage2video.py:512-600` |
| Wan2.2 A14B 如何切换两个专家 | `../../base_models/Wan2.2/wan/text2video.py:101-117, 169-191, 335-360` |
| 最小训练一步 | [`wan_i2v_train_step.py`](wan_i2v_train_step.py) |
| 真实 5B flow schedule | `projects/RayLift/v0/src/models/wan/flow_match.py` |
| 真实 loss | `projects/RayLift/v0/src/training/objective.py:86-150` |
| 真实优化循环 | `projects/RayLift/v0/src/training/engine.py:538-592` |

先运行最小脚本：

```bash
cd /mnt/vision-gen-ks3/Video_Generation/lihaoyu/PaperReading
/root/.conda/envs/RayLift/bin/python code/mycode/wan2.1/wan_i2v_train_step.py
```

当前已跑通。核心输出是：像素 `[3,9,16,16]` 经假 VAE 得到 `[16,3,2,2]`，条件 `mask[4] + z_c[16]` 与 `X_t[16]` 拼成 36 通道，模型只输出 16 通道速度。

## 2. 先分清论文方向和代码方向

论文写法：

$$
x_t=t x_1+(1-t)x_0,\qquad
x_1=\text{干净 latent},\quad x_0=\text{高斯噪声}
$$

所以 `t=1` 是干净数据，`t=0` 是噪声，目标速度是：

$$
v=x_1-x_0
$$

Wan 官方仓库的推理 scheduler（不是训练循环）和 RayLift 的训练实现都写作 `sigma`，它只是把论文时间轴反过来了：

$$
\operatorname{sigma}=1-t,\qquad
z_{\operatorname{sigma}}=(1-\operatorname{sigma})x_1+\operatorname{sigma}x_0
$$

于是 `sigma=1` 是噪声，`sigma=0` 是干净数据。scheduler 的更新方向对应目标：

$$
u=x_0-x_1=-v
$$

`RayLift/v0/src/models/wan/flow_match.py` 完全采用这个约定：

```python
sample = (1 - sigma) * clean + sigma * noise
target = noise - clean
prev_sample = sample + model_output * (sigma_next - sigma)
```

所以读代码时要固定一条规则：

- 论文实验：`t: 0 -> 1`，模型预测 `x1-x0`。
- Wan 推理 scheduler / RayLift：`sigma: 1 -> 0`，模型预测 `noise-clean`。
- 两者符号相反、时间轴相反，但描述的是同一条直线路径。

这是第一次对照论文和代码时最容易卡住的地方。

## 3. 一个训练 step 到底做什么

训练时每个样本只抽 **一个时间步**，没有推理时的多步去噪循环。伪代码如下：

```python
clean = vae.encode(video)                 # x1
noise = torch.randn_like(clean)           # x0
sigma = sample_one_train_sigma()          # paper 中 t 的镜像
noisy = (1 - sigma) * clean + sigma * noise
target = noise - clean
pred = dit(noisy, sigma, text_condition)
loss = mse(pred, target)
loss.backward()
```

[`wan_i2v_train_step.py`](wan_i2v_train_step.py) 对应的关键行：

- `causal_vae_encode()`：用平均池化伪造 Wan-VAE，只模拟“首帧不做时间压缩、后续 4 帧压成 1 格、输出 16 通道”。
- `pack_frame_mask()`：把像素帧的 keep 标记打包成每格 4 通道。
- `make_condition()`：造 `y = cat(mask, z_c)`。
- `train_one_sample()`：造 `X_t`、目标速度、36 通道输入和 MSE。
- `infer_loop()`：只用于对照推理；训练永远不会进入这个循环。

这个脚本里的 VAE 和 DiT 都是假的，不能用于质量或收敛结论。它的价值是让你在没有 14B 权重和 GPU 的情况下看懂数据流。

可以马上做的两个练习：

```python
# 练习 1：看不同 t 下 noisy 如何从 clean 走向 noise
out0 = train_one_sample(dit, video, prompt_emb, clip_fea, i2v_keep(), t=torch.tensor([0.0]))
out5 = train_one_sample(dit, video, prompt_emb, clip_fea, i2v_keep(), t=torch.tensor([0.5]))
out9 = train_one_sample(dit, video, prompt_emb, clip_fea, i2v_keep(), t=torch.tensor([0.9]))

# 练习 2：确认 loss 真的能反传
out = train_one_sample(dit, video, prompt_emb, clip_fea, i2v_keep(), t=torch.tensor([0.3]))
out["loss"].backward()
print(dit.patch.weight.grad.abs().mean())
```

## 4. Wan2.1 I2V 条件如何进入模型

官方 [`image2video.py`](../../base_models/Wan2.1/wan/image2video.py) 做两件不同的事：

```text
条件视频 = [首帧图像, 0, 0, ..., 0]
z_c       = VAE(条件视频)
mask      = 哪些像素帧真实存在
y         = cat(mask, z_c)
```

当前 81 帧例子中：

```text
像素帧            81
latent 时间帧    1 + (81-1)/4 = 21
mask              4 x 21 x H/8 x W/8
z_c              16 x 21 x H/8 x W/8
y                20 x 21 x H/8 x W/8
X_t              16 x 21 x H/8 x W/8
模型输入          36 x 21 x H/8 x W/8
```

通道顺序必须是：

```text
[X_t | mask | z_c] = [16 | 4 | 16]
```

这个顺序由 `image2video.py:240-249` 和 `model.py:530-532` 共同决定。加载官方权重时不能改成 `[X_t | z_c | mask]`。

注意：`mycode/wan2.1/wan_i2v_dit_input.py` 是一张早期形状草稿，它写成了 `[X_t | z_c | mask]`，与官方前向的顺序不一致。学习真实权重路径时以 `wan_i2v_train_step.py` 和官方 `image2video.py` 为准。

CLIP 与 T5 不进 36 通道：

```text
CLIP(first frame) -> 257 x 1280 -> MLP -> D
T5(prompt)        -> up to 512 x 4096 -> MLP -> D
```

`WanI2VCrossAttention` 用视频 token 当 Q，分别对 CLIP 和 T5 做 cross-attention，再把两路输出相加。因此“图像身份”和“文本语义”都是 side context，不会被拼进 latent 的通道维。

## 5. 官方 DiT 一次 forward

以 Wan2.1 I2V 为例，[`model.py:493-582`](../../base_models/Wan2.1/wan/modules/model.py) 的顺序是：

1. `cat(X_t, y)`，通道从 16 变成 36。
2. 3D Conv patch embedding：`(1,2,2)` 的 kernel 和 stride，把 latent 变成 `[L,D]` token 序列。
3. T5/CLIP 条件分别投影到隐维 `D`。
4. timestep 经 sinusoidal embedding 和 MLP，再投影成 6 组 AdaLN 参数。
5. 重复 `WanAttentionBlock`：Self-Attn -> Text/Image Cross-Attn -> FFN。
6. `Head` 把每个 token 投回 `patch_volume * 16`，再 unpatchify 成 `[16,F,H,W]` 速度。

每个 block 的代码顺序见 `model.py:278-317`。核心是：

```text
AdaLN 调制的 Self-Attention（带 3D RoPE）
  -> Text/Image Cross-Attention
  -> AdaLN 调制的 FFN
```

6 个 AdaLN 参数分别是 self-attention 和 FFN 的 shift、scale、gate。所有 block 共享 time embedding/projection，但每个 block 有自己的 `modulation` bias，这是论文所说“共享 MLP + 每层独立 bias”的实现。

3D RoPE 初始化在 `model.py:397-405`：head 维被切成时间、height、width 三段，因此同一个 attention 能同时感知帧位置和空间位置。

`Head` 位于 `model.py:320-347`，最后 `unpatchify()` 位于 `model.py:584-608`。模型输出始终是 16 通道 latent 速度，不是 36 通道，也不是最终像素。

## 6. Wan2.2 要单独看的两条分支

当前 RayLift 使用 Wan2.2-TI2V-5B，不能把 Wan2.1 I2V 的 36 通道结论直接搬过来。

### 6.1 TI2V-5B

5B 使用高压缩 VAE：时间 4 倍、空间 16 倍、48 通道，再叠加 `(1,2,2)` patch。首帧是干净锚点，后续帧加噪；官方 [`textimage2video.py:548-600`](../../base_models/Wan2.2/wan/textimage2video.py) 用 `mask2` 做两件事：

- 把首帧 condition latent 写回当前 latent。
- 让首帧 token 的 timestep 为 0，其余 token 使用当前采样 timestep。

Wan2.2 模型把 timestep embedding 做成逐 token 的 `[B,L,6,D]`，见 [`Wan2.2/wan/modules/model.py:459-468`](../../base_models/Wan2.2/wan/modules/model.py)。这不是额外 CLIP 分支，而是 5B 把图像条件融合进 latent、再用 token timestep 保持首帧干净。

RayLift 的训练实现与这一语义一致：

```text
latents = add_noise(clean, noise, sigma)
latents[:, :, :1] = first_frame_latents
loss 只对 latent frame 1: 计算
首帧 token 的 modulation time = 0
后续 token 的 modulation time = 当前 sigma
```

对应代码是 `projects/RayLift/v0/src/training/objective.py:116-138` 和 `projects/RayLift/v0/src/models/modulation.py`。

### 6.2 A14B 的 MoE

Wan2.2 A14B 的 MoE 不是每个 token 动态选专家，而是按去噪时间切换两个完整 DiT：

```text
t >= boundary * num_train_timesteps -> high_noise_model
t <  boundary * num_train_timesteps -> low_noise_model
```

推理切换见 `Wan2.2/wan/text2video.py:169-191`。前段高噪声专家负责布局和运动，后段低噪声专家负责细节；每步只跑一个约 14B 的专家。公开仓库仍然没有给出这两个专家的训练循环或数据配方。

## 7. 从教学 step 接到 RayLift 真实训练

`PaperReading` 里的最小脚本只做一次前向和 loss。真实训练外壳在 RayLift：

1. `projects/RayLift/v0/src/models/wan/flow_match.py` 提供 `add_noise`、`training_target`、`step` 和 loss weight。
2. `projects/RayLift/v0/src/training/objective.py:86-150` 采样噪声与时间步、构造 `X_t`、固定首帧、调用 `joint_forward`，再算 MSE。
3. `projects/RayLift/v0/src/models/forward.py:67-143` 做 patchify、文本嵌入、3D RoPE、逐 token 时间调制和全部 DiT block。
4. `projects/RayLift/v0/src/training/engine.py:547-592` 做 autocast、backward、梯度裁剪、AdamW、EMA、保存和自动验证。

RayLift 的 VAE 文档明确说明训练 step 读取预先算好的 latent，不经过 VAE。这是工业训练常见做法：VAE 和文本编码器离线或前置处理，GPU 训练循环只保留 DiT 和可训练模块。

第一次读 `objective.py` 时请忽略 `trajectory`、`camera`、`depth_*` 这些 RayLift 专有分支，只跟踪 `video_noise`、`timestep`、`latents`、`video_target`、`video_loss` 五个变量。看懂主线后再回来看附加条件。

还要区分“能否训练”和“原始 Wan 怎么预训练”：RayLift 的 `builder.py:110-167` 会冻结大部分 Wan，只开放 self-attention Q/K/V/O 和 RayLift 新模块。这是针对项目的选择性微调，不是 Wan 论文的大规模全量预训练配方。

## 8. 推荐的四步学习法

### 第一步：只学一次 I2V step

运行教学脚本，能不看代码画出下面这条链即可：

```text
video -> VAE(x1) -> X_t <- noise/t
first frame -> z_c, mask -> y
X_t + y -> 36ch -> DiT -> 16ch velocity -> MSE
```

完成标准：能解释为什么训练没有 `for timestep in timesteps`，而推理有。

### 第二步：在官方 forward 里放断点

优先看三个位置：

1. `image2video.py` 的 `y = torch.concat([msk, y])`。
2. `model.py` 的 `x = [torch.cat([u, v], dim=0) ...]`。
3. `WanAttentionBlock.forward()` 的 self-attn、cross-attn、FFN 三段。

每次只打形状，不打印大张量。先确认 `[C,F,H,W] -> [L,D] -> [C,F,H,W]`，再看数值。

### 第三步：把论文公式映射到 RayLift

在 `flow_match.py` 同时写出两套记号：

```text
paper: t=1 clean, t=0 noise, target = clean-noise
code:  sigma=1 noise, sigma=0 clean, target = noise-clean
```

完成标准：能手算一个 `sigma=0.7` 的 `latents` 和 target，并解释 scheduler 为什么用 `sigma_next - sigma` 更新。

### 第四步：再选训练方式

- 想理解 Wan 本身：完成上面三步即可，不必先跑 14B。
- 想做标准 Wan LoRA/全量微调：官方 README 指向 DiffSynth-Studio，它明确支持 Wan2.1/Wan2.2 的 LoRA training 和 full training。先用 1.3B 或单样本配置跑通，再扩到 14B/5B。
- 想改 RayLift：先跑 `bash train.sh configs/train/stage1.yaml -- --check_only`，再在协议确认后跑 2-step smoke；不要直接把正式长训当作学习实验。

## 9. 当前代码里的坑

1. **不要在官方仓库找 train loop**：两个官方 checkout 都只有推理入口。
2. **不要混用时间方向**：论文 `t` 与 Wan 推理 scheduler / RayLift 的 `sigma` 反向，速度符号也反向。
3. **不要在训练时跑去噪循环**：训练每个样本只抽一个时间步。
4. **不要混用 Wan2.1 I2V 和 Wan2.2 TI2V-5B 的输入**：前者 36 通道 `[X|mask|z_c]`，后者 48 通道融合 latent + token timestep；5B 没有 CLIP cross-attention。
5. **不要改通道顺序**：`wan_i2v_dit_input.py` 的草稿顺序与官方权重不兼容。
6. **不要拿假 VAE 的数值做结论**：教学脚本只验证形状和计算方向。
7. **不要把 A14B MoE 理解成 token router**：它是按时间切换两个完整专家。

## 10. 学完后的自检

能独立回答下面五题，基础训练代码就算过关：

1. 为什么训练时一个样本只有一个 timestep，而推理要循环几十步？
2. Wan2.1 I2V 的 36 通道分别是什么，为什么 CLIP/T5 不在里面？
3. 论文的 `x1-x0` 和官方的 `noise-clean` 为什么可以同时成立？
4. Wan2.2 TI2V-5B 为什么要把首帧 token 的 timestep 设为 0？
5. RayLift 的 `objective.py` 中哪一行产生训练输入，哪一行产生监督目标，哪一行执行真正的 DiT 前向？

