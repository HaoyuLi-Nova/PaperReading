# DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory

- **Authors / Year / Venue**: Shengming Yin et al. / 2023 / arXiv
- **Link**: https://arxiv.org/abs/2308.08089
- **One-line summary（一句话概括）**: 将文本、首帧和任意轨迹结合，以轨迹采样、多尺度融合和自适应训练实现开放域视频运动控制。

## 1. Motivation（动机）
- 动机：早期视频生成控制往往只支持文本、图像或轨迹其中一种，且轨迹研究局限于简单数据和简单路径。
- 是否成立 / 是否有异味（理由）：成立，尤其强调开放域与曲线运动；但多个条件联合会让每项贡献难以拆分，需看消融是否证明真正来自轨迹机制。

## 2. Problem（问题）
- 核心问题：在开放域视频生成中同时满足语义、外观和任意运动路径控制。
- 普适性 & 重要性（理由）：是后续轨迹控制工作的直接问题设置。**判断：重要且具有开创性，但模型代际和数据规模较早，绝对画质不能与 Wan 系模型直接比较。**

## 3. Method & Core Novelty（方法与创新）
- 方法概述：Trajectory Sampler 处理用户任意路径；Multiscale Fusion 提供不同粒度的轨迹控制；Adaptive Training 提高按轨迹运动的一致性；联合图像、文本条件进行生成。
- 核心创新点（vs 已有工作的 delta）：把轨迹路径和多模态条件集成进开放域视频扩散流程。**判断：是点轨迹条件生成的重要早期系统；相较 Motion Prompting 和 Wan-Move，轨迹表示和控制融合仍有专门模块。**

## 4. Related Work（相关工作）
- 关键相关工作：早期基于 Human3.6M 的轨迹生成、Motion Prompting、Tora、Motion-I2V、Wan-Move。
- 本文定位与区别：在较早的视频扩散模型上完成开放域轨迹控制；Wan-Move 的贡献是将特征级控制做得更直接并扩展到 Wan-I2V。
- 缺失的重要引用（如有）：当前比较更应使用统一底模复现，而非跨代模型直接按视觉质量排序。

## 5. Citation / Seed Potential（高引潜力）
- 判断：高 seed 潜力。
- 依据：较早提出开放域文本/图像/轨迹联合控制问题，后续工作持续沿用轨迹条件；受制于底模迭代速度。

## 6. Future Work & Improvements（改进空间）
- 局限性：控制信号编码/融合有额外设计，路径复杂时控制精度、视频一致性和语义保持彼此牵制；较早底模限制视觉效果。
- 可做的未来工作 / 研究机会：在现代 I2V backbone 上复现其多尺度轨迹表示，与 latent feature propagation 做受控消融；研究控制区域与非目标区域的副作用。

## Overall Verdict（总评）
- 是否值得深读：建议理解其轨迹控制问题定义和训练策略，实验细节可选择性深读。
- 最大亮点 / 最大隐患：亮点是将任意轨迹带入开放域生成；隐患是早期模型能力和多模块组合让效果归因困难。
