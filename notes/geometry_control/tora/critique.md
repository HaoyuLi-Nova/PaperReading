# Tora: Trajectory-oriented Diffusion Transformer for Video Generation

- **Authors / Year / Venue**: Zhenghao Zhang et al. / 2024 / arXiv
- **Link**: https://arxiv.org/abs/2407.21705
- **One-line summary（一句话概括）**: 用 3D motion VAE 把轨迹压成时空 motion patches，再由 Motion-guidance Fuser 注入视频 DiT。

## 1. Motivation（动机）
- 动机：视频 DiT 质量高但运动控制不足；轨迹比文本更明确，且需要与 DiT 的时空 token 结构对齐。
- 是否成立 / 是否有异味（理由）：成立，但“首个轨迹 DiT”是时间窗口内的定位；核心价值要看轨迹编码与大规模视频 DiT 的耦合是否优于已有控制器。

## 2. Problem（问题）
- 核心问题：在保留文本和视觉条件的同时，把用户轨迹稳定注入 DiT，并适配多分辨率、时长和宽高比。
- 普适性 & 重要性（理由）：对轨迹可控 T2V/I2V 都重要。**判断：问题实际，但模型质量与训练数据规模仍是控制效果的重要混杂变量。**

## 3. Method & Core Novelty（方法与创新）
- 方法概述：Trajectory Extractor 用 3D motion VAE 将轨迹向量编码为与视频 patch 对齐的时空表示；Motion-guidance Fuser 将其融入 DiT。训练采用稠密光流与稀疏轨迹的两阶段策略，先学一般运动再适配用户点轨迹。
- 核心创新点（vs 已有工作的 delta）：轨迹成为 DiT 的时空 patch 条件，而非 UNet 上的附加像素图。**判断：创新是面向 DiT 的轨迹表示和融合设计；它仍增加专用模块，Wan-Move 的核心主张正是绕开这类模块。**

## 4. Related Work（相关工作）
- 关键相关工作：Motion Prompting、DragNUWA、Motion-I2V、MotionCtrl、ControlNet、视频 DiT。
- 本文定位与区别：在模型架构层显式编码轨迹；Motion Prompting 用 ControlNet，Wan-Move 编辑原 I2V 的条件 latent，二者模块开销与可扩展性假设不同。
- 缺失的重要引用（如有）：与 Wan-Move 同 backbone、同数据规模、同轨迹输入的严格消融是判断融合方式优劣的关键。

## 5. Citation / Seed Potential（高引潜力）
- 判断：中高。
- 依据：公开实现、可变时长分辨率、清晰的 DiT 轨迹接口；复现成本仍受视频训练资源影响。

## 6. Future Work & Improvements（改进空间）
- 局限性：额外 VAE/fuser 带来参数与训练复杂度；两阶段训练与数据流可能影响稀疏轨迹泛化；轨迹之外的物理约束未被显式建模。
- 可做的未来工作 / 研究机会：在同一底模上比较随机 embedding、motion patches、latent feature propagation；检查密集训练是否造成稀疏推理域偏移。

## Overall Verdict（总评）
- 是否值得深读：值得，作为 Wan-Move 的直接架构对照。
- 最大亮点 / 最大隐患：亮点是将轨迹结构化对齐到 DiT token；隐患是新增组件和训练配方可能成为主要收益来源，跨模型可迁移性需核实。
