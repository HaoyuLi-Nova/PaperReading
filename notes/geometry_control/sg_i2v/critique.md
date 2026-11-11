# SG-I2V: Self-Guided Trajectory Control in Image-to-Video Generation

- **Authors / Year / Venue**: Koichi Namekata et al. / 2024 / arXiv
- **Link**: https://arxiv.org/abs/2411.04989
- **One-line summary（一句话概括）**: 不训练新控制模块，通过优化预训练 I2V latent 并匹配轨迹框内特征实现零样本运动控制。

## 1. Motivation（动机）
- 动机：监督控制需要标注运动数据和昂贵微调；预训练 I2V 的内部表征可能已经包含可利用的运动知识。
- 是否成立 / 是否有异味（理由）：成立，但 test-time latent optimization 仍有计算成本，也可能把 latent 推离训练分布；“zero-shot”不等于零代价。

## 2. Problem（问题）
- 核心问题：给定输入图像、文本和期望轨迹，如何不改参数地控制生成视频中的区域运动。
- 普适性 & 重要性（理由）：适合快速编辑和缺乏标注数据时使用。**判断：实用问题成立，但优化稳定性与底模能力限制了大运动和复杂交互。**

## 3. Method & Core Novelty（方法与创新）
- 方法概述：把用户轨迹转成随时间移动的 bounding boxes；在预训练模型的特征上构造对齐目标，优化初始 latent；再对频率成分做后处理以降低伪影。
- 核心创新点（vs 已有工作的 delta）：把内部特征一致性作为免训练轨迹控制信号。**判断：属于对预训练先验的测试时搜索，不是新增可泛化的运动理解模块；与 Wan-Move 的训练型 latent feature guidance 属于不同成本—质量取舍。**

## 4. Related Work（相关工作）
- 关键相关工作：FreeTraj、TrailBlazer、Motion Prompting、DragNUWA、Motion-I2V。
- 本文定位与区别：免训练、逐样本优化；Wan-Move 通过训练让控制前向一次生效，追求规模化推理和视频质量。
- 缺失的重要引用（如有）：应对比优化步数、推理延迟、不同随机种子成功率和 latent 偏移，而不只比较视觉结果。

## 5. Citation / Seed Potential（高引潜力）
- 判断：中高。
- 依据：无需微调适配不同预训练 I2V，容易迁移；但每个请求优化成本和参数敏感性影响实际使用。

## 6. Future Work & Improvements（改进空间）
- 局限性：底模质量封顶；高学习率产生分布外 latent 和伪影；区域框是粗粒度监督，难表达局部形变。
- 可做的未来工作 / 研究机会：把置信度约束、局部特征复制和 latent 优化结合；研究尽可能少的测试时步数达到稳定控制。

## Overall Verdict（总评）
- 是否值得深读：值得与 FreeTraj 对读，理解免训练路线。
- 最大亮点 / 最大隐患：亮点是不更新权重也能做轨迹控制；隐患是测试时优化的稳定性、时延和底模上限。
