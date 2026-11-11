# Wan-Move 相关论文阅读路线：点轨迹、跟踪与运动控制

## 范围

这份清单从 Wan-Move 的参考文献中筛出与其核心主张直接相关的论文：点轨迹如何获得，轨迹怎样作为控制条件，如何在 I2V/视频扩散中注入，以及相机、物体和深度运动如何区分。没有把视频生成基座、通用扩散方法、数据集背景和评测指标论文全部纳入；那些不直接回答上述问题的引用仍保留在 Wan-Move 原文参考文献中。

Wan-Move 中“点轨迹的论文”可能指两个不同环节：**Motion Prompting** 是直接用点轨迹控制生成的前驱；**CoTracker** 是 Wan-Move 数据标注和推理所用的点追踪器。两篇都值得先读。

## 阅读顺序

1. **输入轨迹从哪里来**： [CoTracker](../../notes/geometry_control/cotracker/critique.md) → [SpatialTracker](../../notes/geometry_control/spatialtracker/critique.md，可选，补 3D 跟踪视角)
2. **生成模型怎样消费轨迹**： [DragNUWA](../../notes/geometry_control/drag_nuwa/critique.md) → [Motion Prompting](../../notes/geometry_control/motion_prompting/critique.md) → [Tora](../../notes/geometry_control/tora/critique.md) → [Motion-I2V](../../notes/geometry_control/motion_i2v/critique.md) → Wan-Move
3. **不训练新控制器的路线**： [TrailBlazer](../../notes/geometry_control/trailblazer/critique.md) → [FreeTraj](../../notes/geometry_control/freetraj/critique.md) → [SG-I2V](../../notes/geometry_control/sg_i2v/critique.md)
4. **相邻控制设计和评测**： [MotionCtrl](../../notes/geometry_control/motionctrl/critique.md) → [LeviTor](../../notes/geometry_control/levitor/critique.md) → [Image Conductor](../../notes/geometry_control/imageconductor/critique.md) → [MagicMotion](../../notes/geometry_control/magicmotion/critique.md)

## 方法脉络

| 工作 | 输入/表示 | 注入方式 | 训练范式 | 对 Wan-Move 的学习价值 |
|---|---|---|---|---|
| DragNUWA | 稀疏/稠密轨迹、图像、文本 | 多尺度轨迹融合 | 训练型 | 早期开放域轨迹控制，理解稀疏到稠密训练动机 |
| Motion Prompting | 任意时空稀疏度的点轨迹 | 随机轨迹 embedding + ControlNet | 训练型 | 最直接前驱；暴露像素级轨迹缺少邻域语义的问题 |
| Tora | 轨迹时空 token | 3D motion VAE + DiT fuser | 训练型 | 对照“新增轨迹模块”与 Wan-Move 的原条件编辑 |
| Motion-I2V | 参考图像像素运动场 | motion-warp 特征增强时序注意力 | 两阶段训练 | 与沿轨迹传播首帧特征的思路接近，但使用预测稠密运动和改造网络 |
| CoTracker | 查询点及视频 | 联合跨点追踪 | 监督跟踪器 | Wan-Move 轨迹质量的上游来源；需关注遮挡和漂移误差 |
| SpatialTracker | 2D 查询点 + 单目深度 | 3D triplane 追踪与几何约束 | 监督跟踪器 | 判断出平面运动是否值得引入显式 3D |
| TrailBlazer / FreeTraj / SG-I2V | 框、噪声、模型内部特征 | 注意力/噪声编辑或 latent 优化 | 免微调推理 | 训练成本与控制质量的另一端；“免训练”仍有推理成本和底模限制 |
| MotionCtrl / LeviTor / Image Conductor | 相机 pose、点/深度/实例轨迹 | 控制分支、ControlNet 或 LoRA | 训练型 | 分别学习镜头/物体解耦、3D 深度与相机自由 guidance |
| MagicMotion | mask、框、稀疏框轨迹 | 多粒度条件控制 | 训练型 | 对照输入粒度、数据规模和专用 benchmark |

## 对照 Wan-Move 时要盯住的三个问题

1. **轨迹条件的上下文**：单点轨迹只告诉模型一个像素去哪；Motion Prompting 用随机 embedding，Wan-Move 则复制轨迹起点的首帧 latent 特征。要检验改善是否来自邻域语义、latent 尺度还是训练数据。
2. **轨迹误差如何传播**：CoTracker 错跟、遮挡后不可见和轨迹断裂会变成错误条件。生成评测应加入人工轨迹、扰动轨迹及可见性标注，区分生成器失败和 tracker 失败。
3. **实验比较是否公平**：MotionCtrl、Tora、Image Conductor、MagicMotion 和 Wan-Move 的底模、数据、分辨率、帧长、训练预算及输入形式不同。跨论文结果可定位路线，不足以证明某个模块在同等条件下更优。

## 本地文件

13 篇论文的 PDF 与 arXiv TeX 源码分别位于 `arxiv/geometry_control/<slug>/`，大文件按仓库规则由 gitignore 忽略；每篇六维精读在 `notes/geometry_control/<slug>/critique.md`。论文元数据和路径已加入 `papers.yaml`。
