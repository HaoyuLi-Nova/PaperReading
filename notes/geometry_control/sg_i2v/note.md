# SG-I2V 速记

## 问题与动机
监督轨迹控制需要视频标注和昂贵微调；SG-I2V 检验预训练 I2V 模型能否通过测试时引导而无需更新参数。

## 方法主线
把期望轨迹转成时变区域框，在模型特征空间构造跨帧对齐目标，优化输入 latent；再用频率处理降低伪影。

## 读后要点
- Zero-shot 描述的是不训练新权重，不代表无需额外推理时间。
- 与 FreeTraj 一样依赖模型内部先验，但 SG-I2V 主要通过 latent 优化/特征匹配控制。
- Wan-Move 把控制学进模型，换取标准推理流程中的较低额外控制成本。
- 延伸阅读：[SG-I2V 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
