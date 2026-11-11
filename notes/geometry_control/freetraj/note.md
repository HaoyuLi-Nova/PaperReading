# FreeTraj 速记

## 问题与动机
控制型生成常需训练专用 adapter；FreeTraj 研究能否利用采样噪声和 attention 直接引导预训练模型。

## 方法主线
通过 noise guidance 调整不同帧初始噪声的相关性以影响对象运动，并用 attention isolation 减少目标与背景干扰；进一步设计长视频噪声策略。

## 读后要点
- 它是免训练推理控制，不依赖监督学会点轨迹语义；输入到输出的映射受底模内部先验约束。
- 对照 SG-I2V：FreeTraj 改噪声与注意力；SG-I2V 更偏 latent/特征优化。
- 延伸阅读：[FreeTraj 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
