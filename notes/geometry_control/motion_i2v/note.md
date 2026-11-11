# Motion-I2V 速记

## 问题与动机
直接从单图学习视频，容易在大运动下丢失首帧外观；显式预测运动再生成可以把运动与外观传播拆开。

## 方法主线
第一阶段预测参考图各像素到未来帧的运动场，并允许稀疏轨迹 ControlNet 控制；第二阶段在 latent temporal attention 中加入 motion-augmented feature warping，传播参考图特征。

## 读后要点
- 与 Wan-Move 有共同直觉：沿运动传播起始图像信息。
- 关键差异：Motion-I2V 预测稠密运动场且改网络；Wan-Move 接收用户点轨迹、映射到 latent 并复制轨迹起点特征。
- 两阶段结构的风险是运动预测错误级联到生成质量。
- 延伸阅读：[Motion-I2V 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
