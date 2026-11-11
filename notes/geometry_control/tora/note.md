# Tora 速记

## 问题与动机
DiT 视频生成的时空表达能力强，但轨迹条件必须与视频 token 空间对齐，才能保持扩展性。

## 方法主线
Trajectory Extractor 以 3D motion VAE 把轨迹压成时空 motion patches；Motion-guidance Fuser 将这些 patches 融入 DiT。训练用稠密运动和稀疏轨迹的两阶段方案。

## 读后要点
- Tora 是“专用轨迹编码器 + DiT 融合”的代表。
- 与 Wan-Move 对照时，应关注新增模块是否换来更强控制，以及相同数据/预算下的表现。
- 论文称只用稀疏或只用稠密轨迹都不理想，混合训练有助于兼顾学习信号和用户输入。
- 延伸阅读：[Tora 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
