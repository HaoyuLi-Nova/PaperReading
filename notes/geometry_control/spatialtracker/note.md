# SpatialTracker 速记

## 问题与动机
2D 轨迹在遮挡、投影和出平面转动下不连续；作者认为在 3D 中跟踪能更贴近真实运动结构。

## 方法主线
使用单目深度把像素提升到 3D，用 triplane 表示场景；Transformer 迭代估计 3D 轨迹，并用 ARAP 几何约束和 rigidity embedding 建模刚性部件。

## 读后要点
- 相比 CoTracker，SpatialTracker 直接处理 3D 运动，但多依赖深度估计且假设部分局部刚性。
- Wan-Move 当前使用 2D latent 坐标；3D tracker 更可能帮助相机运动和出平面控制，不一定普遍更好。
- 延伸阅读：[SpatialTracker 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
