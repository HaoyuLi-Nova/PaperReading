# CoTracker 速记

## 问题与动机
逐点独立追踪忽视同一物体内点的相关性；联合点信息能帮助遮挡恢复和大规模轨迹估计。

## 方法主线
模型在时间窗口内联合更新查询点轨迹，用跨轨迹注意力交换信息；virtual tracks 作为代理降低全量点间注意力成本，窗口状态递归传递到长视频。

## 读后要点
- Wan-Move 的轨迹条件可以由 CoTracker 等 tracker 获得，跟踪器质量直接影响生成控制上限。
- 论文指出合成训练数据在阴影、反射等真实场景可能产生泛化问题；“阴影跟着物体”可能是外观编辑所需，也可能是物理运动分析中的错误。
- 下游评测应报告轨迹遮挡/置信度，并把 tracker 误差和生成器误差分开。
- 延伸阅读：[CoTracker 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
