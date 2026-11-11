# DragNUWA 速记

## 问题与动机
目标是让开放域视频同时听懂语义（文本）、首帧外观（图像）和运动路径（轨迹），并超越简单数据或直线路径。

## 方法主线
Trajectory Sampler 处理任意轨迹，Multiscale Fusion 提供不同控制粒度，Adaptive Training 提高生成视频跟随轨迹的一致性。

## 读后要点
- 它是点轨迹进入开放域视频生成的重要早期工作。
- 多模态条件和多个模块让效果归因要依赖完整消融。
- Wan-Move 延续轨迹接口，但追求以 latent 特征修改替代独立运动融合模块。
- 延伸阅读：[DragNUWA 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
