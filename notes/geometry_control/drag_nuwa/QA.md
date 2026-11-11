# DragNUWA 自测问答

1. **DragNUWA 联合了哪三类条件？**
   文本、图像和运动轨迹，分别提供语义、外观/空间和时间动态信息。
2. **Trajectory Sampler 解决什么问题？**
   让用户轨迹可以覆盖开放域的不同形状和采样需求。
3. **Multiscale Fusion 的作用是什么？**
   让模型在不同空间/运动粒度上吸收轨迹条件。
4. **它与 Motion Prompting 的共同点和差异是什么？**
   两者都用点轨迹控制视频生成；DragNUWA 更强调图像/文本/轨迹整合和多尺度设计，Motion Prompting 强调统一的稀疏/稠密 motion prompt 与通用 ControlNet。
5. **比较它和 Wan-Move 要控制哪些变量？**
   使用相同底模、视频数据、轨迹和训练预算，分别比较运动跟随、外观保持与额外模块开销。
