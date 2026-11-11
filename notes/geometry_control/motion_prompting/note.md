# Motion Prompting 速记

## 问题与动机
文本很难精确描述路径、时序、速度和同步动作。Motion Prompting 把点轨迹当作可直接表达运动的 prompt。

## 方法主线
论文训练轨迹 ControlNet：轨迹可在时空上稀疏或稠密，点通过随机 embedding 表示并栅格化到像素网格；也探索把高层请求扩展为半稠密轨迹。轨迹从视频跟踪器获得或由用户指定。

## 读后要点
- 这是 Wan-Move 最直接的“点轨迹控制生成”前驱。
- 轨迹的运动位置很精确，但像素级随机编码没有周围纹理/语义。Wan-Move 的 latent feature replication 针对这一点搬运首帧上下文。
- 注意区分此文的轨迹 ControlNet 与 CoTracker：前者生成控制方法，后者是产生轨迹的追踪器。
- 延伸阅读：[Motion Prompting 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
