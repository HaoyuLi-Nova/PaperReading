# MotionCtrl 速记

## 问题与动机
视频动态由相机运动和物体运动共同构成；若条件混用，用户很难知道模型会移动镜头还是主体。

## 方法主线
MotionCtrl 分别建模相机 pose 与对象运动轨迹，在同一视频生成框架下允许二者独立或组合控制。它的关键不只是“加运动条件”，而是把两类运动当成不同控制变量处理。

## 读后要点
- 对象轨迹、相机轨迹不是天然可辨识的监督信号，训练数据如何拆分决定解耦上限。
- Wan-Move 的点轨迹接口可覆盖多种运动，却没有 MotionCtrl 那样显式的条件分支；两者对比需检查相机/物体控制串扰。
- 延伸阅读：[MotionCtrl 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
