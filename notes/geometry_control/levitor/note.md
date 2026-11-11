# LeviTor 速记

## 问题与动机
2D 拖拽接口容易使用，但点在图像中的运动无法表达靠近/远离相机等深度变化。

## 方法主线
用户为轨迹点提供相对深度；方法结合对象实例信息，从 mask 中聚类出少量控制点，再将点、深度和实例作为视频扩散模型的条件。它补的是深度维度，不要求用户完整建模 3D 场景。

## 读后要点
- 相对深度能消除部分出平面歧义，但不等价于完整 3D pose 或准确遮挡关系。
- 与 Wan-Move 对照时，重点看深度带来的出平面控制收益，以及 ControlNet/实例信息的额外成本。
- 延伸阅读：[LeviTor 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
