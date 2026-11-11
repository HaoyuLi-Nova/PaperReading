# TrailBlazer 速记

## 问题与动机
点或像素级控制比较费力；用户通常更容易给出主体框在几个关键时刻的位置和大小。

## 方法主线
利用预训练视频扩散模型的空间、时间注意力，通过 trailing maps 和注意力编辑，让主体跟随动态 bounding boxes，并可结合局部提示控制关键帧外观。

## 读后要点
- 控制粒度是对象框，不等同于点轨迹：内部姿态、局部形变和遮挡顺序仍交给底模决定。
- 方法免训练但与底模 attention 结构耦合，且只能生成底模本身支持的时长。
- 延伸阅读：[TrailBlazer 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
