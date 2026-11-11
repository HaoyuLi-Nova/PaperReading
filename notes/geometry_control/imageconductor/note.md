# Image Conductor 速记

## 问题与动机
真实视频轨迹混有镜头运动和主体运动；创作任务常希望只改变其中一类。

## 方法主线
用 CoTracker 标注并筛选轨迹视频；训练 video ControlNet，以对象/相机 LoRA 分别表达两种运动并用正交损失减少串扰；camera-free guidance 在推理时增强物体运动、抑制镜头移动。

## 读后要点
- 重要贡献包含数据整理和镜头/物体运动解耦，不应只看网络结构。
- 相比 Wan-Move，Image Conductor 显式分支更重；Wan-Move 则以统一点轨迹方式服务于可扩展 I2V。
- 两者需在固定视角对象控制和纯相机控制上分别评测。
- 延伸阅读：[Image Conductor 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
