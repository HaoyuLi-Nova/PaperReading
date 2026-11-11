# MagicMotion 速记

## 问题与动机
不同任务需要不同控制粒度；仅支持一种轨迹形式会限制交互，同时缺少专门 benchmark 使结果难比较。

## 方法主线
MagicMotion 支持 mask、bounding box 和 sparse box 三种由密到疏的条件；MagicData 提供训练视频和标注筛选流程，MagicBench 按对象数量评估轨迹控制和视频质量。

## 读后要点
- 贡献覆盖模型、数据集和 benchmark；阅读时需分别评估这三部分。
- MagicBench 与 Wan-Move 的 MoveBench 都瞄准轨迹控制评测，应核对样例、长度、标注精度和评价指标差异。
- 只应在相同输入格式上比较控制精度；mask 输入信息量高于稀疏点轨迹。
- 延伸阅读：[MagicMotion 精读](critique.md) · [Wan-Move 路线对照](../../../../comparisons/geometry_control/wan_move_related_reading.md)
