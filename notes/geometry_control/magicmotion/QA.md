# MagicMotion 自测问答

1. **MagicMotion 提供哪几种控制粒度？**
   Mask、bounding box 和 sparse box trajectory。
2. **MagicData 与 MagicBench 分别是什么？**
   前者是训练数据及标注/筛选管线，后者是评估视频质量和轨迹控制精度的 benchmark。
3. **比较 mask 控制和稀疏点轨迹时为何要小心？**
   Mask 提供的空间信息更多；不同输入的信息量不同，不能把分数差异直接归因于模型优劣。
4. **与 Wan-Move 的关系是什么？**
   两者都强调轨迹控制和配套评测；Wan-Move 以点轨迹 latent feature replication 为核心，MagicMotion 以多粒度条件和数据/benchmark 为体系重点。
5. **基准可信度应重点查什么？**
   自动标注误差、人工核验比例、视频时长与对象分布、训练/测试泄漏，以及不同控制条件的难度是否可比。
