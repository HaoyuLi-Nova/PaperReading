# Tora 自测问答

1. **Trajectory Extractor 输出什么？**
   与视频时空 token 对齐的 motion patches，而非原始像素轨迹图。
2. **Motion-guidance Fuser 做什么？**
   将轨迹 patches 融入 DiT 的生成特征，使其影响视频合成。
3. **为什么使用稠密到稀疏训练？**
   稠密信号帮助模型学习一般运动结构，稀疏轨迹更贴近实际交互输入；单独使用一类可能信号不足或与推理稀疏度不匹配。
4. **Tora 与 Wan-Move 在“注入”上的核心差别是什么？**
   Tora 加轨迹 VAE 与融合器；Wan-Move 直接编辑 I2V 条件 latent，并沿轨迹复制首帧特征。
5. **怎样验证模块复杂度是否值得？**
   在同底模、同训练数据和预算下比较控制误差、画质、速度、参数量，并做移除 extractor/fuser 的消融。
