# Wan-Move 官方推理代码

来源：https://github.com/ali-vilab/Wan-Move（HTTPS 浅克隆 `--depth 1`，`GIT_LFS_SKIP_SMUDGE=1`，经本机 `127.0.0.1:7890` 代理）。嵌套 `.git` 已删除，源码由本仓库直接跟踪。

`main` @ `80c58a7`（2026-01-05：MoveBench 增加 first-frame mask）。本目录不含 14B 权重。

| 内容 | 路径 |
|------|------|
| I2V 推理入口 | `generate.py`（`--task wan-move-i2v`） |
| 轨迹条件注入 | `wan/wan_move.py`（`WanMove.generate`，约第 279 行调用 `replace_feature`） |
| 首帧潜特征沿轨迹搬运 | `wan/modules/trajectory.py`（`replace_feature`） |
| 轨迹可视化 | `scripts/visualize.py`、`draw_tracks_on_video` |
| 交互轨迹绘制 | `gradio_app.py` |
| 示例轨迹 | `examples/example_tracks.npy`、`examples/example_visibility.npy` |

权重从 Hugging Face 拉到仓库外目录（见官方 `README.md`）：[Ruihang/Wan-Move-14B-480P](https://huggingface.co/Ruihang/Wan-Move-14B-480P)。

论文原文：[`arxiv/geometry_control/wan_move/`](../../../arxiv/geometry_control/wan_move/)。项目页：https://wan-move.github.io/
