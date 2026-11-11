# `code/` — 论文学习代码

按 `code/<track>/<slug>/` 与笔记、原文对齐。这里放**可运行的最小实现 / 教程**，或官方仓库的源码；不是仓库运维脚本（运维在 `scripts/`）。

放入官方代码时完整克隆，并在交给本仓库跟踪之前删掉嵌套 `.git`：

```bash
git clone <url> code/<track>/<slug>
rm -rf code/<track>/<slug>/.git
```

不要使用 `--depth 1`。留下 `.git` 时，父仓库只会记下 gitlink，GitHub 收不到源码。权重、图片和压缩包仍由仓库根目录 `.gitignore` 排除。新增目录后，在 `papers.yaml` 该条目加上 `code:` 字段。

## Foundations

| 论文 | 路径 | 运行 |
|------|------|------|
| Transformer | [`foundations/transformer/toy/`](foundations/transformer/toy/) | `cd foundations/transformer/toy && python demo.py` |
| SimCLR | [`foundations/simclr/`](foundations/simclr/) | `python foundations/simclr/simclr_minimal.py` |
| Jigsaw | [`foundations/jigsaw/`](foundations/jigsaw/) | 见脚本内说明 |
| Dynamic Conv | [`foundations/dynamic_conv/`](foundations/dynamic_conv/) | 见脚本内说明 |
| Deformable Conv | [`foundations/deformable_conv/`](foundations/deformable_conv/) | 见脚本内说明 |
| AdaLN | [`foundations/adaln/`](foundations/adaln/) | `cd foundations/adaln && python demo.py` |
| LayerNorm | [`foundations/adaln/layernorm_demo.py`](foundations/adaln/layernorm_demo.py) | `cd foundations/adaln && python layernorm_demo.py` |

## Base models

| 论文 | 路径 | 运行 |
|------|------|------|
| LDM | [`base_models/ldm/`](base_models/ldm/) | `python base_models/ldm/featuremap_to_sequence.py` |
| Wan | [`base_models/Wan2.1/`](base_models/Wan2.1/)、[`base_models/Wan2.2/`](base_models/Wan2.2/) | 官方推理代码；笔记见 `notes/base_models/wan2.1/`；训练代码阅读路线见 [`mycode/wan2.1/README.md`](mycode/wan2.1/README.md) |
| Ovi | [`base_models/ovi/`](base_models/ovi/) | 官方推理代码；见目录内 `README.md` / `README_LOCAL.md` |

## Audio driven

| 论文 | 路径 | 运行 |
|------|------|------|
| MultiTalk | [`audio_driven/multitalk/`](audio_driven/multitalk/) | 官方推理代码（SSH clone）；见 `README.md` / `README_LOCAL.md` |
| AnyTalker | [`audio_driven/anytalker/`](audio_driven/anytalker/) | 官方推理代码（SSH 完整克隆）；见 `README.md` / `README_LOCAL.md` |
| InterActHuman | [`audio_driven/interacthuman/`](audio_driven/interacthuman/) | 官方 demonstration 代码（SSH 完整克隆）；见 `README.md` / `README_LOCAL.md` |

## Geometry control

路径与 `papers.yaml` 的 `geometry_control` slug 对齐。权重不在本仓库，见目录内 `README_LOCAL.md`。

| 论文 | 路径 | 来源 |
|------|------|------|
| Wan-Move | [`geometry_control/wan_move/`](geometry_control/wan_move/) | [ali-vilab/Wan-Move](https://github.com/ali-vilab/Wan-Move) |
| LeviTor | [`geometry_control/levitor/`](geometry_control/levitor/) | [ant-research/LeviTor](https://github.com/ant-research/LeviTor) |
| MotionCanvas | [`geometry_control/motioncanvas/`](geometry_control/motioncanvas/) | [Doubiiu/MotionCanvas](https://github.com/Doubiiu/MotionCanvas) |
| MotionCtrl | [`geometry_control/motionctrl/`](geometry_control/motionctrl/) | [TencentARC/MotionCtrl](https://github.com/TencentARC/MotionCtrl) |
| CameraCtrl | [`geometry_control/cameractrl/`](geometry_control/cameractrl/) | [hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl) |
| DragAnything | [`geometry_control/draganything/`](geometry_control/draganything/) | [showlab/DragAnything](https://github.com/showlab/DragAnything) |
| 3DTrajMaster | [`geometry_control/3dtrajmaster/`](geometry_control/3dtrajmaster/) | [KlingAIResearch/3DTrajMaster](https://github.com/KlingAIResearch/3DTrajMaster) |
| UCPE | [`geometry_control/ucpe/`](geometry_control/ucpe/) | [chengzhag/UCPE](https://github.com/chengzhag/UCPE) |
| EchoMotion | [`geometry_control/echomotion/`](geometry_control/echomotion/) | [D2I-ai/EchoMotion](https://github.com/D2I-ai/EchoMotion) |
| MotionCrafter | [`geometry_control/motioncrafter/`](geometry_control/motioncrafter/) | [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) |
| Free-Form Motion Control | [`geometry_control/free_form_motion_control/`](geometry_control/free_form_motion_control/) | [FudanCVL/SynFMC](https://github.com/FudanCVL/SynFMC) |
| SymphoMotion | [`geometry_control/symphomotion/`](geometry_control/symphomotion/) | [grenoble-zhang/SymphoMotion](https://github.com/grenoble-zhang/SymphoMotion) |
| CoMoVi | [`geometry_control/comovi/`](geometry_control/comovi/) | [IGL-HKUST/CoMoVi](https://github.com/IGL-HKUST/CoMoVi) |
| MoGe4D | [`geometry_control/moge4d/`](geometry_control/moge4d/) | [Zhangyr2022/MoGe4D](https://github.com/Zhangyr2022/MoGe4D) |
| Gen4P | [`geometry_control/gen4p/`](geometry_control/gen4p/) | [yitongx/Gen4P](https://github.com/yitongx/Gen4P) |
| D4RT | [`geometry_control/d4rt/`](geometry_control/d4rt/) | 非官方 [Lijiaxin0111/Open-d4rt](https://github.com/Lijiaxin0111/Open-d4rt) |
| DVD | [`geometry_control/dvd/`](geometry_control/dvd/) | [EnVision-Research/DVD](https://github.com/EnVision-Research/DVD) |
| SpatialTrackerV2 | [`geometry_control/spatialtracker_v2/`](geometry_control/spatialtracker_v2/) | [henry123-boy/SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) |
| TAPIP3D | [`geometry_control/tapip3d/`](geometry_control/tapip3d/) | [zbw001/TAPIP3D](https://github.com/zbw001/TAPIP3D) |
| St4RTrack | [`geometry_control/st4rtrack/`](geometry_control/st4rtrack/) | [HavenFeng/St4RTrack](https://github.com/HavenFeng/St4RTrack) |
| MonST3R | [`geometry_control/monst3r/`](geometry_control/monst3r/) | [Junyi42/monst3r](https://github.com/Junyi42/monst3r) |
| VGGT | [`geometry_control/vggt/`](geometry_control/vggt/) | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |
| 4RC | [`geometry_control/four_rc/`](geometry_control/four_rc/) | [Luo-Yihang/4RC](https://github.com/Luo-Yihang/4RC) |
| V-DPM | [`geometry_control/vdpm/`](geometry_control/vdpm/) | [eldar/vdpm](https://github.com/eldar/vdpm) |
| Any4D | [`geometry_control/any4d/`](geometry_control/any4d/) | [Any-4D/Any4D](https://github.com/Any-4D/Any4D) |
| Trace Anything | [`geometry_control/traceanything/`](geometry_control/traceanything/) | [ByteDance-Seed/TraceAnything](https://github.com/ByteDance-Seed/TraceAnything) |
| TrackCraft3R | [`geometry_control/trackcraft3r/`](geometry_control/trackcraft3r/) | [cvlab-kaist/TrackCraft3r](https://github.com/cvlab-kaist/TrackCraft3r) |
| Track2View | [`geometry_control/track2view/`](geometry_control/track2view/) | [mvrl/Track2View](https://github.com/mvrl/Track2View)（目前仅 README / assets） |

## Multi view

路径与 `papers.yaml` 的 `multi_view` slug 对齐。权重不在本仓库，见各目录 `README_LOCAL.md`。

| 论文 | 路径 | 来源 |
|------|------|------|
| QuerySplat | [`multi_view/querysplat/`](multi_view/querysplat/) | [inspatio/QuerySplat](https://github.com/inspatio/QuerySplat) |

## EEG Visual

路径与 `papers.yaml` 的 `eeg_visual` slug 对齐。权重不在本仓库。

| 论文 | 路径 | 来源 |
|------|------|------|
| NICE | [`eeg_visual/nice/`](eeg_visual/nice/) | [eeyhsong/NICE-EEG](https://github.com/eeyhsong/NICE-EEG) |
| ATM | [`eeg_visual/atm/`](eeg_visual/atm/) | [ncclab-sustech/EEG_Image_decode](https://github.com/ncclab-sustech/EEG_Image_decode) |
| DreamDiffusion | [`eeg_visual/dreamdiffusion/`](eeg_visual/dreamdiffusion/) | [bbaaii/DreamDiffusion](https://github.com/bbaaii/DreamDiffusion) |
| CognitionCapturer | [`eeg_visual/cognitioncapturer/`](eeg_visual/cognitioncapturer/) | [XiaoZhangYES/CognitionCapturer](https://github.com/XiaoZhangYES/CognitionCapturer) |
| NECOMIMI | [`eeg_visual/necomimi/`](eeg_visual/necomimi/) | [ChiShengChen/EEG_gen_img_NECOMIMI](https://github.com/ChiShengChen/EEG_gen_img_NECOMIMI) |
| MindCine | [`eeg_visual/mindcine/`](eeg_visual/mindcine/) | [KevinZhou6/MindCine](https://github.com/KevinZhou6/MindCine) |
| Neuro-3D | [`eeg_visual/neuro_3d/`](eeg_visual/neuro_3d/) | [gzq17/neuro-3D](https://github.com/gzq17/neuro-3D) |
| Mind2Matter | [`eeg_visual/mind2matter/`](eeg_visual/mind2matter/) | [sddwwww/Mind2Matter](https://github.com/sddwwww/Mind2Matter) |
| 3D-Telepathy | [`eeg_visual/eeg_to_3d/`](eeg_visual/eeg_to_3d/) | [gegen666/EEGTo3D](https://github.com/gegen666/EEGTo3D) |

## Remote sensing

路径与 `papers.yaml` 的 `remote_sensing` slug 对齐。权重与训练数据不在本仓库。

| 论文 | 路径 | 来源 |
|------|------|------|
| S³Mamba-Pan | [`remote_sensing/s3mamba_pan/`](remote_sensing/s3mamba_pan/) | [FreeZS-a/S3Mamba](https://github.com/FreeZS-a/S3Mamba) |

