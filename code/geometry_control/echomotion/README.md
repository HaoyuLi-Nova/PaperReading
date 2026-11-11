<h1 align="center">
  EchoMotion: Unified Human Video and Motion Generation via Dual-Modality Diffusion Transformer
</h1>

<h3 align="center">
  ICLR 2026
</h3>

<p align="center">
  <a href="https://yuxiaoyang23.github.io/EchoMotion-webpage/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=flat-square" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2512.18814">
    <img src="https://img.shields.io/badge/arXiv-2512.18814-b31b1b?style=flat-square" alt="arXiv">
  </a>
  <a href="https://github.com/D2I-ai/EchoMotion">
    <img src="https://img.shields.io/badge/Code-GitHub-yellow?style=flat-square" alt="Code">
  </a>
  <a href="https://huggingface.co/Yuxiao319/EchoMotion">
    <img src="https://img.shields.io/badge/Models-HuggingFace-orange?style=flat-square" alt="Models">
  </a>
  <a href="https://github.com/D2I-ai/EchoMotion/extract_motion.py">
    <img src="https://img.shields.io/badge/Dataset-Dataset-green?style=flat-square" alt="Dataset Pipe">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-green?style=flat-square" alt="License">
  </a>
</p>

<!-- Author Information Section -->
<p align="center">
  <!-- Authors -->
  <a href="https://yuxiaoyang23.github.io/">Yuxiao Yang</a><sup>1,2</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=73JaDUQAAAAJ&hl=zh-CN">Hualian Sheng</a><sup>2</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=LMVeRVAAAAAJ&hl=en">Sijia Cai</a><sup>2,*</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://jinglin7.github.io/">Jing Lin</a><sup>3</sup>
  <br>
  <a href="https://scholar.google.com/citations?user=zQnTBEoAAAAJ&hl=en">Jiahao Wang</a><sup>4</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=VQp_ye4AAAAJ&hl=zh-CN">Bing Deng</a><sup>2</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=907PxdcAAAAJ&hl=en">Junzhe Lu</a><sup>1</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?hl=zh-CN&user=eldgnIYAAAAJ&view_op=list_works&sortby=pubdate">Haoqian Wang</a><sup>1,†</sup>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=T9AzhwcAAAAJ&hl=en">Jieping Ye</a><sup>2,†</sup>
  <br><br>
  <!-- Affiliations -->
  <sup>1</sup>Tsinghua University&nbsp;&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Alibaba Group&nbsp;&nbsp;&nbsp;&nbsp;
  <sup>3</sup>Nanyang Technological University&nbsp;&nbsp;&nbsp;&nbsp;
  <sup>4</sup>Xi'an Jiaotong University
  <br>
  <!-- Legend -->
  <em><small><sup>*</sup>Project Lead&nbsp;&nbsp;&nbsp;&nbsp;<sup>†</sup>Corresponding Author</small></em>
</p>

<p align="center">
  <img src="assets/pipeline.png" alt="EchoMotion Pipeline" width="100%">
</p>

<p align="center">
  <b>Overview of EchoMotion.</b> (a) The dual-modality DiT block for joint video-motion modeling. (b) Our MVS-RoPE to serve as a synchronized coordinate for dual-modal token sequence.
</p>

## 💡 Abstract

> Video generation models have advanced significantly, yet they still struggle to synthesize complex human movements due to the high degrees of freedom in human articulation. This limitation stems from the intrinsic constraints of pixel-only training objectives, which inherently bias models toward appearance fidelity at the expense of learning underlying kinematic principles. To address this, we introduce **EchoMotion**, a framework designed to model the joint distribution of appearance and human motion, thereby improving the quality of complex human action video generation. EchoMotion extends the DiT (Diffusion Transformer) framework with a dual-branch architecture that jointly processes tokens concatenated from different modalities. Furthermore, we propose **MVS-RoPE (Motion-Video Synchronized RoPE)**, which offers unified 3D positional encoding for both video and motion tokens. By providing a synchronized coordinate system for the dual-modal latent sequence, MVS-RoPE establishes an inductive bias that fosters temporal alignment between the two modalities. We also propose a **Motion-Video Two-Stage Training Strategy**. This strategy enables the model to perform both the joint generation of complex human action videos and their corresponding motion sequences, as well as versatile cross-modal conditional generation tasks. To facilitate the training of a model with these capabilities, we construct ***HuMoVe***, a large-scale dataset of approximately 80,000 high-quality, human-centric video-motion pairs. Our findings reveal that explicitly representing human motion is complementary to appearance, significantly boosting the coherence and plausibility of human-centric video generation.

## ✨ Key Features

- **Joint Video & Motion Modeling**: Instead of just pixels, EchoMotion learns the relationship between appearance and underlying human motion, leading to more physically plausible results.
- **Novel Architecture**: Introduces a **Dual-Branch Diffusion Transformer** with **MVS-RoPE** for synchronized positional encoding, effectively aligning video and motion modalities.
- **Versatile Generation Tasks**: A single unified framework supports multiple tasks:
  - Text-to-Video-and-Motion Generation
  - Motion-to-Video Generation
  - Video-to-Motion Prediction
- **New Large-Scale Dataset**: We introduce ***HuMoVe***, a high-quality dataset of ~80,000 video-motion pairs to facilitate research in this area.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EchoMotion.git
cd EchoMotion
```

### 2. Setup Environment

We recommend using Conda to create an environment:

```bash
conda create -n echomotion python=3.10
conda activate echomotion
```

### 3. Install Dependencies

```bash
apt-get install -y libgl1-mesa-glx
apt-get install -y libegl1-mesa libegl1

pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

pip install --no-build-isolation https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl#sha256=ca1403ccf84c27ab3ac409b043603281a7f82c6e0abd197c308a492d1ef49739
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# fix chumpy dependency issue
CHUMPY_SITE_PACKAGES=$(pip show chumpy | grep Location | cut -d ' ' -f 2) && sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf/#&/' ${CHUMPY_SITE_PACKAGES}/chumpy/__init__.py && echo "chumpy fix applied successfully!"

# fix torchgeometry dependency issue
TGM_PATH=$(pip show torchgeometry | grep Location | cut -d ' ' -f 2) && \
sed -i \
-e 's/1 - mask_d0_d1/~mask_d0_d1/g' \
-e 's/1 - mask_d2/~mask_d2/g' \
-e 's/1 - mask_d0_nd1/~mask_d0_nd1/g' \
${TGM_PATH}/torchgeometry/core/conversions.py && \
echo "torchgeometry fix applied successfully!"
```

### 4. Download Pre-trained Models
Download SMPL model:
```bash
bash extract_motion/CameraHMR/fetch_smpl_model.sh
```

Download the pre-trained models and place them in the appropriate directories:
- Model checkpoints: Place in `./ckpt`
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Yuxiao319/EchoMotion --local-dir ./ckpt
```

## 🤖 Inference
### Enhanced Prompt Rewriting with Dashscope. 
To significantly improve detail of generated videos especially for motion-to-video task, we integrates with the Alibaba Cloud Dashscope API to automatically rewrite user prompts.
  - Obtain an API Key according to the instructions ([EN](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen) | [CN](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen)).
  - Configure the environment variable `DASH_API_KEY` to specify the Dashscope API key. For users of Alibaba Cloud's international site, you also need to set the environment variable `DASH_API_URL` to 'https://dashscope-intl.aliyuncs.com/api/v1'. For more detailed instructions, please refer to the [dashscope document](https://www.alibabacloud.com/help/en/model-studio/developer-reference/use-qwen-by-calling-api?spm=a2c63.p38356.0.i1).

  ```bash
  export DASH_API_KEY=your_key 
  ```
  - Model Usage: 
  The prompt rewriting feature utilizes the following models for different tasks:
    - Text-to-Video-with-Motion Task: `qwen-plus`.
    - Motion-to-Video Task: `qwen-vl-max`.

### 1. Text-to-Video-and-Motion Generation

Generate both video and SMPL motion from a text prompt:

```bash

python generate.py \
    --sample_mode joint \
    --prompt "视频记录了一位单板滑雪者在雪地环境中完成一次障碍技巧动作的全过程。场景设定在阳光明媚的滑雪场，背景是覆盖着皑皑白雪的山坡和茂密的常绿针叶林，天空呈现蓝色并伴有大块白色云朵。视频主角是一位非裔面孔的滑雪者，身着鲜艳的红色滑雪夹克、黑色滑雪裤，头戴黑色头盔和反光雪镜，佩戴黑色手套，脚踏单板滑雪板。视频开始时，滑雪者正沿雪坡向下滑行，身体微屈，双臂展开以维持平衡。随后，滑雪者驱动雪板，流畅地跃上障碍道具，完成障碍技巧动作。最后，他稳稳落地继续向前滑行。镜头调度流畅，稳定地跟随主体运动，没有明显的剪辑转场，呈现为一个连贯的动作记录。"

# To use prompt extension, add --use_prompt_extend
python generate.py \
    --sample_mode joint \
    --use_prompt_extend \
    --prompt "一个男人在森林中骑行山地自行车。"

python generate.py \
    --sample_mode joint \
    --use_prompt_extend \
    --prompt "视频记录一个男人在户外完成高尔夫击球的过程。视频主角完成一整套连贯的击球动作。"
```

### 2. Motion-to-Video Generation

Generate video from an existing SMPL motion sequence:

```bash
python generate.py \
    --sample_mode motion_to_video \
    --input_motion dataset/demo/motions/demo_motion_0.pt \
    --prompt "一位年轻女性在湖畔步道晨跑，阳光微露的清晨时分，天空泛着淡淡的粉紫色光晕，薄雾轻笼水面。她有着浅褐色皮肤，乌黑的长发编成一条松散的辫子垂于肩侧，佩戴深灰色无线耳机，手腕上戴着简约银色运动手环。身穿深酒红色运动套装，包括短袖紧身衣与高腰长裤，勾勒出流畅的身体线条。她正从画面左侧向右侧匀速前行，步伐稳健，双臂摆动自然，脸上带着宁静的微笑，神情专注而放松。背景是开阔的淡水湖，水面倒映着朦胧天色，远处山影隐约可见。镜头采用侧面中景跟随拍摄，主体清晰居中，背景虚化柔和，光线为斜射的晨光，温暖的光束穿过薄雾洒在人物身上，形成细腻的明暗过渡，营造出静谧而富有生机的运动氛围。"

python generate.py \
    --sample_mode motion_to_video \
    --input_motion dataset/demo/motions/demo_motion_1.pt \
    --prompt "一名约十至十二岁的东亚男孩在空手道训练馆内进行基础冲拳练习，神情专注而沉稳。他留着乌黑的短发，眼神坚定地望向前方，透出超越年龄的自律与决心。身穿整洁的白色空手道服，腰间系着橙色腰带，象征其初学者中段级别，道服随着动作发出轻微的沙沙声。他站定马步，双拳紧握，交替完成正拳出击与回拉动作，每一次出拳都配合转腰发力，节奏清晰，展现出扎实的基本功与身体控制力。场景位于一间光线充足的室内道场，地面为浅橡木色实木地板，反射出柔和光泽。背景虚化处理下，仍可见墙边整齐排列的蓝绿相间海绵护垫和角落放置的灰色健身球，右侧设有宽大的落地窗，透露出室外微阴天气下的漫射光线。镜头采用中景侧拍视角，主体居于画面中央，浅景深突出人物轮廓与动作张力。主光源来自右侧窗户的自然侧光，在面部左侧形成明亮高光，另一侧过渡为细腻阴影，强化了面部立体感与皮肤质感，整体色调保持真实自然，白色道服与木质地面、背景器械形成温暖而协调的视觉层次。"

```

### 3. Video-to-Motion Prediction

Extract SMPL motion from a video:

```bash
python generate.py \
    --sample_mode video_to_motion \
    --input_video dataset/demo/videos/demo_video_0.mp4
```


## 🎬 Extract Motion from Custom Videos and Inference with Motion-to-Video Mode.

To download pretrained models:
```bash
bash extract_motion/CameraHMR/fetch_pretrained_models.sh
```

To extract SMPL motion from your own videos:

```bash
python3 extract_motion.py \
  --video_path dataset/demo/videos/demo_video_0.mp4 \
  --output_path outputs/extract_motion/motion_0.pt \
  --use_shape \
  --visualize \
  --overwrite
```

To generate a video from a custom motion sequence and apply prompt extension using the input video as reference:

```bash
python generate.py \
    --sample_mode motion_to_video \
    --input_motion outputs/extract_motion/motion_0.pt \
    --input_video dataset/demo/videos/demo_video_0.mp4 \
    --use_prompt_extend \
    --prompt "一个身着运动服的男孩在室外慢跑。"

python generate.py \
    --sample_mode motion_to_video \
    --input_motion outputs/extract_motion/motion_1.pt \
    --input_video dataset/demo/videos/demo_video_1.mp4 \
    --use_prompt_extend \
    --prompt "一个亚洲女孩正在练习拳击。"

python generate.py \
    --sample_mode motion_to_video \
    --input_motion outputs/extract_motion/motion_2.pt \
    --input_video dataset/demo/videos/demo_video_2.mp4 \
    --use_prompt_extend \
    --prompt "一个穿着运动衣的女生，在室外练习投篮动作。相机静止不动。"
```

## 📊 HuMoVe Dataset

Training a model like EchoMotion requires a large-scale, high-quality dataset of paired video and motion data. We introduce ***HuMoVe***, containing approximately 80,000 video-motion pairs.

- **Wide Category Coverage**: Spans a diverse range of human activities
- **High-Quality Annotations**: Detailed text descriptions and precise SMPL motion sequences
- **High-Fidelity Videos**: High-resolution, clean video clips

Due to legal compliance restrictions, we are unable to release the complete video materials. Instead, we provide the full motion processing pipeline in [extract_motion.py](extract_motion.py). Text annotations can be generated using [Qwen-VL-Narrator](https://github.com/D2I-ai/Qwen-VL-Narrator).

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@article{yang2025echomotion,
  title={EchoMotion: Unified Human Video and Motion Generation via Dual-Modality Diffusion Transformer},
  author={Yang, Yuxiao and Sheng, Hualian and Cai, Sijia and Lin, Jing and Wang, Jiahao and Deng, Bing and Lu, Junzhe and Wang, Haoqian and Ye, Jieping},
  journal={arXiv preprint arXiv:2512.18814},
  year={2025}
}
```

## 🙏 Acknowledgements

We would like to express our gratitude for the following projects and teams that were instrumental in the development of our work:

*   **[Qwen-VL-Narrator](https://github.com/D2I-ai/Qwen-VL-Narrator)**: For their excellent tool, which was used for the textual annotation of our ***HuMoVe*** dataset.
*   **[CameraHMR](https://camerahmr.is.tue.mpg.de/)**: For providing the robust framework used for the SMPL annotations in our dataset.
*   **[The Wan Team](https://github.com/Wan-Video)**: For their valuable open-source models that contributed to our research.

## 📜 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) License. See the [LICENSE](LICENSE) file for more details.

