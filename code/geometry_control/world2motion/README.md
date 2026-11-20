# World2Motion Inference

Generate an RGB video and a SMPL-X motion sequence from a first-frame image and its corresponding first-frame motion.

## Setup

Use Linux x86-64 (glibc 2.35+), Python 3.13, and CUDA 12.8. The default inference command was tested on an H100 MIG with 47 GiB of GPU memory; peak allocated memory was about 40.9 GiB.

```bash
git clone --depth 1 https://github.com/fyantu/World2Motion.git world2motion
cd world2motion
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r scripts/requirements.txt
python -m pip install --no-deps \
  'flash-attn-3-nv==1.0.3+cu128.torch210' \
  'natten==0.21.6.dev6+cu128.torch210' \
  --index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0
```

Place the inference assets in `checkpoints/`:

```text
world2motion/
├── scripts/                  # MoT, video VAE, and inference sampler
├── checkpoints/
│   ├── model.safetensors     # HOI-mix, step 20000; final EMA BF16 weights
│   ├── Wan2.2_VAE.pth        # Video encoder/decoder weights
│   ├── config.json
│   ├── qwen_config.json
│   └── tokenizer/           # tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
├── data/                     # Image/motion preprocessing and normalization statistics
├── inference.py
└── README.md
```

The model and VAE weights are distributed separately from GitHub. The download link will be added after the weight upload. The joint model is approximately 28.3 GiB; the video VAE is also required.

## Inputs

- **Image:** a `.png` or `.jpg` showing the first frame. It is resized while preserving its aspect ratio.
- **Motion:** an `.npz` containing the corresponding first-frame SMPL-X pose. Use axis-angle rotations in **radians** and camera-space translation in **meters**.

| Field | First-frame shape | Meaning |
| --- | --- | --- |
| `root_pose` | `(3,)` | Global root orientation |
| `body_pose` | `(63,)` or `(21, 3)` | The 21 SMPL-X body-joint rotations |
| `cam_trans` | `(3,)` | Root translation in the camera coordinate system |

A leading frame dimension is accepted; only frame zero is read. Optional `betas` and a Unicode-string `gender` are carried into the output. Hand and facial pose parameters are not generated. Supply raw pose values; preprocessing performs the required normalization.

```python
import numpy as np

# root_pose, body_pose, and cam_trans are the pose corresponding to frame0.png.
np.savez(
    "motion0.npz",
    root_pose=np.asarray(root_pose, dtype=np.float32).reshape(3),
    body_pose=np.asarray(body_pose, dtype=np.float32).reshape(63),
    cam_trans=np.asarray(cam_trans, dtype=np.float32).reshape(3),
)
```

## Run inference

```bash
python inference.py \
  --image /path/to/frame0.png \
  --motion /path/to/motion0.npz \
  --output outputs/example
```

Optionally add `--prompt "A person sits down on a chair."` to describe the desired motion. Use `--seed 42` for a fixed seed. Defaults are **121 frames**, **24 FPS**, the **720 resolution tier**, and **50 sampling steps**. Actual output dimensions follow the input aspect ratio and the model's spatial alignment.

For a short test, add `--frames 5 --resolution 256 --steps 2`. Lowering the resolution or number of frames reduces memory use. Choose a new output directory for each run.

## Outputs

```text
outputs/example/
├── video.mp4       # Generated RGB video, 24 FPS
├── motion.npz      # Numerical SMPL-X motion, aligned frame-for-frame with the video
└── inference.json  # Input paths, settings, and output dimensions
```

`motion.npz` contains `root_pose` shaped `(T, 3)`, `body_pose` shaped `(T, 63)`, `cam_trans` shaped `(T, 3)`, and scalar `fps`. Frame zero contains the supplied conditioning pose. Translations are restored to the input camera coordinate system.

```python
import numpy as np

motion = np.load("outputs/example/motion.npz", allow_pickle=False)
print(motion["root_pose"].shape, motion["body_pose"].shape, motion["cam_trans"].shape)
```

## TODO

- [ ] Release training code.
