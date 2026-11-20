
<div align="center">

# **BEDLAM2.0: Synthetic Humans and Cameras in Motion (NeurIPS 2025)**

🌐 [**Project Page**](https://bedlam2.is.tue.mpg.de) | 📄 [**Paper**](https://bedlam2.is.tuebingen.mpg.de/media/upload/BEDLAM2_NeurIPS2025.pdf) | 🎥 [**Video Results**](https://www.youtube.com/watch?v=ylyqHnwhpsY)

</div>

---

This ReadMe provides instructions for using the BEDLAM2 (SMPL-X) dataset in training and evaluating CameraHMR. For the SMPL version, please refer to the main [ReadMe.md](../ReadMe.md).

## **Installation**
Create a conda environment and install all the requirements.

```
conda create -n camerahmr python=3.10
conda activate camerahmr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## **Demo (SMPLX)**
###  **Download Demo required data**
 1. Register on the [BEDLAM2 website](https://bedlam2.is.tue.mpg.de/)
 2. Register on the [CameraHMR website](https://camerahmr.is.tue.mpg.de/)
 3. Register on the [SMPL-X website](https://smpl-x.is.tue.mpg.de/).
 4. Run the following script:
 ```
 bash scripts/fetch_demo_data_bedlam2.sh
 ```
### **Run Demo**
Run the demo with the following command. It will run the demo on all images in the specified `--image_folder`, and save renderings of the reconstructions and the output mesh in `--out_folder`. Make sure to set `--model_type` to `smplx`.

```
python demo.py --image_folder demo_images --output_folder output_images_smplx --model_type smplx
```

## **Training and Evaluation (SMPLX)**

###  **Download Training required data**
1. Register on the [BEDLAM2 website](https://bedlam2.is.tue.mpg.de/).
2. Download and untar the BEDLAM2 30fps images (`-png`) in `data/training-images/bedlam_v2` and the GT motion file `b2_motions_npz_training.tar` in `data/training-labels/bedlam_v2` from the [BEDLAM2 download page](https://bedlam2.is.tue.mpg.de/download.php).
3.  Run the following script to download extra necessary files.
        ```bash
        bash download_util_bedlam2.sh 
4. Note that if you want to use BEDLAM1 data in training as well then you need to download the labels from [BEDLAM website](https://bedlam.is.tue.mpg.de/) from the section **SMPL-X ground truth labels compatible with BEDLAM2**. After downloading them unzip in `data/training-labels/bedlam-labels-v2-format`

### **Training**
Once the data is downloaded, you can run the training with the following command. We override the `MODEL.TYPE` to `smplx` to train an SMPLX model.

```
python train.py data=bedlam_v2_v1 experiment=bedlam2 exp_name=train_smplx_run1
```

### **Evaluation**

###  **Download Evaluation required data**

To run the evaluation along with the checkpoints and SMPLX model files downloaded in **Demo** section you also need to download the test labels for 3DPW, EMDB, RICH from CameraHMR website using the following script. The following script also download some utilty files needed to run the evaluation.

```
bash scripts/fetch_test_labels.sh
```

> **Note:** We cannot provide the original images for 3DPW, EMDB, RICH. These images must be obtained from their original sources.

The images could be downloaded and stored in **data/test-images**. This is the default directory structure for the images. If you have stored images at other location you could modify the path of the images [here](../core/configs/__init__.py)

```
├── 3DPW
│   └── imageFiles
├── EMDB
│   ├── P0
│   ├── P1
│   ├── P2
│   ├── P3
│   ├── P4
│   ├── P5
│   ├── P6
│   ├── P7
│   ├── P8
│   └── P9
├── RICH
│   └── test
```
### **Evaluation**

For evaluation, use the following command. 
```
python eval.py data=eval_smplx experiment=bedlam2
```
