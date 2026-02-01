# ControlNet Sketch-Guided Super-Resolution Deployment Guide

# Overview

This guide details the deployment process of a dual-condition ControlNet model (trained for sketch-guided image and segmentation map super-resolution) to Stable Diffusion WebUI. The model takes a low-resolution image and a corresponding sketch as inputs, then generates a high-resolution image while preserving content and structure.

# Key Specifications
Base Model: Stable Diffusion v1.5
ControlNet Version: 1.1
Input: Low-resolution image (256×256, 3-channels) + Sketch (256×256, 3-channels) → 6-channel spliced input
Output: High-resolution image (512×512, 3-channels)
Training Mode: Lightweight LoRA training (GPU ≥12GB available) / Full parameter training (GPU ≥24GB recommended)
Inference Env: Stable Diffusion WebUI + ControlNet Extension

# Prerequisites (For Training & Deployment)
Ensure the following dependencies and files are prepared before training or deployment (shared environment for both training and deployment, no need to build separately):

1.[Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

2.[ControlNet Extension for WebUI](https://github.com/Mikubill/sd-webui-controlnet)

Clone to WebUI extension directory:
```
cd stable-diffusion-webui/extensions
git clone https://github.com/Mikubill/sd-webui-controlnet.git
```
3.Python Environment (3.8~3.10 recommended)
Install core dependencies for training (one-click command):
```
pip install torch==2.0.1 torchvision==0.15.2 diffusers==0.26.0 peft==0.8.2 accelerate==0.27.2 albumentations tqdm tensorboard opencv-python numpy
```
4.Trained/To-Train Model Files
For training: Only need the official SD1.5 base model (automatically downloaded by the code)
For deployment: Trained model weight (controlnet_sketch_guided_sr_webui.pth) + matching YAML config file (controlnet_sketch_guided_sr_webui.yaml)

# Model Training Guide (Core Content)
1. Training Code Preparation
Create a training script file train_controlnet_sketch_guided_sr.py in the project root directory, copy the complete training code (you can place the code in your GitHub repo and fill in the actual link here) into the file. The code has the following key features:
· Supports 6-channel dual-condition input (low-res image + sketch)
· Default enables LoRA lightweight training (≤1% trainable parameters, 12GB GPU available)
· Automatic dataset pairing verification, tensorboard log recording, breakpoint resume training
· One-click export of WebUI-compatible model weights (no additional conversion required)
2. Dataset Preparation (Critical for Training)
The model requires triple paired datasets (low-res image + sketch + high-res image) with 100% consistent filenames (to avoid data misalignment). Follow the directory structure below to organize the dataset:
Standard Dataset Directory Structure
```
your-project/
├─ dataset/                # Root directory of the dataset
│  ├─ low_res_ori/         # Condition 1: Low-resolution images (256×256, RGB)
│  │  ├─ 001.png
│  │  ├─ 002.jpg
│  │  └─ ...
│  ├─ sketch_mask/         # Condition 2: Corresponding sketch maps (256×256, RGB/GRAY)
│  │  ├─ 001.png           # Must have the same name as low_res_ori/001.png
│  │  ├─ 002.jpg           # Must have the same name as low_res_ori/002.jpg
│  │  └─ ...
│  └─ high_res_ori/        # Target: High-resolution images (512×512, RGB)
│     ├─ 001.png           # Must have the same name as the above two
│     ├─ 002.jpg           # Must have the same name as the above two
│     └─ ...
└─ train_controlnet_sketch_guided_sr.py  # Training script
```
Dataset Production Requirements

1.Size Consistency: Low-res images and sketches must be 256×256, high-res target images must be 512×512 (consistent with model input/output settings)

2.Filename Consistency: The three files corresponding to the same sample must have the same name and suffix (e.g., 001.png in all three folders)

3.Format Compatibility: Support PNG/JPG/JPEG; single-channel gray sketches are automatically converted to 3-channel RGB by the code

4.Dataset Scale: ≥1000 samples (avoid overfitting); unified style (e.g., only anime/only real scenes) is recommended
Quick Dataset Generation (Batch Resize)

If you only have high-resolution original images, use the following code to batch generate low-resolution images and resize sketches/high-res images to the specified size:
python
```
# batch_resize.py - One-click generation of dataset with specified size
import os
import cv2
from tqdm import tqdm

def batch_resize(img_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(img_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            # Use INTER_LINEAR for upscaling, INTER_AREA for downscaling (better effect)
            if img.shape[0] < target_size[0] or img.shape[1] < target_size[1]:
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, img_name), img_resized)

# Usage Example (modify the path according to your actual situation)
if __name__ == "__main__":
    # 1. Generate 512×512 high-res target images from raw high-res images
    batch_resize("raw_high_res", "dataset/high_res_ori", (512, 512))
    # 2. Generate 256×256 low-res images from 512×512 high-res images
    batch_resize("dataset/high_res_ori", "dataset/low_res_ori", (256, 256))
    # 3. Resize raw sketches to 256×256
    batch_resize("raw_sketch", "dataset/sketch_mask", (256, 256))
```
3. Training Parameter Configuration
Modify the core configuration section in the training script train_controlnet_sketch_guided_sr.py according to your GPU memory and dataset path (all key parameters are annotated, easy to modify):
python
```
# ===================== Core Training Config (Modify According to Your Env) =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"  # SD1.5 base model (auto-download)
# Dataset path (modify to your actual dataset root directory)
LOW_RES_ORI_DIR = "dataset/low_res_ori"
SKETCH_MASK_DIR = "dataset/sketch_mask"
HIGH_RES_ORI_DIR = "dataset/high_res_ori"
# Training parameters (GPU memory priority)
BATCH_SIZE = 2  # 12GB GPU=1/2, 24GB GPU=4/8, 32GB GPU=8/16 (even number recommended)
EPOCHS = 50     # ≤1000 samples: 50 epochs; ≥10000 samples: 20 epochs
LEARNING_RATE = 1e-4  # LoRA training:1e-4; Full training:1e-5
# LoRA lightweight training (MANDATORY for 12GB GPU, disable only for full training)
USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
# Image size (consistent with dataset, DO NOT MODIFY randomly)
CONDITION_SIZE = (256, 256)  # Dual condition image size
TARGET_SIZE = (512, 512)     # High-res target image size
# Output path (auto-created, no manual setup)
OUTPUT_DIR = "trained_model"  # Trained model save directory
LOG_DIR = "tensorboard_logs"  # Training log directory
# ============================================================================================
```
4. Start Training
Run the training script in the activated Python environment (ensure the GPU is in use and no other memory-intensive programs are running):
Start Training Command
```
# Basic training (recommended)
python train_controlnet_sketch_guided_sr.py

# Training with mixed precision (faster, less memory) - for PyTorch 2.0+
python -m torch.distributed.run --nproc_per_node=1 train_controlnet_sketch_guided_sr.py
```
Key Training Process Notes
 1.First run: The code will automatically download the SD1.5 base model (≈4GB) from Hugging Face, it is recommended to use a proxy for acceleration; the model will be cached locally after the first download, and no repeated download is required for subsequent training.
 
 2.Loss change: The training loss will gradually decrease and stabilize (normal range: from ~1.0 to <0.1). If the loss rises continuously, check the dataset filename pairing and image size.
 
 3.Log monitoring: Use TensorBoard to view real-time loss and learning rate changes (run the command below and access http://localhost:6006 in the browser):
```
tensorboard --logdir=tensorboard_logs
```
4.Breakpoint resume training: If the training is interrupted due to GPU downtime/power failure, re-run the training command directly—the code will automatically load the latest saved model and continue training.

5. epoch in the trained_model directory, including:
· LoRA weights (if USE_LORA=True): lora_step_xxx/ folder
· Full model weights: controlnet_step_xxx.pth / controlnet_epoch_xxx.pth
 ·WebUI compatible final model: controlnet_sketch_guided_sr_webui.pth (automatically exported after 
 training completion, for direct deployment)

5. Training Result Verification
After training, check the trained_model directory for the WebUI compatible model file controlnet_sketch_guided_sr_webui.pth—this file is the final training result and will be used for subsequent WebUI deployment.

Note: If the training is stopped early (not completed), select the model file with the smallest loss (e.g., controlnet_step_1000.pth) and rename it to controlnet_sketch_guided_sr_webui.pth for deployment.

# Deployment Steps

# Step 1: Place Model Files

Copy the trained model weight and configuration file to the ControlNet model directory of WebUI:
```
stable-diffusion-webui/
├─ extensions/
│  └─ sd-webui-controlnet/
│     └─ models/  # Paste both files here
│        ├─ controlnet_sketch_guided_sr_webui.pth
│        └─ controlnet_sketch_guided_sr_webui.yaml
```
Critical Note: The model weight and YAML file must have identical filenames (including extensions) to ensure proper loading.

# Step 2: Configure YAML File

Ensure the YAML file contains the following content (adapts to 6-channel dual-condition input):
```
model_type: controlnet
base_model: sd15
controlnet_version: 1.1
input_channels: 6  # 3 channels for low-res img + 3 for sketch
input_size: [256, 256]
target_size: [512, 512]
preprocessor: none
encoder_hidden_size: 768
down_block_types: ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
up_block_types: ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"]
```
# Step 3: Restart Stable Diffusion WebUI

Close any running WebUI instances and restart to load the new model. Use the following command to enable xFormers for acceleration:

# Windows
```
webui-user.bat --xformers
```
# Linux/Mac
```
./webui.sh --xformers
```
# Step 4: Verify Model Loading

1. Open Stable Diffusion WebUI in your browser (default: http://localhost:7860).

2. Navigate to the ControlNet tab.

3. In the Model dropdown menu, select controlnet_sketch_guided_sr_webui.

4. If the model appears in the dropdown, deployment is successful.

# Inference Guide (WebUI Operation)

Follow these steps to generate high-resolution images using the deployed model:

# Step 1: Prepare Input Images

- Low-resolution image: 256×256 (content to be upscaled)

- Sketch image: 256×256 (matches the structure of the low-res image)

# Step 2: Combine Dual-Condition Images

ControlNet only accepts a single input image. Combine the low-res image and sketch into one 256×256 image (WebUI built-in tool):

1. Go to the Extras tab → Image Editor.

2. Upload the low-res image, then paste the sketch to form a left-right combined image (size: 512×256).

3. Resize the combined image to 256×256 (use WebUI’s upscale function) → Save as combined_cond.png.

# Step 3: Configure ControlNet Parameters

In the ControlNet tab, set the following parameters:

- ✅ Enable: Check to activate ControlNet.

- ✅ Pixel Perfect: Check to auto-match input size.

- Model: Select controlnet_sketch_guided_sr_webui.

- Preprocessor: Select none (no built-in preprocessing needed).

- Upload Image: Upload the 256×256 combined image (combined_cond.png).

- Resize Mode: Select Crop and Resize.

- Control Weight: 0.8–1.0 (balance sketch constraint and detail generation).

- Starting Control Step: 0.0

- Ending Control Step: 0.8–1.0 (reduce to 0.8 for softer details).

# Step 4: Configure SD Generation Parameters

Set the following in the main WebUI interface for optimal results:

- Checkpoint: SelectStable Diffusion v1.5 (match the training base model).

- Width/Height: 512×512 (match the model’s target size).

- Steps: 20–30 (no significant improvement beyond 30 steps).

- CFG Scale: 5–8 (avoid over-exposure with values >8).

- Sampler: DPM++ 2M Karras (best for super-resolution tasks).

- Prompt (optional): Add style-specific keywords (e.g., high resolution, detailed, clear lines).

- Negative Prompt: blurry, low resolution, ugly, distorted, pixelated.

# Step 5: Generate High-Resolution Image

Click the Generate button. The output will be a 512×512 high-resolution image that preserves the content of the low-res input and the structure of the sketch.

# Troubleshooting

# 1. Model Not Found in WebUI

- Ensure the model weight and YAML file have identical filenames.

- Confirm files are placed in sd-webui-controlnet/models/ (not subdirectories).

- Restart WebUI to reload the model.

# 2. Blurry/ Distorted Output

- Verify the base model is Stable Diffusion v1.5.

- Increase Control Weight to 0.9–1.0.

- Ensure the combined input image is 256×256.

- Adjust CFG Scale to 6–8.

# 3. Overly Rigid (Sketch-Like) Output

- Reduce Control Weight to 0.7–0.8.

- Lower Ending Control Step to 0.8 (let SD optimize details in final steps).

# 4. Out-of-Memory Errors

- Enable xFormers (add --xformers to the WebUI launch command).

- Reduce batch size (set to 1 in WebUI settings if needed).

# Notes

- The model’s performance depends on the training dataset. For best results, use inputs consistent with the training data’s style.

- Avoid modifying the YAML file’s input_channels parameter (critical for dual-condition input).

- For faster inference, use GPUs with ≥12GB VRAM (e.g., RTX 3060/4060 or higher).

