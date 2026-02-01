Overview

This guide details the deployment process of a dual-condition ControlNet model (trained for sketch-guided image super-resolution) to Stable Diffusion WebUI. The model takes a low-resolution image and a corresponding sketch as inputs, then generates a high-resolution image while preserving content and structure.

Key Specifications:

- Base Model: Stable Diffusion v1.5

- ControlNet Version: 1.1

- Input: Low-resolution image (256×256) + Sketch (256×256) (combined as 6-channel input)

- Output: High-resolution image (512×512)

Prerequisites

Ensure the following are installed before deployment:

1. Stable Diffusion WebUI

2. ControlNet Extension for WebUI

  - Clone the extension to the stable-diffusion-webui/extensions/ directory.

3. Trained Model Files
        

  - Model weight: controlnet_sketch_guided_sr_webui.pth

  - Configuration file: controlnet_sketch_guided_sr_webui.yaml (must have the same name as the weight file)

Deployment Steps

Step 1: Place Model Files

Copy the trained model weight and configuration file to the ControlNet model directory of WebUI:

stable-diffusion-webui/
├─ extensions/
│  └─ sd-webui-controlnet/
│     └─ models/  # Paste both files here
│        ├─ controlnet_sketch_guided_sr_webui.pth
│        └─ controlnet_sketch_guided_sr_webui.yaml

Critical Note: The model weight and YAML file must have identical filenames (including extensions) to ensure proper loading.

Step 2: Configure YAML File

Ensure the YAML file contains the following content (adapts to 6-channel dual-condition input):

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

Step 3: Restart Stable Diffusion WebUI

Close any running WebUI instances and restart to load the new model. Use the following command to enable xFormers for acceleration:

# Windows
webui-user.bat --xformers

# Linux/Mac
./webui.sh --xformers

Step 4: Verify Model Loading

1. Open Stable Diffusion WebUI in your browser (default: http://localhost:7860).

2. Navigate to the ControlNet tab.

3. In the Model dropdown menu, select controlnet_sketch_guided_sr_webui.

4. If the model appears in the dropdown, deployment is successful.

Inference Guide (WebUI Operation)

Follow these steps to generate high-resolution images using the deployed model:

Step 1: Prepare Input Images

- Low-resolution image: 256×256 (content to be upscaled)

- Sketch image: 256×256 (matches the structure of the low-res image)

Step 2: Combine Dual-Condition Images

ControlNet only accepts a single input image. Combine the low-res image and sketch into one 256×256 image (WebUI built-in tool):

1. Go to the Extras tab → Image Editor.

2. Upload the low-res image, then paste the sketch to form a left-right combined image (size: 512×256).

3. Resize the combined image to 256×256 (use WebUI’s upscale function) → Save as combined_cond.png.

Step 3: Configure ControlNet Parameters

In the ControlNet tab, set the following parameters:

- ✅ Enable: Check to activate ControlNet.

- ✅ Pixel Perfect: Check to auto-match input size.

- Model: Selectcontrolnet_sketch_guided_sr_webui.

- Preprocessor: Select none (no built-in preprocessing needed).

- Upload Image: Upload the 256×256 combined image (combined_cond.png).

- Resize Mode: Select Crop and Resize.

- Control Weight: 0.8–1.0 (balance sketch constraint and detail generation).

- Starting Control Step: 0.0

- Ending Control Step: 0.8–1.0 (reduce to 0.8 for softer details).

Step 4: Configure SD Generation Parameters

Set the following in the main WebUI interface for optimal results:

- Checkpoint: Select Stable Diffusion v1.5 (match the training base model).

- Width/Height: 512×512 (match the model’s target size).

- Steps: 20–30 (no significant improvement beyond 30 steps).

- CFG Scale: 5–8 (avoid over-exposure with values >8).

- Sampler: DPM++ 2M Karras (best for super-resolution tasks).

- Prompt (optional): Add style-specific keywords (e.g., high resolution, detailed, clear lines).

- Negative Prompt: blurry, low resolution, ugly, distorted, pixelated.

Step 5: Generate High-Resolution Image

Click the Generate button. The output will be a 512×512 high-resolution image that preserves the content of the low-res input and the structure of the sketch.

Troubleshooting

1. Model Not Found in WebUI

- Ensure the model weight and YAML file have identical filenames.

- Confirm files are placed insd-webui-controlnet/models/ (not subdirectories).

- Restart WebUI to reload the model.

2. Blurry/ Distorted Output

- Verify the base model is Stable Diffusion v1.5.

- Increase Control Weight to 0.9–1.0.

- Ensure the combined input image is 256×256.

- Adjust CFG Scale to 6–8.

3. Overly Rigid (Sketch-Like) Output

- Reduce Control Weight to 0.7–0.8.

- Lower Ending Control Step to 0.8 (let SD optimize details in final steps).

4. Out-of-Memory Errors

- Enable xFormers (add --xformers to the WebUI launch command).

- Reduce batch size (set to 1 in WebUI settings if needed).

Notes

- The model’s performance depends on the training dataset. For best results, use inputs consistent with the training data’s style.

- Avoid modifying the YAML file’s input_channels parameter (critical for dual-condition input).

- For faster inference, use GPUs with ≥12GB VRAM (e.g., RTX 3060/4060 or higher).
