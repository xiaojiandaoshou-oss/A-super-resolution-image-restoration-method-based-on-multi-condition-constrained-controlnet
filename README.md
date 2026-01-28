# A-super-resolution-image-restoration-method-based-on-multi-condition-constrained-controlnet
# Overview
This guide details the deployment process of a dual-condition ControlNet model (trained for sketchor or semantic segmentation image-guided image super-resolution) to Stable Diffusion WebUI. The model takes a low-resolution image and a corresponding sketch as inputs, then generates a high-resolution image while preserving content and structure.

Key Specifications:

- Base Model: Stable Diffusion v1.5

- ControlNet Version: 1.1

- Input: Low-resolution image (256×256) + Sketch (256×256) (combined as 6-channel input)

- Output: High-resolution image (512×512)
