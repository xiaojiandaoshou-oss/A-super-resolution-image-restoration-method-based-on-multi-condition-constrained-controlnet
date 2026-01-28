# A-super-resolution-image-restoration-method-based-on-multi-condition-constrained-controlnet
Overview
A ControlNet training scheme that utilizes dual-condition input (low-resolution original image + corresponding line drawing ) to guide SD in generating high-resolution original images
The core task of this scheme is to enable ControlNet to learn the mapping relationship from "low-resolution original image + line drawing or semantic segmentation map" to "high-resolution original image", while maintaining the characteristics of compatibility with SD1.5+WebUI, lightweight training (LoRA), and memory-friendly performance. Training can be conducted with a graphics card memory of ≥12G.
Overview
This guide details the deployment process of a dual-condition ControlNet model (trained for sketchor or semantic segmentation image-guided image super-resolution) to Stable Diffusion WebUI. The model takes a low-resolution image and a corresponding sketch as inputs, then generates a high-resolution image while preserving content and structure.

Key Specifications:

- Base Model: Stable Diffusion v1.5

- ControlNet Version: 1.1

- Input: Low-resolution image (256×256) + Sketch (256×256) (combined as 6-channel input)

- Output: High-resolution image (512×512)
