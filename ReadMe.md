# Dissecting Image Generation with Stable Diffusion and ControlNet
## For my detailed observations and analysis follow this word document - [Dissecting Image Generation.docx](https://github.com/user-attachments/files/17393693/Dissecting.Image.Generation.docx)



## Project Overview

This project focuses on image generation using Stable Diffusion and ControlNet, guided by depth maps and Canny edges. The objective is to critique various conditioning techniques (depth maps, Canny edges) to produce the best possible output images. Additionally, this project explores the impact of different aspect ratios and generation latency.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup and Installation](#setup-and-installation)
3. [Tasks](#tasks)
    - [Task 1: Generating the Best Output Images](#task-1-generating-the-best-output-images)
    - [Task 2: Aspect Ratio Analysis](#task-2-aspect-ratio-analysis)
    - [Task 3: Generation Latency Analysis](#task-3-generation-latency-analysis)
4. [Results](#results)
5. [Conclusion](#conclusion)


---

## Setup and Installation

### Requirements

- Python 3.9 or later
- PyTorch 1.11.0+
- Transformers (for ControlNet)
- Diffusers
- OpenCV
- Matplotlib
- Skimage

```bash
# Install PyTorch and torchvision (for GPU version, make sure CUDA is installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Diffusers for Stable Diffusion and ControlNet
pip install diffusers transformers accelerate
pip install timm

# Install PIL (Pillow) for image processing
pip install pillow
pip install diffusers transformers accelerate

# Install OpenCV for Canny edge detection
pip install opencv-python

# Install Matplotlib (optional, for plotting images)
pip install matplotlib

```

## Models and Checkpoints

- **ControlNet Model**: `lllyasviel/control_v11f1p_sd15_depth`
- **Stable Diffusion Checkpoint**: `runwayml/stable-diffusion-v1-5`

---

## Tasks

### Task 1: Generating the Best Output Images

For this task, I used the provided depth maps and applied various conditioning techniques such as Canny edges to enhance the output. I experimented with different configurations to generate the "best" possible images.

#### Depth Map Only

#### Depth Map + Canny Edges

#### Observations Across 25, 50, and 100 Steps

- The number of inference steps significantly impacts both the quality and time taken to generate images.
- **25 steps**: Provided faster results, but the image quality was lower than 50 or 100 steps.
- **50 steps**: Achieved a balance between speed and image quality.
- **100 steps**: Produced the most detailed images but required much longer generation times.

#### Example output images at 25, 50, and 100 steps:

- *Image generated in 25, 50, 100 steps*
  
Prompt = "beautiful landscape, mountains in the background."
![download (21)](https://github.com/user-attachments/assets/b611213d-0c41-4aab-95d9-c536e92e1b15)
Prompt = "luxurious bedroom interior."
![download (23)](https://github.com/user-attachments/assets/7d0f5483-feb3-4351-9c0a-196dd10437af)
Prompt = "room with chair."
![download (25)](https://github.com/user-attachments/assets/df5f371a-993a-40b0-ade2-67952283aeb5)
Prompt = "house in the forest."
![download (26)](https://github.com/user-attachments/assets/e808235e-f2ab-4fd4-8f8d-769269bb2761)



---

### Task 2: Aspect Ratio Analysis

In this task, I explored the impact of aspect ratio on image quality by generating images in 1:1 and 4:3 aspect ratios.

#### Resized vs Cropped Images

- The depth map image `nocrop.png` was resized to both 1:1 and 4:3 aspect ratios.
- I also cropped the original image to these aspect ratios to compare the visual differences between resizing and cropping.

#### Observations

- **1:1 Aspect Ratio**: Maintains a balanced composition, but resizing may lead to distortion in some regions.
- **4:3 Aspect Ratio**: Provides a wider field of view but introduces some stretching when resized. Cropping yielded better results for preserving the visual quality.

#### Example images:

- *Resized to 1:1 aspect ratio and 4:3 aspect ratio*
- ![download (16)](https://github.com/user-attachments/assets/aee9a3a8-5594-4b2b-aec3-4a4ae63f6a1f)
- *Cropped to 1:1 aspect ratio and 4:3 aspect ratio*
- ![download (17)](https://github.com/user-attachments/assets/51ce52e3-d4c1-4976-a4a8-089e99c11578)




---

### Task 3: Generation Latency Analysis

This task evaluates the time taken to generate images and explores ways to reduce latency.

#### Observations on Latency

- **25 steps**: Faster but lower-quality images.
- **50 steps**: Provides a balance between speed and quality.
- **100 steps**: Best image quality, but the generation time is significantly longer.
- ### Prompt = "Majestic mountains at dusk, the peaks glowing in the setting sun, with a calm lake reflecting the sky and trees scattered across a serene valley."

  ![Screenshot 2024-10-16 011558](https://github.com/user-attachments/assets/0cd5f060-c9d4-4fe4-bf5a-e116fc455333)




#### Optimization Techniques

- **Model Quantization**: By converting the model to INT8 precision, we can speed up inference without significantly compromising image quality.
- **Scheduler Tuning**: We experimented with different schedulers (DDIM, LMS, Euler) to reduce inference time.
- **Low-Resolution Images**: Reducing image resolution (e.g., 256x256) can decrease the overall generation time.

#### Example of latency results:

```plaintext
Image generation (25 steps) took 5.42 seconds.
Image generation (50 steps) took 10.82 seconds.
Image generation (100 steps) took 20.57 seconds.
```
## Results

### Task 1 Results
- **Depth Map vs Depth Map + Canny Edges**: 
  - The combination of depth maps and Canny edges provides sharper, more detailed images compared to using depth maps alone.
- **Inference Steps**: 
  - Higher inference steps (50 or 100) provide better quality, but with a significant increase in generation time.

### Task 2 Results
- **Aspect Ratio Differences**: 
  - 1:1 vs 4:3 aspect ratios produced different compositions. The 1:1 aspect ratio provided a more balanced image, while 4:3 gave a broader view.
- **Resized vs Cropped**: 
  - Cropped images maintained visual quality better than resized images.

### Task 3 Results
- **Latency Optimization**: 
  - Reducing Inference steps and reducing image resolution helped reduce generation time, with a slight impact on image quality.
## Conclusion
The project demonstrates how depth maps and Canny edges can be effectively used to guide image generation with Stable Diffusion and ControlNet. Higher inference steps produce better quality images, but they also significantly increase the generation time. Using techniques like INT8 quantization and reducing image resolution can optimize the image generation process while still maintaining acceptable quality. Resizing and cropping images to different aspect ratios also provided interesting insights on composition and quality.
