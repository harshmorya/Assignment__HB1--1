import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Function to normalize depth maps
def normalize_depth_map(depth_map):
    """Normalize depth map to a range of [0, 255]."""
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    depth_map_normalized = (depth_map - min_val) / (max_val - min_val) * 255
    return Image.fromarray(depth_map_normalized.astype(np.uint8))

# Load ControlNet and Stable Diffusion models in FP32 or FP16
control_net = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
stable_diffusion = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=control_net, torch_dtype=torch.float16
)

# Configure scheduler for faster generation
stable_diffusion.scheduler = DDIMScheduler.from_config(stable_diffusion.scheduler.config)
stable_diffusion.to("cuda")

# Function to perform INT8 quantization manually
def quantize_model_manually(pipe):
    print("Performing manual INT8 quantization on U-Net layers...")
    for name, module in pipe.unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(module, inplace=True)
            torch.quantization.convert(module, inplace=True)
    print("Quantization completed.")
    return pipe

# Quantize the U-Net model
stable_diffusion = quantize_model_manually(stable_diffusion)

# Start image generation process
selected_prompt =  input("Please enter your desired prompt: ")
depth_map_path = '/content/drive/MyDrive/Images/6.png'  # Replace with your depth map path
depth_map = Image.open(depth_map_path).convert("RGB")

# Start image generation process
start_time = time.time()

# Generate image with depth map
output_image = stable_diffusion(prompt=selected_prompt, image=depth_map, generator=torch.manual_seed(12345), num_inference_steps=25).images[0]

# Save and display the generated image
output_image.save("generated_image.png")
plt.imshow(output_image)
plt.axis('off')  # Disable the axis/ruler
plt.title("Generated Image with Depth Map (INT8 Quantized Model)")
plt.show()

# Time measurement for image generation
end_time = time.time()
print(f"Image generation (quantized) took {end_time - start_time:.2f} seconds.")


