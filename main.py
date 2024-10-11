import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, PNDMScheduler
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import mean_squared_error, structural_similarity as ssim
import torchvision.transforms as transforms

# Image transformation for the model
def get_image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load the MiDaS depth estimation model
def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.eval()
    return model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Function to normalize depth maps
def normalize_depth(depth_map):
    """Normalize a depth map to a range of [0, 1]."""
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    return (depth_map - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(depth_map)

# Estimate depth from an image
def estimate_image_depth(image_path, model, transform):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    input_tensor = transform(img_resized).unsqueeze(0).to(next(model.parameters()).device)

    with torch.no_grad():
        depth_prediction = model(input_tensor)

    return depth_prediction.squeeze().cpu().numpy()

# Resize the image to a different aspect ratio
def resize_image_to_aspect_ratio(image_path, aspect_ratio):
    """Resize the image to the desired aspect ratio."""
    image = Image.open(image_path)

    original_width, original_height = image.size
    if aspect_ratio == "1:1":  # Square
        new_size = (512, 512)
    elif aspect_ratio == "16:9":  # Widescreen
        new_size = (960, 540)
    elif aspect_ratio == "4:3":  # Standard
        new_size = (640, 480)
    elif aspect_ratio == "3:2":  # Classic
        new_size = (768, 512)
    else:
        raise ValueError(f"Aspect ratio {aspect_ratio} is not supported.")

    return image.resize(new_size)

# Function to generate canny edges
def generate_canny_edges(image, low_threshold=2, high_threshold=15):
    """Generate canny edges from the given image."""
    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, low_threshold, high_threshold)
    edges_image = Image.fromarray(edges).convert("RGB")  # Convert edges back to RGB to be used in the pipeline
    return edges_image

def calculate_metrics(original_depth, generated_depth):
    """Calculate MSE and SSIM between original and generated depth maps."""
    mse_value = mean_squared_error(original_depth, generated_depth)
    ssim_value = ssim(original_depth, generated_depth, data_range=generated_depth.max() - generated_depth.min())
    return mse_value, ssim_value


# Function to allow the user to choose the scheduler
def choose_scheduler(stable_diffusion_pipeline):
    print("\nAvailable Schedulers:")
    print("1. DDIMScheduler - High quality, very fast")
    print("2. PNDMScheduler - Moderate to high quality, moderate speed")
    print("3. LMSDiscreteScheduler - High quality, moderate speed")
    print("4. EulerDiscreteScheduler - Moderate to high quality, fast")
    scheduler_choice = int(input("Select the scheduler number: "))
    
    if scheduler_choice == 1:
        stable_diffusion_pipeline.scheduler = DDIMScheduler.from_config(stable_diffusion_pipeline.scheduler.config)
    elif scheduler_choice == 2:
        stable_diffusion_pipeline.scheduler = PNDMScheduler.from_config(stable_diffusion_pipeline.scheduler.config)
    elif scheduler_choice == 3:
        stable_diffusion_pipeline.scheduler = LMSDiscreteScheduler.from_config(stable_diffusion_pipeline.scheduler.config)
    elif scheduler_choice == 4:
        stable_diffusion_pipeline.scheduler = EulerDiscreteScheduler.from_config(stable_diffusion_pipeline.scheduler.config)
    else:
        print("Invalid selection. Defaulting to DDIMScheduler.")
        stable_diffusion_pipeline.scheduler = DDIMScheduler.from_config(stable_diffusion_pipeline.scheduler.config)

    return stable_diffusion_pipeline
# List of depth maps (image and .npy files)
depth_map_paths = [
    "/content/drive/MyDrive/Images/1.png",
    "/content/drive/MyDrive/Images/2.png",
    "/content/drive/MyDrive/Images/2_nocrop.png",
    "/content/drive/MyDrive/Images/3.png",
    "/content/drive/MyDrive/Images/4.png",
    "/content/drive/MyDrive/Images/5.png",
    "/content/drive/MyDrive/Images/6.png",
    "/content/drive/MyDrive/Images/7.png",
    "/content/drive/MyDrive/Images/6.npy",
    "/content/drive/MyDrive/Images/7.npy"
]

# Prompt selection: Let the user enter the prompt at runtime
selected_prompt = input("Please enter your desired prompt: ")

# Depth map selection
print("\nChoose a depth map (image or .npy file):")
for idx, depth_path in enumerate(depth_map_paths, start=1):
    print(f"{idx}: {depth_path}")
depth_index = int(input("Select depth map number: ")) - 1
selected_depth_path = depth_map_paths[depth_index]

# Ask if the user wants to use canny edges
use_canny = input("Do you want to use canny edges? (yes/no): ").lower().strip() == 'yes'

# Load and process depth map
if selected_depth_path.endswith(".npy"):
    depth_map = np.load(selected_depth_path)
    normalized_depth = normalize_depth(depth_map)
    depth_image = Image.fromarray((normalized_depth * 255).astype(np.uint8)).convert("RGB")
else:
    depth_image = Image.open(selected_depth_path).convert("RGB")

# Load ControlNet and Stable Diffusion models
control_net = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
stable_diffusion = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=control_net, torch_dtype=torch.float16
)

# Configure scheduler for faster inference
stable_diffusion = choose_scheduler(stable_diffusion)
stable_diffusion.to("cuda")
seed = torch.manual_seed(12345)

# Start image generation process
start_time = time.time()

# Save and display the image generated using only the depth map
if selected_depth_path.endswith("nocrop.png"):
    non_square_depth = Image.open(selected_depth_path).convert("RGB").resize((940, 564))

    square_depth = resize_image_to_aspect_ratio(selected_depth_path,"1:1")
    depth43 = resize_image_to_aspect_ratio(selected_depth_path,"4:3")

    output_non_square = stable_diffusion(prompt=selected_prompt, image=non_square_depth, generator=seed, num_inference_steps=25)
    output_square = stable_diffusion(prompt=selected_prompt, image=square_depth, generator=seed, num_inference_steps=25)
    output_43 = stable_diffusion(prompt=selected_prompt, image=depth43, generator=seed, num_inference_steps=25)

    output_non_square.images[0].save("generated_non_square.png")
    output_square.images[0].save("generated_square.png")
    output_43.images[0].save("generated_43.png")

    # Display generated images
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(Image.open("generated_non_square.png"))
    axes[0].set_title("Non-Square Image")
    axes[0].axis('off')  # Disable the axis/ruler

    axes[1].imshow(Image.open("generated_square.png"))
    axes[1].set_title("Square Image")
    axes[1].axis('off')

    axes[2].imshow(Image.open("generated_43.png"))
    axes[2].set_title("4:3")
    axes[2].axis('off')  # Disable the axis/ruler
    plt.show()
else:
    depth_image_resized = depth_image.resize((512, 512))
    output_image = stable_diffusion(prompt=selected_prompt, image=depth_image_resized, generator=seed, num_inference_steps=25)

    output_image.images[0].save("generated_image.png")
    plt.imshow(Image.open("generated_image.png"))
    plt.axis('off')  # Disable the axis/ruler
    plt.show()

# If using canny edges, generate canny edges from the depth image and also generate another image
if use_canny:
    depth_image_with_canny = generate_canny_edges(depth_image)
    depth_image_with_canny_resized = depth_image_with_canny.resize((512, 512))
    output_image_canny = stable_diffusion(prompt=selected_prompt, image=depth_image_with_canny_resized, generator=seed, num_inference_steps=10)

    # Save and display the image generated using canny edges
    output_image_canny.images[0].save("generated_image_canny.png")
    plt.imshow(Image.open("generated_image_canny.png"))
    plt.title("Generated Image (Canny Edges)")
    plt.axis('off')  # Disable the axis/ruler
    plt.show()

depth_image_resized = depth_image.resize((512, 512))
# Time measurement for image generation
end_time = time.time()
print(f"Image generation took {end_time - start_time:.2f} seconds.")

# Compare performance for 25 vs. 50 steps (you can add more if needed)
print("\nComparing image generation times for 25 and 50 steps...")
time_25_steps = end_time - start_time

# Depth map comparison for non-square image
midas_model = load_depth_model()
transform = get_image_transform()

if selected_depth_path.endswith("nocrop.png"):
    # Estimate depth map for the non-square generated image
    generated_depth_map_non_square = estimate_image_depth("generated_non_square.png", midas_model, transform)

    if generated_depth_map_non_square is not None:
        input_depth_map_non_square = np.asarray(non_square_depth)
        if input_depth_map_non_square.shape[2] == 3:
            input_depth_map_non_square = input_depth_map_non_square[:, :, 0]  # Convert to single channel

        generated_depth_map_non_square_resized = cv2.resize(generated_depth_map_non_square,
            (input_depth_map_non_square.shape[1], input_depth_map_non_square.shape[0]))

        input_depth_map_non_square_normalized = normalize_depth(input_depth_map_non_square)
        generated_depth_map_non_square_normalized = normalize_depth(generated_depth_map_non_square_resized)

        # Display the input and generated depth maps side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_depth_map_non_square_normalized, cmap="gray")
        axes[0].set_title("Input Depth Map (Non-Square)")

        axes[1].imshow(generated_depth_map_non_square_normalized, cmap="gray")
        axes[1].set_title("Generated Depth Map (Non-Square)")
        plt.show()

        # Calculate metrics
        mse_non_square, ssim_non_square = calculate_metrics(input_depth_map_non_square_normalized, generated_depth_map_non_square_normalized)
        print(f"MSE (Non-Square): {mse_non_square:.4f}")
        print(f"SSIM (Non-Square): {ssim_non_square:.4f}")

# Depth map comparison for square image
generated_depth_map = estimate_image_depth("generated_image.png", midas_model, transform)

if generated_depth_map is not None:
    input_depth_map = np.asarray(depth_image_resized)
    if input_depth_map.shape[2] == 3:
        input_depth_map = input_depth_map[:, :, 0]  # Convert to single channel

    generated_depth_map_resized = cv2.resize(generated_depth_map, (input_depth_map.shape[1], input_depth_map.shape[0]))

    input_depth_map_normalized = normalize_depth(input_depth_map)
    generated_depth_map_normalized = normalize_depth(generated_depth_map_resized)

    # Display the input and generated depth maps side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_depth_map_normalized, cmap="gray")
    axes[0].set_title("Input Depth Map (Square)")

    axes[1].imshow(generated_depth_map_normalized, cmap="gray")
    axes[1].set_title("Generated Depth Map (Square)")
    plt.show()

    # Calculate metrics
    mse_square, ssim_square = calculate_metrics(input_depth_map_normalized, generated_depth_map_normalized)
    print(f"MSE (Square): {mse_square:.4f}")
    print(f"SSIM (Square): {ssim_square:.4f}")

# Generate images using 50 steps
if selected_depth_path.endswith("nocrop.png"):
    output_non_square_50 = stable_diffusion(prompt=selected_prompt, image=non_square_depth, generator=seed, num_inference_steps=50)
    output_square_50 = stable_diffusion(prompt=selected_prompt, image=square_depth, generator=seed, num_inference_steps=50)

    output_non_square_50.images[0].save("generated_non_square_50_steps.png")
    output_square_50.images[0].save("generated_square_50_steps.png")

    # Display the results for 50 steps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(Image.open("generated_non_square.png"))
    axes[0, 0].set_title("Original Aspect ratio with 25 Steps")
    axes[0, 0].axis('off')  # Disable the axis/ruler

    axes[0, 1].imshow(Image.open("generated_square.png"))
    axes[0, 1].set_title("1:1 with 25 Steps")
    axes[0, 1].axis('off')  # Disable the axis/ruler

    axes[1, 0].imshow(Image.open("generated_non_square_50_steps.png"))
    axes[1, 0].set_title("Original Aspect ratio with 50 Steps")
    axes[1, 0].axis('off')  # Disable the axis/ruler

    axes[1, 1].imshow(Image.open("generated_square_50_steps.png"))
    axes[1, 1].set_title("1:1 with 50 Steps")
    axes[1, 1].axis('off')  # Disable the axis/ruler

    plt.show()
else:

    start_time_50 = time.time()
    output_50 = stable_diffusion(prompt=selected_prompt, image=depth_image_resized, generator=seed, num_inference_steps=50)
    output_50.images[0].save("generated_image_50_steps.png")
    end_time_50 = time.time()

    time_50_steps = end_time_50 - start_time_50

    start_time_100 = time.time()
    output_100 = stable_diffusion(prompt=selected_prompt, image=depth_image_resized, generator=seed, num_inference_steps=100)
    output_100.images[0].save("generated_image_100_steps.png")
    end_time_100 = time.time()

    time_100_steps = end_time_100 - start_time_100

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(Image.open("generated_image.png"))
    axes[0].set_title("25 Steps")
    axes[0].axis('off')  # Disable the axis/ruler

    axes[1].imshow(Image.open("generated_image_50_steps.png"))
    axes[1].set_title("50 Steps")
    axes[1].axis('off')  # Disable the axis/ruler

    axes[2].imshow(Image.open("generated_image_100_steps.png"))
    axes[2].set_title("100 Steps")
    axes[2].axis('off')

    plt.show()

    # Report times
    print(f"Image generation (25 steps) took {time_25_steps:.2f} seconds.")
    print(f"Image generation (50 steps) took {time_50_steps:.2f} seconds.")
    print(f"Image generation (100 steps) took {time_100_steps:.2f} seconds.")


