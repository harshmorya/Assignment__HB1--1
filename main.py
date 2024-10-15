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


def crop_image(image, aspect_ratio):
    """Crop image to specified aspect ratio."""
    width, height = image.size

    if aspect_ratio == "1:1":
        new_size = min(width, height)
        left = (width - new_size) // 2
        top = (height - new_size) // 2
        right = (width + new_size) // 2
        bottom = (height + new_size) // 2
    elif aspect_ratio == "4:3":
        target_width = (height * 4) // 3
        if target_width > width:
            new_width = width
            new_height = (width * 3) // 4
            top = (height - new_height) // 2
            left = 0
            right = new_width
            bottom = top + new_height
        else:
            new_width = target_width
            new_height = height
            left = (width - new_width) // 2
            top = 0
            right = left + new_width
            bottom = height
    else:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")

    return image.crop((left, top, right, bottom))


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


    # Crop images
    cropped_square = crop_image(depth_image, "1:1")
    cropped_43 = crop_image(depth_image, "4:3")

    
    
    output_square_cropped = stable_diffusion(prompt=selected_prompt, image=cropped_square, num_inference_steps=50).images[0]
    output_43_cropped = stable_diffusion(prompt=selected_prompt, image=cropped_43, num_inference_steps=50).images[0]

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))

    

    axes[0].imshow(output_square_cropped)
    axes[0].set_title("Cropped Square (1:1)")
    axes[0].axis('off')

    axes[1].imshow(output_43_cropped)
    axes[1].set_title("Cropped 4:3")
    axes[1].axis('off')

    plt.show()



else:
    depth_image_resized = depth_image.resize((512, 512))
    output_image = stable_diffusion(prompt=selected_prompt, image=depth_image_resized, generator=seed, num_inference_steps=50)

    output_image.images[0].save("generated_image.png")
    plt.imshow(Image.open("generated_image.png"))
    plt.axis('off')  # Disable the axis/ruler
    plt.show()

# If using canny edges, generate canny edges from the depth image and also generate another image
if use_canny:
        # Function to normalize depth maps
    def normalize_depth_map(depth_map):
        """Normalize depth map to a range of [0, 255]."""
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        depth_map_normalized = (depth_map - min_val) / (max_val - min_val) * 255
        return Image.fromarray(depth_map_normalized.astype(np.uint8))

    # Function to generate Canny edges
    def generate_canny_edges(image, low_threshold=100, high_threshold=600):
        """Generate Canny edges from the given image."""
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image_gray, low_threshold, high_threshold)
        edges_image = Image.fromarray(edges).convert("RGB")  # Convert edges back to RGB for blending
        return edges_image

    # Function for overlaying Canny edges on original image
    def overlay_canny_on_image(original_image, canny_edges, alpha=0.4):
        """Overlay Canny edges on the original image using alpha blending."""
        combined_image = Image.blend(original_image, canny_edges, alpha)
        return combined_image

    # Function to apply morphological operations
    def apply_morphological_operations(edges):
        """Apply dilation to thicken the edges."""
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(np.array(edges.convert("L")), kernel, iterations=1)
        return Image.fromarray(dilated_edges).convert("RGB")

    # Function for multi-scale edge detection
    def multi_scale_canny(image):
        """Apply multi-scale Canny edge detection and combine the results."""
        edges_1 = generate_canny_edges(image, low_threshold=50, high_threshold=150)
        edges_2 = generate_canny_edges(image, low_threshold=100, high_threshold=200)
        combined_edges = Image.blend(edges_1, edges_2, 0.5)
        return combined_edges

    # Function for combining with color information
    def combine_with_color_info(image, canny_edges):
        """Combine Canny edges with the original image by coloring the edges."""
        image_array = np.array(image)
        edges_array = np.array(canny_edges.convert("L"))
        edges_colored = cv2.applyColorMap(edges_array, cv2.COLORMAP_JET)
        combined_colored = cv2.addWeighted(image_array, 0.9, edges_colored, 0.8, 0)
        return Image.fromarray(combined_colored)

    # Function to perform image segmentation using Canny edges
    def segment_image(edges):
        """Use contours from Canny edges for segmentation."""
        edges_gray = np.array(edges.convert("L"))
        contours, _ = cv2.findContours(edges_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        segmented = np.zeros_like(edges_gray)
        cv2.drawContours(segmented, contours, -1, (255), thickness=cv2.FILLED)
        return Image.fromarray(segmented).convert("RGB")

    
    
    depth_map = Image.open(selected_depth_path).convert("RGB")

    # Apply all the techniques in sequence with the same prompt
    print("Applying all techniques...")

    # 1. Generate and display image with Canny Edges overlayed
    canny_edges = generate_canny_edges(depth_map)
    image_with_canny_overlay = overlay_canny_on_image(depth_map, canny_edges)
    print("Image with Canny Edges Overlay:")
    generated_image_canny_overlay = stable_diffusion(prompt=selected_prompt, image=image_with_canny_overlay, num_inference_steps=50).images[0]
    plt.imshow(generated_image_canny_overlay)
    plt.axis('off')
    plt.title("Generated Image with Canny Edges Overlay")
    plt.show()

    # 2. Combine depth map with Canny edges and generate image
    image_combined = Image.blend(depth_map, canny_edges, alpha=0.3)
    print("Image with Depth Map + Canny Edges Combined:")
    generated_image_combined = stable_diffusion(prompt=selected_prompt, image=image_combined, num_inference_steps=50).images[0]
    plt.imshow(generated_image_combined)
    plt.axis('off')
    plt.title("Generated Image with Depth Map + Canny Edges")
    plt.show()



    # 6. Combine Canny edges with color information and generate image
    colored_canny_image = combine_with_color_info(depth_map, canny_edges)
    print("Colored Canny Edges:")
    generated_image_colored = stable_diffusion(prompt=selected_prompt, image=colored_canny_image, num_inference_steps=50).images[0]
    plt.imshow(generated_image_colored)
    plt.axis('off')
    plt.title("Generated Image with Colored Canny Edges")
    plt.show()

depth_image_resized = depth_image.resize((512, 512))
# Time measurement for image generation
end_time = time.time()
print(f"Image generation took {end_time - start_time:.2f} seconds.")

# Compare performance for 25 vs. 50 steps (you can add more if needed)
print("\nComparing image generation times for 25, 50, and 100 steps...")
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

    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
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
