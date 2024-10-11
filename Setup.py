# Install PyTorch and torchvision (for GPU version, make sure CUDA is installed)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Diffusers for Stable Diffusion and ControlNet
!pip install diffusers transformers accelerate
!pip install timm

# Install PIL (Pillow) for image processing
!pip install pillow
! pip install diffusers transformers accelerate

# Install OpenCV for Canny edge detection
!pip install opencv-python

# Install Real-ESRGAN for image upscaling
!pip install realesrgan basicsr
!pip install realesrgan


# Install Matplotlib (optional, for plotting images)
!pip install matplotlib
