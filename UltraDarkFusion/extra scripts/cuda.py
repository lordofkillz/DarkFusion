import torch
import cv2

# Check for CUDA availability in PyTorch
if torch.cuda.is_available():
    print("PyTorch - CUDA is available!")
    device = torch.device("cuda:0")  # Use the first available GPU
    print(f"PyTorch - Number of GPUs available: {torch.cuda.device_count()}")
    print(f"PyTorch - Name of current GPU: {torch.cuda.get_device_name(device)}")
else:
    print("PyTorch - CUDA is not available.")
    device = torch.device("cpu")  # Use CPU

# Check for CUDA support in OpenCV
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("OpenCV - CUDA is available!")
    print(f"OpenCV - Number of CUDA enabled devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
else:
    print("OpenCV - CUDA is not available.")

# Check for OpenCL support in OpenCV
if cv2.ocl.haveOpenCL():
    print("OpenCV - OpenCL is available!")
else:
    print("OpenCV - OpenCL is not available.")
