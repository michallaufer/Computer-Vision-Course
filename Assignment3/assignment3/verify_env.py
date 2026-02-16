
import torch
import torchvision
import numpy as np
import matplotlib
import scipy
import imageio
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Scipy: {scipy.__version__}")
print(f"Imageio: {imageio.__version__}")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))

print("Environment verification successful!")
