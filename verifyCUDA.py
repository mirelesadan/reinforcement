"""
Verify that your system has a CUDA-enabled device.
Print the following lines:
"""

import torch

# CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model created by NVIDIA.

torch.cuda.is_available()
# Should print: 
# >> True

torch.cuda.device_count()
# Should print, at least: 
# >> 1

torch.cuda.device(torch.cuda.current_device())
# Should print something with the format:
# >> <torch.cuda.device at 0x123abcd45d6>

torch.cuda.get_device_name(torch.cuda.current_device())
# Should print something such as:
# >> 'NVIDIA GeForce XX700'
