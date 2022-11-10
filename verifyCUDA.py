import torch

# CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model created by NVIDIA.

torch.cuda.is_available()
# >> True

torch.cuda.device_count()
# >> 1

torch.cuda.device(torch.cuda.current_device())
# >> <torch.cuda.device at 0x193ddee57c0>

torch.cuda.get_device_name(torch.cuda.current_device())
# >> 'NVIDIA GeForce MX150'