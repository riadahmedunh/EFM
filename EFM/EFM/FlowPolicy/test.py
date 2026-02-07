import torch
import torchvision
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"Torchvision CUDA: {torchvision.__variant__.get('CUDA', 'Not Found')}") 
# Or simply try to load an op:
from torchvision.ops import nms
print("Success: Extension ops loaded!")