import pyaotriton
import torch


add = pyaotriton.v2.pointwise.add_kernel
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

