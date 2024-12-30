import pyaotriton
import torch

from pyaotriton import T1, T2, T4, DType, Stream, cudaError_t

def cast_dtype(dtype):
    assert not dtype.is_complex
    bits = dtype.itemsize * 8
    if dtype.is_floating_point:
        maintype = 'Float' if 'bfloat' not in str(dtype) else 'BFloat'
    else:
        maintype = 'Int' if 'uint' not in str(dtype) else 'UInt'
    typename = f'k{maintype}{bits}'
    return getattr(DType, typename)

def mk_aotensor(q):
    return T1(q.data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

add = pyaotriton.v2.pointwise.add_kernel
torch.manual_seed(0)
size = 984320000
x = torch.ones(size, device='cuda')
y = torch.ones(size, device='cuda')
z = torch.zeros(size, device='cuda')

for i in range(100):
    add(mk_aotensor(x),mk_aotensor(y),mk_aotensor(z), size, Stream())
print("----------------------------")
print(x)
print(y)
print(z)



