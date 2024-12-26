#!/usr/bin/env python

"""
vector add test for cuda
a copy from 
https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py
"""

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, 
               y_ptr,  
               output_ptr, 
               n_elements, 
               BLOCK_SIZE: tl.constexpr,  
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
