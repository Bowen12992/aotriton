# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# import itertools
from ._common import PointwiseKernel

class add(PointwiseKernel):
    ARGUMENTS = [
        'x_ptr',
        'y_ptr',
        'output_ptr',
        'n_elements',
        'BLOCK_SIZE',  # constexpr
    ]
    TENSOR_STRIDE_INPUTS = {}
    TYPE_CHOICES = {
        frozenset(['x_ptr', 'y_ptr', 'output_ptr']) : PointwiseKernel.MAIN_DATATYPES,
        frozenset(['n_elements']) : ['i32'],
    }

    PERF_CHOICES = {
        frozenset(['BLOCK_SIZE']) : [32],
    }
    
    PARTIALLY_TUNED_FUNCTIONALS = []
    
    AUTOTUNE_KEYS = {}
    
    SHIM_KERNEL_NAME = 'add_kernel'
    
    TENSOR_RANKS = {
        '_default' : 1,
    }