# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .flash import kernels as flash_kernels

from .pointwise import kernels as pointwise_kernels
kernels = pointwise_kernels + flash_kernels