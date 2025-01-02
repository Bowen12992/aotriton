# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .add import add

SOURCE_FILE = "tritonsrc/add_kernel.py"
kernels = [
    add("add_kernel", SOURCE_FILE),
]
