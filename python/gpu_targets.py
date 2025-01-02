# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AOTRITON_SUPPORTED_GPUS = {
    "A100": "GPU_ARCH_NVIDIA_SM_80",  # Ampere
    "RTX3090": "GPU_ARCH_NVIDIA_SM_80",  # Ampere
}

AOTRITON_GPU_ARCH_TUNING_STRING = {
    "A100": "sm_80",  # Ampere
    "RTX3090": "sm_80",  # Ampere
}

AOTRITON_GPU_WARPSIZE = {
    "A100": 32,
    "RTX3090": 32,
}
