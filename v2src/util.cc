// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/util.h>
#include <string>
#include <unordered_map>
#include <string_view>

namespace AOTRITON_NS {

struct LazyArch {
  LazyArch(cudaDevice_t dev)
    : dev_(dev) {
  }
  operator GpuArch() {
    cudaDeviceProp_t prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, dev_);
    if (err != cudaSuccess)
      return GPU_ARCH_UNKNOWN;
    std::string_view arch(prop.gcnArchName);
    const auto colon = arch.find(':');
    if (colon != arch.npos) {
      arch = std::string_view(prop.gcnArchName, colon);
    }
    auto iter = string_to_arch.find(std::string(arch));
    if (iter == string_to_arch.end())
      return GPU_ARCH_UNKNOWN;
    return iter->second;
  }

private:
  cudaDevice_t dev_;
  static std::unordered_map<std::string, GpuArch> string_to_arch;
};

std::unordered_map<std::string, GpuArch> LazyArch::string_to_arch = {
  {"gfx90a", GPU_ARCH_AMD_GFX90A},
  {"gfx942", GPU_ARCH_AMD_GFX942},
  {"gfx1100", GPU_ARCH_AMD_GFX1100},
  {"gfx1101", GPU_ARCH_AMD_GFX1101},
};

GpuArch
getArchFromStream(cudaStream_t stream) {
  static std::unordered_map<cudaDevice_t, GpuArch> device_to_arch;
  cudaDevice_t dev;
  cudaError_t err = cudaStreamGetDevice(stream, &dev);
  if (err != cudaSuccess)
    return GPU_ARCH_UNKNOWN;
  LazyArch lazy(dev);
  device_to_arch.try_emplace(dev, lazy);
  return device_to_arch[dev];
}

template class TensorView<1>;
template class TensorView<2>;
template class TensorView<3>;
template class TensorView<4>;

} // namespace AOTRITON_NS
