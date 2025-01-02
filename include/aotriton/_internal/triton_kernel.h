// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#include <aotriton/config.h>

#include <memory>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../runtime.h"

namespace AOTRITON_NS {

class PackedKernel;

class TritonKernel {
 public:
  using Essentials = std::tuple<const void*, int, dim3>;

  TritonKernel(const char* package_path, const char* stem_name);

  cudaError_t invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, cudaStream_t stream);

  void clear_decompressed_image();

 private:
  std::tuple<cudaFunction_t, cudaError_t> load_for_device(int device_id, const char* kernel_name);
  cudaFunction_t cfind_function(int device_id) const;

  const char* package_path_ = nullptr;
  const char* stem_name_ = nullptr;
  size_t image_size_ = 0;
  struct DeviceFunction {
    DeviceFunction(int device_id_, CUmodule mod_, cudaFunction_t func_);
    ~DeviceFunction();
    int device_id = -1;
    CUmodule mod = nullptr;
    cudaFunction_t func = nullptr;
  };
  std::unordered_map<int, DeviceFunction> funcache_;
  std::shared_mutex funcache_mutex_;

  int shared_memory_size_ = 0;
  dim3 block_ {256, 1, 1};
  const void* kernel_image_ = nullptr;
  bool kernel_loaded_ = false;
  void decompress_kernel();
  std::shared_ptr<PackedKernel> packed_kernel_ = nullptr;
  std::shared_mutex packedkernel_mutex_;
};

}  // namespace AOTRITON_NS

#endif
