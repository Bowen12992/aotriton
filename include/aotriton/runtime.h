// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_RUNTIME_H
#define AOTRITON_V2_API_RUNTIME_H

#include <aotriton/config.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace AOTRITON_NS {

// This is not a class for stream management (at least for now), but a way to
// make sure AOTriton APIs can have python bindings with pybind11
template <typename DeviceStreamType>
class StreamTemplate {
 public:
  StreamTemplate() : stream_(nullptr) {
  }
  StreamTemplate(DeviceStreamType stream) : stream_(stream) {
  }
  DeviceStreamType native() const {
    return stream_;
  }

 private:
  DeviceStreamType stream_;
};

using Stream = StreamTemplate<cudaStream_t>;

}  // namespace AOTRITON_NS

#endif
