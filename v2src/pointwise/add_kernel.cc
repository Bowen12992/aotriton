// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/pointwise.h>
#include <aotriton/util.h>
#include <pointwise/shim.add_kernel.h>

namespace AOTRITON_NS::v2::pointwise {

cudaError_t
add_kernel(T1 x_ptr, T1 y_ptr, T1 output_ptr, int32_t n_elements, AOTRITON_NS::Stream stream_wrap) {
  cudaError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  auto grid_calculator = [](const AddKernelParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.x_ptr->size(0), params.BLOCK_SIZE),
      uint32_t(1),
      uint32_t(params.BLOCK_SIZE),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };

  AddKernelParams params = {
    .x_ptr = &x_ptr,
    .y_ptr = &y_ptr,
    .output_ptr = &output_ptr,
    .n_elements = n_elements
  };

  AddKernelContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, arch);
  if (err != cudaSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

}
