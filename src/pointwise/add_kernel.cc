// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/util.h>
#include <aotriton/config.h>
#include <aotriton/pointwise.h>
#include <aotriton/util.h>
#include <glog/logging.h>
#include <pointwise/shim.add_kernel.h>

namespace AOTRITON_NS::v2::pointwise {

cudaError_t add_kernel(
    T1 x_ptr, T1 y_ptr, T1 output_ptr, int32_t n_elements, AOTRITON_NS::Stream stream_wrap) {
  cudaError_t err;
  AddKernelContext context;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  context.grid_calculator = [](const AddKernelParams& params) -> dim3 {
    dim3 grid {
        AOTRITON_NS::cdiv<uint32_t>(params.x_ptr->size(0), params.BLOCK_SIZE),
        uint32_t(1),
        uint32_t(params.BLOCK_SIZE),
    };
    VLOG(3) << "Grid config : " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };

  AddKernelParams params = {.x_ptr = &x_ptr,
                            .y_ptr = &y_ptr,
                            .output_ptr = &output_ptr,
                            .n_elements = n_elements};

  err = context.lookup_optimal(params, arch);
  if (err != cudaSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  VLOG(1) << "GEMS ADD kernel launched... Result : " << err;
  return err;
}

}  // namespace AOTRITON_NS::v2::pointwise
