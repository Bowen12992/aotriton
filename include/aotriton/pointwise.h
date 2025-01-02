// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_POINTWISE_H
#define AOTRITON_V2_API_POINTWISE_H

#include <aotriton/config.h>

#include "cpp_tune.h"
#include "runtime.h"
#include "util.h"

namespace AOTRITON_NS::v2::pointwise {

cudaError_t check_gpu(AOTRITON_NS::Stream stream);

using T4 = AOTRITON_NS::TensorView<4>;
using T2 = AOTRITON_NS::TensorView<2>;
using T1 = AOTRITON_NS::TensorView<1>;
using T0 = AOTRITON_NS::TensorView<0>;

cudaError_t add_kernel(T1 x_ptr, T1 y_ptr, T1 output_ptr, int32_t n_elements, AOTRITON_NS::Stream stream);

}  // namespace AOTRITON_NS::v2::pointwise

#endif
