// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_POINTWISE_H
#define AOTRITON_V2_API_POINTWISE_H

#include <aotriton/config.h>
#include "runtime.h"
#include "util.h"
#include "cpp_tune.h"

namespace AOTRITON_NS::v2::pointwise {

cudaError_t
check_gpu(AOTRITON_NS::Stream stream);

using T4 = AOTRITON_NS::TensorView<4>;
using T2 = AOTRITON_NS::TensorView<2>;
using T1 = AOTRITON_NS::TensorView<1>;
using T0 = AOTRITON_NS::TensorView<0>;

struct FwdExtraArguments : public CppTune {
};

struct BwdExtraArguments {
#if AOTRITON_BUILD_FOR_TUNING
  FwdExtraArguments dkdv, dqdb;
#endif
};

cudaError_t
attn_fwd(T4 q, // batch_size x num_heads x seqlen_q x head_size
         T4 k, // batch_size x num_heads x seqlen_k x head_size
         T4 v, // batch_size x num_heads x seqlen_k x head_size
         T4 b, // batch_size x num_heads x seqlen_k x head_size
         float sm_scale,
         T2 softmax_lse,
         T4 Out, // batch_size x num_heads x seqlen_q x head_size
         float dropout_p,
         T0 philox_seed,
         T0 philox_offset1,
         int64_t philox_offset2,
         T0 philox_seed_output,
         T0 philox_offset_output,
         T4 encoded_softmax,
         bool is_causal,
         AOTRITON_NS::Stream stream,
         FwdExtraArguments* extargs = nullptr);



cudaError_t
debug_fill_dropout_rng(T4 r,
                       uint64_t philox_seed,
                       uint64_t philox_offset,
                       AOTRITON_NS::Stream stream);

cudaError_t
debug_fill_dropout_rng_tensor(T4 r,
                              T0 philox_seed,
                              T0 philox_offset,
                              AOTRITON_NS::Stream stream);

} // AOTRITON_NS::v2::flash

#endif
