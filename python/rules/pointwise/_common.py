# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ...autotune_binning import BinningExact, BinningLessOrEqual
from ...autotune_config import Config
from ...kernel_desc import KernelDescription, get_possible_types, select_pattern


class PointwiseKernel(KernelDescription):
    KERNEL_FAMILY = "pointwise"

    def sancheck_lut_tensor(self, gpu, lut_tensor, fsels: "list[ArgumentSelection]"):
        # Only kernels that provide gen_autotune_configs may have entries in
        # tuning database
        return True

    # def get_missing_lut_entries(self, gpu, lut_tensor, fsels) -> list[dict]:
    #     SEQLEN_Q = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    #     SEQLEN_K = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    #     from copy import deepcopy
    #     import json
    #     import numpy as np
    #     base = {}
    #     def check_value(repr_name):
    #         for fsel in fsels:
    #             if fsel.repr_name == repr_name:
    #                 return fsel.argument_value
    #     base['causal'] = check_value('CAUSAL')
    #     base['d_head'] = check_value('BLOCK_DMODEL')
    #     base['dropout_p'] = 0.5 if check_value('ENABLE_DROPOUT') else 0.0
    #     def dtype():
    #         value = check_value('Q')
    #         if value.startswith('*fp16'):
    #             return 'float16'
    #         if value.startswith('*bf16'):
    #             return 'bfloat16'
    #         if value.startswith('*fp32'):
    #             return 'float32'
    #     base['dtype'] = dtype()
    #     base['bias_type'] = check_value('BIAS_TYPE')
    #     ret = []
    #     M_idxs, N_idxs = np.where(lut_tensor < 0)
    #     for M_id, N_id in zip(M_idxs, N_idxs):
    #         d = deepcopy(base)
    #         d['seqlen_q'] = SEQLEN_Q[M_id]
    #         d['seqlen_k'] = SEQLEN_K[N_id]
    #         ret.append(json.dumps(d))
    #     return ret
