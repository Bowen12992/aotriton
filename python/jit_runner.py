"""
This is a test file for JIT run a function:
    - move what cmake has done in this file
"""

import argparse
from pathlib import Path

from .jit_aks2 import main_fn as aks2
from .jit_compile import main_fn as compile
from .jit_generate_compile import ClusterRegistry
from .jit_generate_compile import main_fn as generate_compile


def main():
    args = argparse.Namespace(
        target_gpus=["RTX3090"],
        build_dir="/work/aotriton/build/src",
        python=None,
        bare_mode=True,
        test_clustering=False,
        generate_cluster_info=True,
        build_for_tuning=False,
        _cluster_registry=ClusterRegistry(),
    )
    generate_compile(args)

    args = argparse.Namespace(
        path=Path("/work/aotriton/tritonsrc/add_kernel.py"),
        kernel_name="add_kernel",
        out_path=Path(
            "/work/aotriton/build/src/pointwise/gpu_kernel_image.add_kernel/add_kernel-Sig-F__^fp32@16__P__32__CO__-Gpu-RTX3090.cubin"
        ),
        grid="1,1,1",
        num_warps=4,
        num_stages=4,
        waves_per_eu=0,
        target="RTX3090",
        signature="*fp32:16, *fp32:16, *fp32:16, i32, 32",
        verbose=False,
        nostrip=False,
        timeout=0.0,
    )
    compile(args)

    args = argparse.Namespace(
        o="/work/aotriton/build/src/aotriton.images/nv-rtx3090/pointwise/add_kernel/FONLY__^bf16@16___RTX3090.aks2",
        hsaco_files=[
            "/work/aotriton/build/src/pointwise/gpu_kernel_image.add_kernel/add_kernel-Sig-F__^fp32@16__P__32__CO__-Gpu-RTX3090.cubin"
        ],
    )
    aks2(args)


if __name__ == "__main__":
    main()
