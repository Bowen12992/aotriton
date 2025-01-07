#!/bin/bash

# script_name=$(basename "$0")

# for item in * .[^.]* ..?*; do
#     # exclude "." & ".."
#     if [[ "$item" != "$script_name" && "$item" != "." && "$item" != ".." ]]; then
#         rm -rf "$item"
#     fi
# done

cmake .. -DCMAKE_INSTALL_PREFIX=installed_dir/aotriton -DAOTRITON_BUILD_FOR_TUNING=0 -G Ninja &&

ninja install -j128 &&

cp /work/aotriton/build/installed_dir/aotriton/lib/* /usr/local/lib/python3.11/dist-packages -r &&

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 &&
export LD_LIBRARY_PATH=/work/aotriton/build/installed_dir/aotriton/lib:$LD_LIBRARY_PATH &&

python3.11 /work/aotriton/test/test_add_kernel.py

# clang-format
# pip install clang-format==13.0.0
# find . -path "./third_party" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | xargs clang-format -i
# pre-commit run --all-files
