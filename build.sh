#!/bin/bash

script_name=$(basename "$0")

for item in * .[^.]* ..?*; do
    # exclude "." & ".."
    if [[ "$item" != "$script_name" && "$item" != "." && "$item" != ".." ]]; then
        rm -rf "$item"
    fi
done

cmake .. -DCMAKE_INSTALL_PREFIX=installed_dir/aotriton -G Ninja &&

ninja install -j128 &&

cp /work/aotriton/build/installed_dir/aotriton/lib/* /usr/local/lib/python3.11/dist-packages -r &&

rm venv/ -rf &&

python3.11 /work/aotriton/test/test_add_kernel.py