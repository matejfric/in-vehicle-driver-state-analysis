#!/bin/bash

# Inspired by:
# - https://github.com/phohenecker/switch-cuda
# - https://notesbyair.github.io/blog/cs/2020-05-26-installing-multiple-versions-of-cuda-cudnn/

# Optionally, create a custom command in `~/.bashrc` by adding the following lines:
# alias switch-cuda='source /path/to/switch_cuda.sh' # Don't forget to change the path!
# Optionally, select a default cuda version:
# switch-cuda 12.5
# Finally, refresh ~/.bashrc
# $ source ~/.bashrc

# Ensure that the script has been sourced rather than just executed
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
    echo "Please use 'source' to execute switch-cuda.sh!"
    exit 1
fi

# Function to switch CUDA version
switch_cuda() {
    v=$1
    export PATH=/usr/local/cuda-$v/bin:$PATH
    export CUDADIR=/usr/local/cuda-$v
    export CUDA_HOME=/usr/local/cuda-$v
    export LD_LIBRARY_PATH=/usr/local/cuda-$v/lib64:$LD_LIBRARY_PATH
    nvcc --version
}

# Check if the user provided a version argument
if [ -z "$1" ]; then
    echo "Usage: source switch_cuda.sh <cuda_version>"
    echo "Available CUDA versions:"

    # List available CUDA versions
    for d in /usr/local/cuda-*/; do
        if [ -d "$d" ]; then
            echo "  $(basename $d | sed 's/cuda-//')"
        fi
    done

    return
fi

# Call the function with the provided version
switch_cuda $1

return
