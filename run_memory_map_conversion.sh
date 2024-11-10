#!/bin/bash

# Define DRIVER_MAP as an associative array
declare -A DRIVER_MAP=(
    ["geordi"]="2021_08_31_geordi_enyaq"
    ["poli"]="2021_09_06_poli_enyaq"
    ["michal"]="2021_11_05_michal_enyaq"
    ["dans"]="2021_11_18_dans_enyaq"
    ["jakub"]="2021_11_18_jakubh_enyaq"
)

# Define the types to loop through
TYPES=("normal" "anomal")

# Set the resize parameter to the first argument,
# or default to 128 if not provided.
RESIZE=${1:-128}

# Loop through each driver and type
for DRIVER_DIR in "${!DRIVER_MAP[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        # Construct the path and run the command
        PATH_TO_CONVERT="$HOME/source/driver-dataset/2024-10-28-driver-all-frames/${DRIVER_MAP[$DRIVER_DIR]}/$TYPE/depth"
        echo "Running conversion for DRIVER=$DRIVER_DIR, TYPE=$TYPE"
        "$CONDA_PREFIX/bin/python3" run_memory_map_conversion.py --path "$PATH_TO_CONVERT" --resize "$RESIZE"
    done
done
