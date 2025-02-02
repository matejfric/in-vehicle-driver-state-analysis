#!/bin/bash

# This script runs the memory map conversion for all drivers defined in `DRIVER_MAP`
# and types (directories) defined in `TYPES`.
# If the memory file already exists, it will be skipped.

# Example:
# $ bash run_memory_map_conversion.sh 128 png "" "video_depth_anything"

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
EXTENSION=${2:-"png"}
MASK=""
if [[ $3 == "--mask" ]]; then
    MASK="--mask"
fi
SOURCE_TYPE=${4:-"depth"}  # video_depth_anything
CROP=""
if [[ $5 == "--crop" ]]; then
    CROP="--crop"
fi

# Print the settings
echo "Settings:"
echo "Resize: $RESIZE"
echo "Extension: $EXTENSION"
echo "Mask option: $MASK"
echo "Driver Map: ${DRIVER_MAP[@]}"
echo "Types: ${TYPES[@]}"
echo "Source type: $SOURCE_TYPE"
echo

# Prompt the user to continue
read -p "Press Enter to continue..."

# Loop through each driver and type
for DRIVER_DIR in "${!DRIVER_MAP[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        # Construct the path and run the command
        PATH_TO_CONVERT="$HOME/source/driver-dataset/2024-10-28-driver-all-frames/${DRIVER_MAP[$DRIVER_DIR]}/$TYPE/$SOURCE_TYPE"
        echo "Running conversion for DRIVER=$DRIVER_DIR, TYPE=$TYPE"
        "$CONDA_PREFIX/bin/python3" run_memory_map_conversion.py --path "$PATH_TO_CONVERT" --resize "$RESIZE" --extension "$EXTENSION" $MASK $CROP
    done
done
