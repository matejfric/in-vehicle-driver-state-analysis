import argparse
from pathlib import Path

import numpy as np

from model.memory_map import (
    MemMapWriter,
    crop_mask_resize_driver,
    crop_resize_driver,
    intel_515,
    mask_resize_driver,
    resize_driver,
)


def main(args: argparse.Namespace) -> None:
    image_dir = Path(args.path)
    resize = args.resize
    output_file = (
        args.output
        if args.output
        else image_dir.parent / 'memory_maps' / f'{image_dir.name}_{resize}.dat'
    )

    # Gather all image paths in the specified directory
    image_paths = sorted(image_dir.glob(f'*.{args.extension}'), key=lambda x: x.stem)
    if not image_paths:
        raise ValueError(f'No images found in {image_dir}.')

    data_type = np.uint8

    # Write the memory-mapped file
    if args.mask and args.crop:
        func = crop_mask_resize_driver
    elif args.crop:
        func = crop_resize_driver
        output_file = str(output_file).replace('.dat', '_no_mask.dat')
    elif args.mask:
        func = mask_resize_driver
    elif args.L515:
        func = intel_515
        output_file = str(output_file).replace('.dat', '_no_mask.dat')
        data_type = np.float32
    else:
        func = resize_driver
        output_file = str(output_file).replace('.dat', '_no_mask.dat')

    mm_writer = MemMapWriter(
        image_paths, output_file, func, resize=(resize, resize), dtype=data_type
    )
    mm_writer.write()
    print(mm_writer)


if __name__ == '__main__':
    # Example usage:
    # conda activate torch
    # $CONDA_PREFIX/bin/python3 run_memory_map_conversion.py --path "/home/lanter/source/driver-dataset/2024-10-28-driver-all-frames/2021_08_31_geordi_enyaq/normal/depth" --resize 224
    parser = argparse.ArgumentParser(
        description='Process images into a memory-mapped file.',
        usage='python3 run_memory_map_conversion.py --path <path> [--output <output>] [--resize <resize>] [--extension <extension>]',
    )
    parser.add_argument(
        '--path', required=True, help='Path to the directory containing images.'
    )
    parser.add_argument('--output', help='Path to save the output memory-mapped file.')
    parser.add_argument(
        '--resize', type=int, default=256, help='Size to resize images (default: 256).'
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='png',
        help='File extension to filter images (default: png).',
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        help='Mask everything except the driver (default: False).',
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        help='Crop the image to the driver area (default: False).',
    )
    parser.add_argument(
        '--L515',
        action='store_true',
        help='Use preprocessing for Intel RealSense L515 camera (default: False).',
    )

    args = parser.parse_args()
    main(args)
