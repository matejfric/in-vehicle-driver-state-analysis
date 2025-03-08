import argparse
from pathlib import Path

import numpy as np

from model.memory_map import (
    MemMapWriter,
    binary_mask_resize,
    crop_mask_resize_driver,
    crop_resize_driver,
    intel_515,  # noqa F401
    intel_515_cv,
    mask_resize_driver,
    resize_driver,
    rgbd_resize_crop_driver,
    rgbdm_resize_crop_driver,
)


def main(args: argparse.Namespace) -> None:
    image_dir = Path(args.path)
    resize = args.resize
    channels = args.channels
    mem_map_name = (
        f'{args.output_name}_{resize}.dat'
        if args.output_name
        else f'{image_dir.name}_{resize}.dat'
    )
    output_file = image_dir.parent / 'memory_maps' / mem_map_name

    # Gather all image paths in the specified directory
    image_paths = sorted(image_dir.glob(f'*.{args.extension}'), key=lambda x: x.stem)
    if not image_paths:
        raise ValueError(f'No images found in {image_dir}.')

    data_type = np.uint8

    # Write the memory-mapped file
    if args.binary_mask:
        func = binary_mask_resize
        channels = 1
    elif args.rgbd:
        func = rgbd_resize_crop_driver
        channels = 4
    elif args.rgbdm:
        func = rgbdm_resize_crop_driver
        channels = 5
    elif args.mask and args.crop:
        func = crop_mask_resize_driver
    elif args.crop:
        func = crop_resize_driver
        output_file = str(output_file).replace('.dat', '_no_mask.dat')
    elif args.mask:
        func = mask_resize_driver
    elif args.L515:
        func = intel_515_cv
        output_file = str(output_file).replace('.dat', '_no_mask_515.dat')
        data_type = np.float32
        channels = 1
    else:
        func = resize_driver
        output_file = str(output_file).replace('.dat', '_no_mask.dat')

    mm_writer = MemMapWriter(
        image_paths,
        output_file,
        func,
        resize=(resize, resize),
        dtype=data_type,
        channels=channels,
    )
    mm_writer.write(overwrite=args.overwrite)
    print(mm_writer)


DRIVER_MAP = {
    'geordi': '2021_08_31_geordi_enyaq',
    'poli': '2021_09_06_poli_enyaq',
    'michal': '2021_11_05_michal_enyaq',
    'dans': '2021_11_18_dans_enyaq',
    'jakub': '2021_11_18_jakubh_enyaq',
    # 'radovan': '2024_07_02_radovan_enyaq',
}
TYPES = ['normal', 'anomal', 'anomal_182201', 'anomal_181149']
BASE_DIR = Path.home() / 'source/driver-dataset/2024-10-28-driver-all-frames'

if __name__ == '__main__':
    # Example usage:
    # conda activate torch
    # $CONDA_PREFIX/bin/python3 notebooks/memory_map/mrl.py --help
    # $CONDA_PREFIX/bin/python3 notebooks/memory_map/mrl.py --binary-mask --source-type masks --resize 64
    # $CONDA_PREFIX/bin/python3 notebooks/memory_map/mrl.py --source-type images --resize 64 --channels 3 --mask --crop --extension jpg
    parser = argparse.ArgumentParser(
        description='Process images into a memory-mapped file.',
        usage='python3 run_memory_map_conversion.py',
    )
    parser.add_argument(
        '--path',
        help='Path to the directory containing images (single mode).',
    )
    parser.add_argument('--output-name', help='Prefix of the memory-mapped file.')
    parser.add_argument(
        '--resize', type=int, default=128, help='Size to resize images (default: 128).'
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
        help='Crop images to driver area (default: False).',
    )
    parser.add_argument(
        '--L515',
        action='store_true',
        help='Use preprocessing for Intel RealSense L515 camera (default: False).',
    )
    parser.add_argument(
        '--source-type', default='depth', help='Source data type directory.'
    )
    parser.add_argument(
        '--driver', choices=list(DRIVER_MAP.keys()), help='Driver name.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing memory-mapped file.',
    )
    parser.add_argument(
        '--binary-mask',
        action='store_true',
        help='Use binary mask (default: False).',
    )
    parser.add_argument(
        '--channels',
        type=int,
        help='Number of channels in the image (default: None).',
    )
    parser.add_argument(
        '--rgbd',
        action='store_true',
        help='Use RGBD images (default: False).',
    )
    parser.add_argument(
        '--rgbdm',
        action='store_true',
        help='Use RGBDM images (default: False).',
    )

    args = parser.parse_args()

    def _print_settings(single_mode: bool = False) -> None:
        print('Memory map conversion settings:')
        print(f'Resize: {args.resize}')
        print(f'Extension: {args.extension}')
        print(f'Mask: {args.mask}')
        print(f'Crop: {args.crop}')
        print(f'Source Type: {args.source_type}')
        print(f'Overwrite: {args.overwrite}')
        print(f'Binary Mask: {args.binary_mask}')
        print(f'Channels: {args.channels}')
        print(f'RGBD: {args.rgbd}')
        print(f'RGBDM: {args.rgbdm}')
        print(f'Output Name: {args.output_name}')
        if not single_mode:
            if args.driver:
                print(f'Driver: {args.driver}')
            else:
                print(f'Drivers: {list(DRIVER_MAP.keys())}')
            print(f'Types: {TYPES}')
        input('\nPress Enter to start the processing...')

    # Handle single mode
    if args.path:
        _print_settings(single_mode=True)
        main(args)
    else:
        _print_settings()
        driver_map = (
            {args.driver: DRIVER_MAP[args.driver]} if args.driver else DRIVER_MAP
        )
        for driver_key, driver_dir in driver_map.items():
            for normal_or_anomal in TYPES:
                input_path = BASE_DIR / driver_dir / normal_or_anomal / args.source_type
                if not input_path.exists():
                    if (
                        '_182201' not in input_path.parent.name
                        and '_181149' not in input_path.parent.name
                    ):
                        print(f'Path does not exist: {input_path}')
                    continue

                print(f'Processing: {driver_key} - {normal_or_anomal}')
                process_args = argparse.Namespace(
                    path=input_path,
                    output_name=args.output_name,
                    resize=args.resize,
                    extension=args.extension,
                    mask=args.mask,
                    crop=args.crop,
                    L515=args.L515,
                    source_type=args.source_type,
                    driver=driver_key,
                    overwrite=args.overwrite,
                    binary_mask=args.binary_mask,
                    channels=args.channels,
                    rgbd=args.rgbd,
                    rgbdm=args.rgbdm,
                )
                main(process_args)
