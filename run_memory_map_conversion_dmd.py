import argparse
from pathlib import Path
from typing import Literal

from model.dmd import CATEGORIES, ROOT
from model.memory_map import MemMapWriter, resize_driver


def _prepare_train(session_path: Path, resize: int, source: str) -> None:
    sequencies: list[list[Path]] = [
        list((session_path / cat_dir).glob('*')) for cat_dir in CATEGORIES
    ]
    all_dirs: list[Path] = [
        subdir / source for sublist in sequencies for subdir in sublist
    ]

    for dir in all_dirs:
        # Gather all image paths in the specified directory
        image_paths = sorted(dir.glob(f'*.{args.extension}'), key=lambda x: x.stem)
        if not image_paths:
            raise ValueError(f'No images found in {dir}.')

        output_file = (
            args.output
            if args.output
            else dir.parent / 'memory_maps' / f'{source}_{resize}.dat'
        )

        # Write the memory-mapped file
        mm_writer = MemMapWriter(
            image_paths, output_file, func=resize_driver, resize=(resize, resize)
        )
        mm_writer.write()
        print(mm_writer)


def _prepare_test(session_path: Path, resize: int, source: str) -> None:
    session_path = session_path
    src_extension: Literal['jpg', 'png'] = 'jpg' if source == 'rgb' else 'png'
    sequencies: list[list[Path]] = [
        list((session_path / cat_dir).glob('*')) for cat_dir in CATEGORIES
    ]
    all_dirs: list[Path] = [
        subdir / source for sublist in sequencies for subdir in sublist
    ]
    print(f'Found {len(all_dirs)} directories.')

    image_paths = sorted(
        [img for dir in all_dirs for img in dir.glob(f'*.{src_extension}')],
        key=lambda x: x.stem,  # This is crucial for the correct order of the images in the memory-mapped file!
    )
    if not image_paths:
        raise ValueError(f'No images found in {session_path}.')

    output_file = session_path / 'memory_maps' / f'{source}_{resize}.dat'

    # Write the memory-mapped file
    mm_writer = MemMapWriter(
        image_paths, output_file, func=resize_driver, resize=(resize, resize)
    )
    mm_writer.write(overwrite=True)
    print(mm_writer)


def main(args: argparse.Namespace) -> None:
    session_path = ROOT / args.session
    resize = args.resize
    source = args.type
    stage = args.stage

    if stage == 'train':
        _prepare_train(session_path, resize, source)
    elif stage == 'test':
        _prepare_test(session_path, resize, source)
    else:
        _prepare_train(session_path, resize, source)
        _prepare_test(session_path, resize, source)


if __name__ == '__main__':
    # Example usage:
    # conda activate torch
    # $CONDA_PREFIX/bin/python3 run_memory_map_conversion_dmd.py --session "gA_1_s1_2019-03-08T09;31;15+01;00" --resize 128 --type depth --stage test
    # $CONDA_PREFIX/bin/python3 run_memory_map_conversion_dmd.py --session "gA_1_s2_2019-03-08T09;21;03+01;00" --resize 128 --type video_depth_anything
    parser = argparse.ArgumentParser(
        description='Process images into a memory-mapped file.',
        usage='python3 run_memory_map_conversion.py --path <path> [--output <output>] [--resize <resize>] [--extension <extension>]',
    )
    parser.add_argument('--session', required=True, help='Name of the session.')
    parser.add_argument(
        '--type',
        choices=['rgb', 'depth', 'video_depth_anything'],
        required=True,
        help='Type of images to process.',
    )
    parser.add_argument('--output', help='Path to save the output memory-mapped file.')
    parser.add_argument(
        '--resize', type=int, default=256, help='Size to resize images (default: 256).'
    )
    parser.add_argument(
        '--extension',
        type=str,
        choices=['png', 'jpg'],
        default='png',
        help='File extension to filter images (default: png).',
    )
    parser.add_argument(
        '--stage',
        choices=['train', 'test', 'all'],
        default='all',
        help='Stage of the dataset (default: all).',
    )
    # parser.add_argument(
    #     '--mask',
    #     action='store_true',
    #     help='Mask everything except the driver (default: False).',
    # )

    args = parser.parse_args()
    print(args)
    main(args)
