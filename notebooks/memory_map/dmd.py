import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

from model.dmd import CATEGORIES, DRIVER_SESSION_MAPPING, ROOT
from model.memory_map import MemMapWriter
from model.prep_func_builder import PreprocessingFunctionBuilder


def _prepare_train(
    session_path: Path,
    resize: int,
    source: str,
    func: Callable[[Path, tuple[int, int]], np.ndarray],
    channels: int | None = None,
    overwrite: bool = False,
) -> None:
    sequencies: list[list[Path]] = [
        list((session_path / cat_dir).glob('*')) for cat_dir in CATEGORIES
    ]
    source_dir = 'rgb' if source == 'rgbd' else source
    all_dirs: list[Path] = [
        subdir / source_dir for sublist in sequencies for subdir in sublist
    ]
    all_dirs = [dir for dir in all_dirs if dir.is_dir()]
    src_extension: Literal['jpg', 'png'] = 'jpg' if source in ['rgb', 'rgbd'] else 'png'
    for dir in all_dirs:
        # Gather all image paths in the specified directory
        image_paths = sorted(dir.glob(f'*.{src_extension}'), key=lambda x: x.stem)
        if not image_paths:
            raise ValueError(f'No images found in {dir}.')

        output_file = dir.parent / 'memory_maps' / f'{source}_{resize}.dat'

        # Write the memory-mapped file
        mm_writer = MemMapWriter(
            image_paths,
            output_file,
            func=func,
            resize=(resize, resize),
            channels=channels,
        )
        mm_writer.write(overwrite=overwrite)
        print(mm_writer)


def _prepare_test(
    session_path: Path,
    resize: int,
    source: str,
    func: Callable[[Path, tuple[int, int]], np.ndarray],
    channels: int | None = None,
    overwrite: bool = False,
) -> None:
    src_extension: Literal['jpg', 'png'] = 'jpg' if source in ['rgb', 'rgbd'] else 'png'
    sequencies: list[list[Path]] = [
        list((session_path / cat_dir).glob('*')) for cat_dir in CATEGORIES
    ]
    source_dir = 'rgb' if source == 'rgbd' else source
    all_dirs: list[Path] = [
        subdir / source_dir for sublist in sequencies for subdir in sublist
    ]
    all_dirs = [dir for dir in all_dirs if dir.is_dir()]
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
        image_paths, output_file, func=func, resize=(resize, resize), channels=channels
    )
    mm_writer.write(overwrite=overwrite)
    print(mm_writer)


def main(args: argparse.Namespace) -> None:
    if args.driver:
        session_paths = [
            ROOT / session for session in DRIVER_SESSION_MAPPING[args.driver]
        ]
    elif args.session:
        session_paths = [ROOT / args.session]
    else:
        raise ValueError('Please specify either a driver or a session.')

    resize = args.resize
    source = args.type
    stage = args.stage
    overwrite = args.overwrite
    if source == 'rgb':
        channels = 3
    elif source == 'rgbd':
        channels = 4
        args.add_depth_channel = True
    else:
        channels = None

    prep_func_builder = PreprocessingFunctionBuilder().from_pillow().pad_square()
    if args.add_depth_channel:
        prep_func_builder = prep_func_builder.add_depth_channel()
    if args.mask:
        prep_func_builder = prep_func_builder.apply_mask()
    if args.multiply:
        prep_func_builder = prep_func_builder.multiply(args.multiply)

    func = prep_func_builder.build()

    def prep_train() -> None:
        for session_path in session_paths:
            _prepare_train(session_path, resize, source, func, channels, overwrite)

    def prep_test() -> None:
        for session_path in session_paths:
            _prepare_test(session_path, resize, source, func, channels, overwrite)

    if stage == 'train':
        prep_train()
    elif stage == 'test':
        prep_test()
    else:
        prep_train()
        prep_test()


if __name__ == '__main__':
    # Example usage:
    # 1. Activate the virtual environment
    # $ conda activate torch
    # 2. For binary masks:
    # $ python3 notebooks/memory_map/dmd.py --driver 1 --type masks --multiply 255 --resize 64
    # 3. For RGB images:
    # $ python3 notebooks/memory_map/dmd.py --driver 1 --type rgb --resize 64 --mask
    parser = argparse.ArgumentParser(
        description='Process images into a memory-mapped file.',
        usage='python3 run_memory_map_conversion.py --path <path> [--output <output>] [--resize <resize>] [--extension <extension>]',
    )
    parser.add_argument('--session', help='Name of the session.')
    parser.add_argument(
        '--type',
        choices=[
            'rgb',
            'rgbd',
            'depth',
            'source_depth',
            'video_depth_anything',
            'masks',
        ],
        required=True,
        help='Type of images to process.',
    )
    parser.add_argument('--output', help='Path to save the output memory-mapped file.')
    parser.add_argument(
        '--resize', type=int, default=256, help='Size to resize images.'
    )
    parser.add_argument(
        '--stage',
        choices=['train', 'test', 'all'],
        default='all',
        help='Stage of the dataset.',
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        help='Mask everything except the driver.',
    )
    parser.add_argument(
        '--add-depth-channel',
        action='store_true',
        help='Add depth channel.',
    )
    parser.add_argument(
        '--multiply',
        type=float,
        help='Multiply the image by a value.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the existing memory-mapped file.',
    )
    parser.add_argument(
        '--driver',
        choices=[1, 2, 3, 4, 5],
        type=int,
        help='Choose driver ID to extract all corresponding sessions.',
    )

    args = parser.parse_args()
    print(args)
    main(args)
