from pathlib import Path
from sys import argv

from model.memory_map import MemMapWriter, crop_mask_resize_driver


if __name__ == '__main__':
    # /home/lanter/source/driver-dataset/2024-10-28-driver-all-frames/2021_08_31_geordi_enyaq/anomal/depth
    if len(argv) < 2:
        raise ValueError(
            'Please provide the image directory as an argument. Example: `data/`.'
        )

    image_dir = Path(argv[1])
    output_file = argv[2] if len(argv) == 3 else image_dir / 'mem_map.dat'

    image_paths = list(Path(image_dir).glob('*.png'))
    if not image_paths:
        raise ValueError(f'No images found in {image_dir}.')

    # Write the memory-mapped file
    mm_writer = MemMapWriter(image_paths, output_file, crop_mask_resize_driver)
    mm_writer.write()
    print(mm_writer)
