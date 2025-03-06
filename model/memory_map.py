from collections.abc import Callable, Generator
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import crop_driver_image_contains, preprocess_515, preprocess_515_cv


def resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    image_pil = Image.open(image_path)
    image_pil = image_pil.resize(resize)
    return np.array(image_pil)


def intel_515_cv(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    img = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    img = np.array(img, dtype=np.uint16)
    img = (img / np.iinfo(np.uint16).max).astype(np.float32)
    img = preprocess_515_cv(img, opening_kernel_size=8)
    img = cv.resize(img, resize, interpolation=cv.INTER_AREA)
    return img


def intel_515(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    image_pil = Image.open(image_path)
    image_pil = preprocess_515(image_pil)
    image_pil = image_pil.resize(resize)
    image_np = np.array(image_pil, dtype=np.float32)
    image_np = image_np / image_np.max()
    return image_np


def crop_resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    image_pil = Image.open(image_path)
    image_pil = crop_driver_image_contains(image_pil, image_path)
    image_pil = image_pil.resize(resize)
    return np.array(image_pil)


def mask_resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    """Assumes `masks` directory is in the same directory as the images."""
    image_pil = Image.open(image_path)
    image_pil = image_pil.resize(resize)

    mask_path = image_path.parent.parent / 'masks' / image_path.with_suffix('.png').name
    mask_pil = Image.open(mask_path).convert('L').resize(resize)

    image = np.array(image_pil).astype(np.float32)
    mask = (np.array(mask_pil) > 0).astype(np.float32)

    return (image * mask).astype(np.uint8)


def crop_mask_resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    """Assumes `masks` directory is in the same directory as the images."""
    image_pil = Image.open(image_path)
    image_pil = crop_driver_image_contains(image_pil, image_path)
    image_pil = image_pil.resize(resize)

    mask_path = image_path.parent.parent / 'masks' / image_path.with_suffix('.png').name
    mask_pil = Image.open(mask_path).convert('L').resize(resize)

    image = np.array(image_pil).astype(np.float32)
    mask = (np.array(mask_pil) > 0).astype(np.float32)

    if len(image.shape) == 3:
        mask = mask[..., np.newaxis]

    return (image * mask).astype(np.uint8)


def binary_mask_resize(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    mask_pil = Image.open(image_path).convert('L')
    mask_pil = crop_driver_image_contains(mask_pil, image_path).resize(resize)
    return (np.array(mask_pil) > 0).astype(np.uint8) * 255


class MemMapWriter:
    def __init__(
        self,
        image_paths: list[Path],
        output_file: Path | str = 'mem_map.dat',
        func: Callable = crop_resize_driver,
        resize: tuple[int, int] = (256, 256),
        dtype: type = np.uint8,
        channels: int | None = None,
    ) -> None:
        """Initialize MemMapWriter with parameters to process and save images to a memory-mapped file.

        WARNING: The images should be sorted, if the order is important!

        Example
        -------
        >>> from pathlib import Path
        >>> from model.memory_map import MemMapWriter
        >>>
        >>> image_paths = sorted(Path('data').glob('*.png'), key=lambda x: x.stem)
        >>> with MemMapWriter(image_paths, 'mem_map.dat') as f:
        >>>     f.write()
        """
        self.resize = resize
        self.func = func
        self.image_paths = image_paths
        self.output_file = (
            Path(output_file) if isinstance(output_file, str) else output_file
        )
        self.dtype = dtype
        self.channels = channels
        self.memmap_array = ...

    def __call__(self) -> Generator[np.ndarray, None, None]:
        for image_path in self.image_paths:
            image = self.func(image_path, self.resize)
            yield image

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns the processed image at the specified index."""
        return self.func(self.image_paths[index], self.resize)

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.image_paths)

    def __repr__(self) -> str:
        return f'MemMap with {len(self):,} images:\n- Output file: {self.output_file}\n- Resize: {self.resize}\n- dtype: {self.dtype.__name__}\n- func: {self.func.__name__}'

    def write(self, overwrite: bool = False, num_workers: int | None = None) -> None:
        """Process images with `func` and write them to the memory-mapped file using multithreading.

        Parameters
        ----------
        overwrite: bool, default=False
            Whether to overwrite the output file if it exists.
        num_workers: int, default=None
            Number of worker threads to use. If None, uses the number of CPU cores.
        """
        import concurrent.futures
        import os

        # Default to number of CPU cores if num_workers not specified
        if num_workers is None:
            num_workers = os.cpu_count() or 4

        if self.output_file.exists():
            if overwrite:
                self.output_file.unlink()
            else:
                raise FileExistsError(f'{self.output_file} already exists.')
        if self.output_file.suffix != '.dat':
            raise ValueError('`output_file` must have a `.dat` extension.')
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.memmap_array = np.memmap(
            self.output_file,
            dtype=self.dtype,
            mode='w+',
            shape=(len(self), *self.resize, self.channels)
            if self.channels
            else (len(self), *self.resize),
        )

        # Process images in chunks for better performance
        def process_chunk(chunk: list[int]) -> list[tuple[int, np.ndarray]]:
            results = []
            for idx in chunk:
                if idx < len(self):
                    try:
                        image = self[idx]  # Get the processed image
                        results.append((idx, image))
                    except Exception as e:
                        print(f'Error processing image at index {idx}: {e}')
            return results

        # Create chunks of indices to process
        total_images = len(self)
        indices = list(range(total_images))
        # Create more chunks than workers for better load balancing
        chunk_size = max(1, total_images // (num_workers * 4))
        chunks = [
            indices[i : i + chunk_size] for i in range(0, total_images, chunk_size)
        ]

        processed_count = 0

        # Use ThreadPoolExecutor for I/O bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {
                executor.submit(process_chunk, chunk): chunk for chunk in chunks
            }

            # Use tqdm for progress tracking
            with tqdm(total=total_images, desc='Saving memmap') as pbar:
                for future in concurrent.futures.as_completed(future_to_chunk):
                    results = future.result()
                    for idx, image in results:
                        self.memmap_array[idx] = image  # Write to memmap
                        processed_count += 1
                        pbar.update(1)

        # Ensure all data is written to disk
        self.memmap_array.flush()


class MemMapReader:
    def __init__(
        self,
        memmap_file: Path | str,
        shape: tuple[int, int] | tuple[int, int, int],
        dtype: type = np.uint8,
    ) -> None:
        """Initialize MemMapReader with parameters to read images from a memory-mapped file.

        Example
        -------
        >>> from model.memory_map import MemMapReader
        >>>
        >>> memory_map = MemMapReader('mem_map.dat', (256, 256))
        >>> image = memory_map[0]
        """
        if isinstance(memmap_file, str):
            memmap_file = Path(memmap_file)
        self.memmap_file = memmap_file
        self.shape = shape

        memmap_bytes = len(np.memmap(memmap_file, mode='r', dtype=dtype))
        self.n_images = int(memmap_bytes // np.prod(shape))

        if not self.n_images * np.prod(shape) == memmap_bytes:
            raise ValueError(f'{__class__.__name__}: Shape `{shape}` is invalid.')

        self.memmap = np.memmap(
            memmap_file, mode='r', shape=(self.n_images, *shape), dtype=dtype
        )

    def __len__(self) -> int:
        """Returns the number of images in the memory-mapped file."""
        return self.n_images

    def __getitem__(self, index: int) -> np.ndarray:
        return self.memmap[index]

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate over the memory-mapped images."""
        for i in range(self.n_images):
            yield self.memmap[i]

    def __repr__(self) -> str:
        return f'MemMap with {len(self):,} images:\n- File: {self.memmap_file}\n- Shape: {self.shape}'

    def window(
        self, start: int, window_size: int, time_step: int = 1
    ) -> list[np.ndarray]:
        """Return a list of `window_size` read-only images taking every `time_step`-th image. For `time_step=1` this returns consecutive images."""
        end = min(start + window_size * time_step, self.n_images)
        return [self[i] for i in range(start, end, time_step)]

    def window_mut(
        self, start: int, window_size: int, time_step: int = 1
    ) -> list[np.ndarray]:
        """Return a list of `window_size` mutable images taking every `time_step`-th image. For `time_step=1` this returns consecutive images."""
        end = min(start + window_size * time_step, self.n_images)
        return [np.copy(self[i]) for i in range(start, end, time_step)]

    def iter_windows(
        self, window_size: int, time_step: int = 1
    ) -> Generator[list[np.ndarray], None, None]:
        """Iterate over the memory-mapped images in windows of a given size. Yields a list of `window_size` read-only images."""
        for start in range(0, self.n_images):
            yield self.window(start, window_size, time_step)

    def iter_windows_mut(
        self, window_size: int, time_step: int = 1
    ) -> Generator[list[np.ndarray], None, None]:
        """Iterate over the memory-mapped images in windows of a given size. Yields a list of `window_size` read-only images."""
        for start in range(0, self.n_images):
            yield self.window_mut(start, window_size, time_step)
