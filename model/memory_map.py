from collections.abc import Callable, Generator, Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import crop_driver_image_contains


def crop_resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    image_pil = Image.open(image_path)
    image_pil = crop_driver_image_contains(image_pil, image_path)
    image_pil = image_pil.resize(resize)
    return np.array(image_pil)


def crop_mask_resize_driver(image_path: Path, resize: tuple[int, int]) -> np.ndarray:
    """Assumes `masks` directory is in the same directory as the images."""
    image_pil = Image.open(image_path)
    image_pil = crop_driver_image_contains(image_pil, image_path)
    image_pil = image_pil.resize(resize)

    mask_path = image_path.parent.parent / 'masks' / image_path.with_suffix('.png').name
    mask_pil = Image.open(mask_path).convert('L').resize(resize)

    image = np.array(image_pil).astype(np.float32)
    mask = (np.array(mask_pil) > 0).astype(np.float32)

    return (image * mask).astype(np.uint8)


class MemMapWriter:
    def __init__(
        self,
        image_paths: Iterable[Path],
        output_file: Path | str = 'mem_map.dat',
        func: Callable = crop_resize_driver,
        resize: tuple[int, int] = (256, 256),
        dtype: type = np.uint8,
    ) -> None:
        """Initialize MemMapWriter with parameters to process and save images to a memory-mapped file.

        Example
        -------
        >>> from pathlib import Path
        >>> from model.memory_map import MemMapWriter
        >>>
        >>> image_paths = Path('data').glob('*.png')
        >>> with MemMapWriter(image_paths, 'mem_map.dat') as f:
        >>>     f.write()
        """
        self.resize = resize
        self.func = func
        self.image_paths = sorted(image_paths)
        self.output_file = (
            Path(output_file) if isinstance(output_file, str) else output_file
        )
        self.dtype = dtype
        self.memmap_array = ...

    def __call__(self) -> Generator[np.ndarray, None, None]:
        for image_path in self.image_paths:
            image = self.func(image_path, self.resize)
            yield image

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.image_paths)

    def __repr__(self) -> str:
        return f'MemMap with {len(self):,} images:\n- Output file: {self.output_file}\n- Resize: {self.resize}\n- dtype: {self.dtype.__name__}\n- func: {self.func.__name__}'

    def write(self) -> None:
        """Process images with `func` and write them to the memory-mapped file."""

        if self.output_file.exists():
            raise FileExistsError(f'{self.output_file} already exists.')
        if self.output_file.suffix != '.dat':
            raise ValueError('`output_file` must have a `.dat` extension.')
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.memmap_array = np.memmap(
            self.output_file,
            dtype=self.dtype,
            mode='w+',
            shape=(len(self), *self.resize),
        )

        for i, image in tqdm(enumerate(self()), total=len(self), desc='Saving memmap'):
            self.memmap_array[i] = image  # type: ignore

        self.memmap_array.flush()  # type: ignore


class MemMapReader:
    def __init__(self, memmap_file: Path | str, shape: tuple[int, int]) -> None:
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

        memmap_bytes = len(np.memmap(memmap_file, mode='r'))
        self.n_images = int(memmap_bytes // np.prod(shape))

        if not self.n_images * np.prod(shape) == memmap_bytes:
            raise ValueError(f'{__class__.__name__}: Shape `{shape}` is invalid.')

        self.memmap = np.memmap(memmap_file, mode='r', shape=(self.n_images, *shape))

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, index: int) -> np.ndarray:
        return self.memmap[index]

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        for i in range(self.n_images):
            yield self.memmap[i]

    def __repr__(self) -> str:
        return f'MemMap with {len(self):,} images:\n- File: {self.memmap_file}\n- Shape: {self.shape}'

    def window(self, start: int, window_size: int) -> list[np.ndarray]:
        return [self[i] for i in range(start, min(start + window_size, self.n_images))]

    def iter_windows(self, window_size: int) -> Generator[list[np.ndarray], None, None]:
        """Iterate over the memory-mapped images in windows of a given size."""
        for start in range(0, self.n_images):
            yield self.window(start, window_size)
