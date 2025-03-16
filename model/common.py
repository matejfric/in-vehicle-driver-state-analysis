import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

DRIVER_MAPPING: dict[str, int] = {
    '2021_08_31_geordi_enyaq': 1,
    '2021_11_18_jakubh_enyaq': 2,
    '2021_11_05_michal_enyaq': 4,
    '2021_09_06_poli_enyaq': 5,
    '2021_11_18_dans_enyaq': 6,
    '2024_07_02_radovan_enyaq': -1,  # Not in the original papers
}

type ModelStages = Literal['train', 'valid', 'test']


class BatchSizeDict(TypedDict):
    train: int
    valid: int
    test: int


def crop_driver_image(image: Image.Image, image_path: Path) -> Image.Image:
    """Crop from the left to create a square crop while maintaining the height."""
    if image_path.stem.startswith('2021_08_28'):
        # `2021_08_28_radovan_night_mask` different camera placement
        return image.crop((250, 0, 250 + image.size[1], image.size[1]))

    if image_path.stem.startswith('dmd_g'):
        # DMD dataset
        return pad_to_square(image, fill='black')

    return image.crop((0, 0, image.size[1], image.size[1]))


def crop_driver_image_contains(image: Image.Image, image_path: Path) -> Image.Image:
    """Crop from the left to create a square crop while maintaining the height."""
    if '2021_08_28' in str(image_path):
        # `2021_08_28_radovan_night_mask` different camera placement
        return image.crop((250, 0, 250 + image.size[1], image.size[1]))

    if 'dmd_g' in str(image_path):
        # DMD dataset
        return pad_to_square(image, fill='black')

    return image.crop((0, 0, image.size[1], image.size[1]))


def pad_to_square(image: Image.Image, fill: str = 'black') -> Image.Image:
    w, h = image.size
    if h == w:
        return image  # Already square

    padding = (w - h) // 2
    padded_image = ImageOps.expand(
        image,
        border=(0, padding, 0, padding),
        fill=fill,
    )
    return padded_image


def pad_to_square_cv(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image  # Already square

    padding = abs(w - h) // 2
    if h < w:
        padded_image = cv.copyMakeBorder(
            image, padding, padding, 0, 0, cv.BORDER_CONSTANT, value=0
        )
    else:
        padded_image = cv.copyMakeBorder(
            image, 0, 0, padding, padding, cv.BORDER_CONSTANT, value=0
        )

    return padded_image


def preprocess_515_cv(
    image: np.ndarray, opening_kernel_size: int | None = None
) -> np.ndarray:
    """Preprocess the image from the Intel RealSense L515 camera."""
    if opening_kernel_size:
        kernel = np.ones((opening_kernel_size, opening_kernel_size), np.float32)
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return cv.rotate(pad_to_square_cv(image), cv.ROTATE_180)


def preprocess_515(image: Image.Image) -> Image.Image:
    """Preprocess the image from the Intel RealSense L515 camera."""
    return pad_to_square(image, fill='black').rotate(180)


@dataclass
class Anomaly:
    start: int
    end: int
    labels: list[str]

    def middle(self) -> int:
        return (self.start + self.end) // 2


class Anomalies:
    def __init__(
        self, anomalies: list[Anomaly], video_length: int | None = None
    ) -> None:
        self.anomalies = anomalies
        self.video_length = video_length

    def __len__(self) -> int:
        return len(self.anomalies)

    def __getitem__(self, index: int) -> Anomaly:
        return self.anomalies[index]

    def __repr__(self) -> str:
        return f'Anomalies({self.anomalies})'

    def __iter__(self) -> Iterator[Anomaly]:
        return iter(self.anomalies)

    @staticmethod
    def from_json(
        path: str | Path | Sequence[str | Path],
    ) -> 'Anomalies':
        """Create an Anomalies instance from a JSON annotation file(s) in VCD format.

        Note
        ----
        https://vcd.vicomtech.org/
        """
        import json

        from model.dmd import DISTRACTIONS

        anomalies = []
        video_length = 0
        paths = [path] if isinstance(path, str | Path) else sorted(path)

        logger.info(f'Loading annotations from {len(paths)} files.')

        for path in paths:
            with open(path) as f:
                annotations = json.load(f)

            actions = annotations['openlabel']['actions']

            for value in actions.values():
                if (label := value['type']) in DISTRACTIONS:
                    for frame_interval in value['frame_intervals']:
                        anomalies.append(
                            Anomaly(
                                start=video_length + frame_interval['frame_start'],
                                end=video_length + frame_interval['frame_end'],
                                labels=[label],
                            )
                        )

            video_length += annotations['openlabel']['frame_intervals'][0]['frame_end']

        return Anomalies(anomalies=anomalies, video_length=video_length)

    @staticmethod
    def from_file(path: str | Path) -> 'Anomalies':
        """Read a text file and create an Anomalies instance."""
        with open(path) as file:
            data = file.readlines()

        # Parse lines into a list of Anomaly dictionaries
        parsed_data = []
        for line in data:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f'Invalid line: `{line}`. File: `{path}`')
            if '#' in parts[0]:
                continue
            start = int(parts[0])
            end = int(parts[1])
            labels = ' '.join(parts[2:]).split(',')
            labels = [label.strip() for label in labels]
            parsed_data.append(Anomaly(start=start, end=end, labels=labels))

        return Anomalies(parsed_data)

    def to_ground_truth(self, length: int = -1) -> list[int]:
        """Convert the anomalies to a ground truth list for binary classification.
        Negative samples are labeled as 0, positive samples are labeled as 1.
        """
        if self.video_length is not None:
            length = self.video_length
        if length == -1:
            length = max([anomaly.end for anomaly in self.anomalies])
        ground_truth = [0] * length
        for anomaly in self.anomalies:
            ground_truth[anomaly.start : anomaly.end] = [1] * (
                anomaly.end - anomaly.start
            )
        return ground_truth


def convert_depth_images_to_video(
    path: str | Path,
    output: str = 'video_depth.mp4',
    fps: int = 10,
    extension: str = 'png',
    limit: int | None = None,
) -> None:
    """Convert a sequence of depth images to a video."""
    from .depth_anything.utils.dc_utils import save_video
    from .dmd import _load_images_as_array

    path = Path(path)
    if not path.exists():
        raise ValueError(f'Directory not found: {path}')
    frames = _load_images_as_array(path, extension=extension, limit=limit)
    save_video(frames, path / output, fps=fps, is_depths=True)


def preprocess_depth_frame(
    frame: np.ndarray,
    depth_threshold: int = 2000,
) -> np.ndarray:
    """Preprocess depth frame."""
    # frame_clipped = np.clip(frame, 0, depth_threshold)
    frame_clipped = np.where(frame > depth_threshold, 0, frame)
    # Map the range [0, depth_threshold] to [0, 255]
    img8 = ((frame_clipped / depth_threshold) * 255).astype(np.uint8)
    return np.asarray(pad_to_square(Image.fromarray(img8)))
