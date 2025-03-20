import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

logger = logging.getLogger(__name__)

DRIVER_MAPPING: dict[str, int] = {
    '2021_08_31_geordi_enyaq': 1,
    '2021_11_18_jakubh_enyaq': 2,
    '2021_11_05_michal_enyaq': 4,
    '2021_09_06_poli_enyaq': 5,
    '2021_11_18_dans_enyaq': 6,
    '2024_07_02_radovan_enyaq': -1,  # Not in the original papers
}
DRIVER_NAMES_MAPPING: dict[str, int] = {
    'geordi': 1,
    'jakubh': 2,
    'michal': 4,
    'poli': 5,
    'dans': 6,
    'radovan': -1,  # Not in the original papers
}


type ModelStages = Literal['train', 'valid', 'test']


class BatchSizeDict(TypedDict):
    train: int
    valid: int
    test: int


def get_video_frame_count(video_path: str | Path) -> int:
    """Get the frame count of a video using OpenCV."""
    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        logger.error(f'Error opening video: {video_path}')
        return -1

    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def create_video_from_images(
    frame_paths: list[Path],
    output_path: str | Path = 'output.mp4',
    fps: int = 30,
    size: None | tuple[int, int] = None,
) -> bool:
    """
    Create an MP4 video from a list of image paths.

    Parameters
    ----------
    frame_paths: list[Path]
        List of paths to image files.
    output_path: str | Path
        Path where the output video will be saved.
    fps: int
        Frames per second for the output video.
    size: None | tuple[int, int]
        Size (width, height) for the output video. If None, size of first image will be used.

    Returns
    -------
    bool
        True if video was created successfully, False otherwise.
    """
    if not frame_paths:
        print('Error: No image paths provided')
        return False

    # Read the first image to get dimensions if size is not provided
    first_image = cv.imread(frame_paths[0])
    if first_image is None:
        print(f'Error: Could not read image at {frame_paths[0]}')
        return False

    if size is None:
        height, width = first_image.shape[:2]
        size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv.VideoWriter(output_path, fourcc, fps, size)

    # Process each image and write to video
    for i, img_path in tqdm(enumerate(frame_paths), desc='Creating video'):
        img = cv.imread(img_path)
        if img is None:
            print(f'Warning: Could not read image at {img_path}, skipping')
            continue

        # Resize image if necessary
        if img.shape[:2] != (size[1], size[0]):
            img = cv.resize(img, size)

        # Write the frame to video
        out.write(img)

    # Release the VideoWriter
    out.release()
    print(f'Video created successfully at {output_path}')
    return True


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
        video_lengths: Sequence[int],
    ) -> 'Anomalies':
        """Create an Anomalies instance from a JSON annotation file(s) in VCD format.

        Note
        ----
        https://vcd.vicomtech.org/
        """
        import json

        from model.dmd import DISTRACTIONS

        anomalies = []
        video_length_total = 0
        paths = [path] if isinstance(path, str | Path) else sorted(path)

        logger.info(f'Loading annotations from {len(paths)} files.')

        for path, video_length in zip(paths, video_lengths, strict=True):
            with open(path) as f:
                annotations = json.load(f)

            actions = annotations['openlabel']['actions']

            for value in actions.values():
                if (label := value['type']) in DISTRACTIONS:
                    for frame_interval in value['frame_intervals']:
                        anomalies.append(
                            Anomaly(
                                start=video_length_total
                                + frame_interval['frame_start'],
                                end=video_length_total + frame_interval['frame_end'],
                                labels=[label],
                            )
                        )
            # We do not use the video length from the JSON file, because it is often incorrect
            # and the actual video has less frames than what is stated in the JSON file.
            video_length_total += video_length

        return Anomalies(anomalies=anomalies, video_length=video_length_total)

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
    frame_clipped = np.where(frame > depth_threshold, 0, frame)
    # Map the range [0, depth_threshold] to [0, 255]
    img8 = ((frame_clipped / depth_threshold) * 255).astype(np.uint8)
    return np.asarray(pad_to_square(Image.fromarray(img8)))
