from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from PIL import Image

ModelStages: TypeAlias = Literal['train', 'valid', 'test']


class BatchSizeDict(TypedDict):
    train: int
    valid: int
    test: int


def crop_driver_image(image: Image.Image, image_path: Path) -> Image.Image:
    """Crop from the left to create a square crop while maintaining the height."""
    if image_path.stem.startswith('2021_08_28'):
        # `2021_08_28_radovan_night_mask` different camera placement
        return image.crop((250, 0, 250 + image.size[1], image.size[1]))
    return image.crop((0, 0, image.size[1], image.size[1]))


def crop_driver_image_contains(image: Image.Image, image_path: Path) -> Image.Image:
    """Crop from the left to create a square crop while maintaining the height."""
    if '2021_08_28' in str(image_path):
        # `2021_08_28_radovan_night_mask` different camera placement
        return image.crop((250, 0, 250 + image.size[1], image.size[1]))
    return image.crop((0, 0, image.size[1], image.size[1]))


@dataclass
class Anomaly:
    start: int
    end: int
    labels: list[str]

    def middle(self) -> int:
        return (self.start + self.end) // 2


class Anomalies:
    def __init__(self, anomalies: list[Anomaly]) -> None:
        self.anomalies = anomalies

    def __len__(self) -> int:
        return len(self.anomalies)

    def __getitem__(self, index: int) -> Anomaly:
        return self.anomalies[index]

    def __repr__(self) -> str:
        return f'Anomalies({self.anomalies})'

    def __iter__(self) -> Iterator[Anomaly]:
        return iter(self.anomalies)

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
        if length == -1:
            length = max([anomaly.end for anomaly in self.anomalies])
        ground_truth = [0] * length
        for anomaly in self.anomalies:
            ground_truth[anomaly.start : anomaly.end] = [1] * (
                anomaly.end - anomaly.start
            )
        return ground_truth
