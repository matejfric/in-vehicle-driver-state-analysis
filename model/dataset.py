import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import albumentations as albu
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import BatchSizeDict, crop_driver_image
from .memory_map import MemMapReader

logger = logging.getLogger(__name__)


class DatasetItem(TypedDict):
    image: np.ndarray | torch.Tensor
    mask: np.ndarray | torch.Tensor  # copy of `image` for AE or future frames for STAE
    filename: str | int  # first frame index or filename


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images: list[Path],
        masks: list[Path] | None = None,
        transforms: albu.Compose | None = None,
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> DatasetItem:
        """Image will be scaled in the SegmentationModel."""
        image_path = self.images[idx]
        image_pil = Image.open(image_path).convert('RGB')
        # Crop from the left to create a square crop while maintaining the height
        image_pil = crop_driver_image(image_pil, image_path)
        image = np.array(image_pil)

        result = {'image': image}

        if self.masks is not None:
            mask_pil = Image.open(self.masks[idx]).convert('L')

            # Crop from the left to create a square crop while maintaining the height
            mask_pil = crop_driver_image(mask_pil, image_path)

            mask = (np.array(mask_pil) > 0).astype(np.float32)
            result['mask'] = mask

        if self.transforms is not None:
            result = self.transforms(**result)  # type: ignore
            result['mask'] = np.expand_dims(
                result['mask'], 0
            )  # [batch_size, num_classes, height, width]
        else:
            # This is done by albu.ToTensorV2()
            result['mask'] = np.expand_dims(result['mask'], 0)  # CWH
            result['image'] = np.moveaxis(result['image'], -1, 0)  # CWH

        result['filename'] = image_path.name  # type: ignore

        return cast(DatasetItem, result)


class AnomalyDataset(Dataset):
    """
    - Dataset for anomaly detection autoencoder.
    - Loads images and masks.
    - Masks are used to crop the area of interest from the images.
    """

    def __init__(
        self,
        images: list[Path],
        masks: list[Path],
        transforms: albu.Compose | None = None,
        input_transforms: albu.Compose | None = None,
    ) -> None:
        """Dataset for anomaly detection autoencoder.

        Parameters
        ----------
        images : list[Path]
            List of image paths.
        masks : list[Path]
            List of mask paths. In the context of anomaly detection, masks are used to crop the area of interest from the images.
        transforms : albu.Compose, default=None
            Compose object with albumentations transforms.
        input_transforms : albu.Compose, default=None
            Compose object with albumentations transforms to apply to the input image only.
        """
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.input_transforms = input_transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> DatasetItem:
        image_path = self.images[idx]
        image_pil = Image.open(image_path).convert('L')
        mask_pil = Image.open(self.masks[idx]).convert('L')

        image_pil = crop_driver_image(image_pil, image_path)
        mask_pil = crop_driver_image(mask_pil, image_path)

        image = np.array(image_pil).astype(np.float32)
        mask = (np.array(mask_pil) > 0).astype(np.float32)

        # Apply mask to the image
        image *= mask

        # Normalize to [0, 1]
        image /= 255.0

        # Add channel dimension (HWC), image will be converted to CHW format by ToTensorV2.
        image = np.expand_dims(image, 2)

        if self.transforms is not None:
            # Apply basic transforms to both input and output images
            result = self.transforms(image=image, mask=image)

        if self.input_transforms is not None:
            augmented_image = self.input_transforms(image=image)  # type: ignore
            result['image'] = augmented_image['image']

        return DatasetItem(
            image=result['image'],  # type: ignore
            mask=result['mask'],
            filename=image_path.name,
        )


def get_last_window_index(
    collection_length: int, window_size: int, time_step: int = 1
) -> int:
    """Return the start index of the last window. See `tests/test_dataset.py` for examples."""
    if window_size * time_step > collection_length:
        raise ValueError(
            f'Combination of window size {window_size} and time step {time_step} exceeds collection length.'
        )
    last_index = collection_length - (window_size - 1) * time_step - 1
    return max(last_index, 0)


class TemporalAutoencoderDataset(Dataset):
    def __init__(
        self,
        memory_map_file: Path | str,
        memory_map_image_shape: tuple[int, int] | tuple[int, int, int] = (256, 256),
        window_size: int = 4,
        time_step: int = 1,
        time_dim_index: Literal[0, 1] = 0,
        transforms: albu.Compose | None = None,
        input_transforms: albu.Compose | None = None,
        dtype: type = np.uint8,
    ) -> None:
        """Dataset for temporal autoencoder.

        Parameters
        ----------
        memory_map_file : Path | str
            Path to `numpy.memmap` file (`.dat`).
        memory_map_image_shape : tuple[int, int], default=(256, 256)
            Shape of the images in `memory_map_file`.
        window_size : int, default=4
            Number of frames to include in each sample (sequence lenght).
        time_step : int, default=1
            Number of frames to skip between samples.
        time_dim_index : Literal[0, 1], default=0
            Index of the time dimension in the output tensor,
            1 for (C, T, H, W) and 0 for (T, C, H, W).
        transforms : albu.Compose, default=None
            Compose object with albumentations transforms.
        input_transforms : albu.Compose, default=None
            Compose object with albumentations transforms to apply to the input image only.

        Note
        ----
        Window size and time step:

        ```
        window_size = 2
        time_step = 2

        Input 0 1 2 3 4 5 6 7 8 9
        Seq0  x   x
        Seq1    x   x
        Seq2      x   x
        Seq3        x   x
        ```
        """
        # Remove the channel dimension if it is 1
        if len(memory_map_image_shape) == 3 and memory_map_image_shape[2] == 1:
            memory_map_image_shape = memory_map_image_shape[:2]

        self.memory_map_file = memory_map_file
        self.memory_map = MemMapReader(
            memory_map_file, memory_map_image_shape, dtype=dtype
        )
        self.window_size = window_size
        self.time_step = time_step
        self.time_dim_index = time_dim_index
        self.default_transform_ = T.ToTensor()
        self.transforms = transforms
        self.input_transforms = input_transforms

        # Number of temporal slices in the dataset.
        self.length_ = get_last_window_index(
            len(self.memory_map), window_size, time_step
        )

    def __len__(self) -> int:
        """Number of temporal slices in the dataset (samples)."""
        return self.length_

    def __getitem__(self, idx: int) -> DatasetItem:
        # Memory map is read-only by default, we need mutable copies of the numpy arrays.
        temporal_slice = self.memory_map.window_mut(
            idx, self.window_size, self.time_step
        )
        # list[(C, H, W)]
        temporal_slice = [self.default_transform_(image) for image in temporal_slice]
        # (T, C, H, W)
        temporal_tensor = torch.stack(temporal_slice)  # type: ignore
        if self.time_dim_index == 1:
            # (C, T, H, W)
            temporal_tensor = temporal_tensor.permute(1, 0, 2, 3)

        return DatasetItem(
            image=temporal_tensor,
            mask=temporal_tensor,
            filename=idx,
        )


@dataclass
class VideoInfo:
    """Information about a video file. Helper class for `TemporalAutoencoderDatasetDMD`."""

    memory_map: 'MemMapReader'
    start_idx: int  # Global start index for this video
    length: int  # Number of windows in this video
    path: Path  # Path to the video file


class TemporalAutoencoderDatasetDMD(Dataset):
    def __init__(
        self,
        dataset_directories: Path | str | Sequence[str | Path],
        memory_map_image_shape: tuple[int, int] | tuple[int, int, int] = (256, 256),
        window_size: int = 4,
        time_step: int = 1,
        time_dim_index: Literal[0, 1] = 0,
        transforms: albu.Compose | None = None,
        input_transforms: albu.Compose | None = None,
        source_type: Literal[
            'depth', 'rgb', 'masks', 'rgbd', 'source_depth', 'rgb_source_depth'
        ] = 'depth',
    ) -> None:
        """Dataset for temporal autoencoder supporting multiple video files.

        Each video maintains its independence (frames from different videos are never mixed),
        but the dataset presents a unified interface for accessing all videos sequentially.

        Assumes naming convention: `{memory_map_image_shape[0]}.dat`.
        Looks for files like `256.dat` recursively.

        Parameters
        ----------
        dataset_dir : Path | str
            Directory containing multiple `numpy.memmap` files (`.dat`).
        memory_map_image_shape : tuple[int, int], default=(256, 256)
            Shape of the images in memory map files.
        window_size : int, default=4
            Number of frames to include in each sample (sequence length).
        time_step : int, default=1
            Number of frames to skip between samples.
        time_dim_index : Literal[0, 1], default=0
            Index of the time dimension in the output tensor,
            1 for (C, T, H, W) and 0 for (T, C, H, W).
        transforms : albu.Compose, default=None
            Compose object with albumentations transforms.
        input_transforms : albu.Compose, default=None
            Compose object with albumentations transforms to apply to the input image only.
        """
        # Remove the channel dimension if it is 1
        if len(memory_map_image_shape) == 3 and memory_map_image_shape[2] == 1:
            memory_map_image_shape = memory_map_image_shape[:2]

        self.dataset_directories = (
            [Path(directory) for directory in dataset_directories]
            if isinstance(dataset_directories, Sequence)
            else [Path(dataset_directories)]
        )
        self.window_size = window_size
        self.time_step = time_step
        self.time_dim_index = time_dim_index
        self.default_transform_ = T.ToTensor()
        self.transforms = transforms
        self.input_transforms = input_transforms

        # Load all memory map files and calculate cumulative indices
        self.video_files = sorted(
            video_file
            for directory in self.dataset_directories
            for video_file in directory.rglob(
                f'{source_type}?{memory_map_image_shape[0]}.dat'
            )
        )
        if not self.video_files:
            raise ValueError(f'No .dat files found in {dataset_directories}')
        if len(self.video_files) < len(self.dataset_directories):
            raise ValueError('Some directories do not contain the required .dat files.')

        self.videos: list[VideoInfo] = []
        current_idx = 0

        for video_file in self.video_files:
            memory_map = MemMapReader(video_file, memory_map_image_shape)
            video_length = get_last_window_index(
                len(memory_map), window_size, time_step
            )

            self.videos.append(
                VideoInfo(
                    memory_map=memory_map,
                    start_idx=current_idx,
                    length=video_length,
                    path=video_file,
                )
            )
            current_idx += video_length

        self.length_ = current_idx

    def __len__(self) -> int:
        """Total number of temporal slices across all videos."""
        return self.length_

    def _locate_video(self, idx: int) -> tuple[VideoInfo, int]:
        """Find which video contains the given index and convert to local index."""
        for video in self.videos:
            if idx < video.start_idx + video.length:
                return video, idx - video.start_idx
        raise IndexError(f'Index {idx} is out of bounds')

    def __getitem__(self, idx: int) -> DatasetItem:
        # Find which video contains this index
        video_info, local_idx = self._locate_video(idx)

        # Get the temporal slice from the appropriate video
        temporal_slice = video_info.memory_map.window_mut(
            local_idx, self.window_size, self.time_step
        )

        # Convert to tensors
        temporal_slice = [self.default_transform_(image) for image in temporal_slice]
        temporal_tensor = torch.stack(temporal_slice)  # type: ignore

        if self.time_dim_index == 1:
            temporal_tensor = temporal_tensor.permute(1, 0, 2, 3)

        return DatasetItem(
            image=temporal_tensor,
            mask=temporal_tensor,
            filename=f'{video_info.memory_map.memmap_file}_{local_idx}',
        )


class STAEDataset(Dataset):
    def __init__(
        self,
        memory_map_file: Path | str,
        memory_map_image_shape: tuple[int, int] = (256, 256),
        window_size: int = 4,
        time_step: int = 1,
        transforms: albu.Compose | None = None,
        input_transforms: albu.Compose | None = None,
    ) -> None:
        """Dataset for Spatio-Temporal Autoencoder (STAE). Yields two sequences of frames (input and target) and a set of future frames.

        Parameters
        ----------
        memory_map_file : Path | str
            Path to `numpy.memmap` file (`.dat`).
        memory_map_image_shape : tuple[int, int], default=(256, 256)
            Shape of the images in `memory_map_file`.
        window_size : int, default=4
            Number of frames to include in each sample (sequence lenght).
        time_step : int, default=1
            Number of frames to skip between samples.
        time_dim_index : Literal[0, 1], default=0
            Index of the time dimension in the output tensor,
            1 for (C, T, H, W) and 0 for (T, C, H, W).
        transforms : albu.Compose, default=None
            Compose object with albumentations transforms.
        input_transforms : albu.Compose, default=None
            Compose object with albumentations transforms to apply to the input image only.

        Note
        ----
        ```
        window_size = 2  # 2 frames to reconstruct, 2 frames to predict

        Input 0 1 2 3 4 5 6 7 8 9
        Seq0  x x y y
        Seq1    x x y y
        Seq2      x x y y
        ```
        """
        self.memory_map_file = memory_map_file
        self.memory_map = MemMapReader(memory_map_file, memory_map_image_shape)
        self.window_size = window_size
        self.time_step = time_step
        self.time_dim_index = 1
        self.default_transform_ = T.ToTensor()
        self.transforms = transforms
        self.input_transforms = input_transforms

        # Number of temporal slices in the dataset.
        self.length_ = (
            get_last_window_index(len(self.memory_map), window_size, time_step)
            - window_size * time_step
        )

    def __len__(self) -> int:
        """Number of temporal slices in the dataset (samples)."""
        return self.length_

    def _get_temporal_tensor(self, idx: int) -> torch.Tensor:
        # Memory map is read-only by default, we need mutable copies of the numpy arrays.
        temporal_slice = self.memory_map.window_mut(
            idx, self.window_size, self.time_step
        )
        # list[(C, H, W)]
        temporal_slice = [self.default_transform_(image) for image in temporal_slice]
        # (T, C, H, W)
        temporal_tensor = torch.stack(temporal_slice)  # type: ignore
        if self.time_dim_index == 1:
            # (C, T, H, W)
            temporal_tensor = temporal_tensor.permute(1, 0, 2, 3)
        return temporal_tensor

    def __getitem__(self, idx: int) -> DatasetItem:
        return DatasetItem(
            image=self._get_temporal_tensor(idx),
            mask=self._get_temporal_tensor(idx + self.window_size),
            filename=idx,
        )


@dataclass
class DatasetSplit:
    images: list[Path]
    masks: list[Path]


@dataclass
class DatasetPathsLoader:
    """Dataclass to hold the train, valid, and test datasets.
    Automatically validates dataset paths and shapes of images and masks.
    """

    train: DatasetSplit
    valid: DatasetSplit
    test: DatasetSplit
    validate_names: bool = True
    validate_shapes: bool = True

    def __post_init__(self) -> None:
        if self.validate_names:
            self._validate_names()
        if self.validate_shapes:
            self._validate_shapes()

    def _validate_names(self) -> None:
        def validate(images: list[Path], masks: list[Path]) -> bool:
            image_names = sorted([img.stem for img in images])
            mask_names = sorted([mask.stem for mask in masks])
            return image_names == mask_names

        train_validated = validate(self.train.images, self.train.masks)
        valid_validated = validate(self.valid.images, self.valid.masks)
        test_validated = validate(self.test.images, self.test.masks)

        if not all([train_validated, valid_validated, test_validated]):
            raise ValueError('Mismatch in names between images and masks.')
        elif (
            len(self.train.images) < 1
            or len(self.valid.images) < 1
            or len(self.test.images) < 1
        ):
            raise ValueError('Empty dataset. Please check the dataset paths.')
        else:
            logger.info('Dataset paths validated successfully!')

    def _validate_shapes(self) -> None:
        def validate_shapes(images: list[Path], masks: list[Path]) -> bool:
            for img_path, mask_path in zip(images, masks):
                img_shape = get_image_shape(img_path)
                mask_shape = get_image_shape(mask_path)
                if img_shape != mask_shape:
                    return False
            return True

        def get_image_shape(image_path: Path) -> tuple[int, int]:
            img = Image.open(image_path)
            return img.size[::-1]  # type: ignore

        train_validated = validate_shapes(self.train.images, self.train.masks)
        valid_validated = validate_shapes(self.valid.images, self.valid.masks)
        test_validated = validate_shapes(self.test.images, self.test.masks)

        if not all([train_validated, valid_validated, test_validated]):
            raise ValueError('Images and masks do not have the same shape.')
        else:
            logger.info('Shapes of images and masks validated successfully!')

    def __str__(self) -> str:
        return f'Train:\n{self.train}\n\nValid:\n{self.valid}\n\nTest:\n{self.test}'

    def get_loaders(
        self,
        dataset: Literal['segmentation', 'anomaly'] = 'segmentation',
        batch_size_dict: BatchSizeDict = {'train': 8, 'valid': 1, 'test': 1},
        num_workers: int = 4,
        train_transforms: albu.Compose | None = None,
        valid_transforms: albu.Compose | None = None,
        test_transforms: albu.Compose | None = None,
        ae_input_transforms: albu.Compose | None = None,
        pin_memory: bool = True,
    ) -> dict:
        """Create dataloaders for train, valid, and test datasets."""
        if any(
            [
                train_transforms is None,
                valid_transforms is None,
                test_transforms is None,
            ]
        ):
            logger.warning('Transforms not provided. Loaders will return raw images.')

        if dataset == 'segmentation':
            train_dataset = SegmentationDataset(
                images=self.train.images,
                masks=self.train.masks,
                transforms=train_transforms,
            )
            valid_dataset = SegmentationDataset(
                images=self.valid.images,
                masks=self.valid.masks,
                transforms=valid_transforms,
            )
            test_dataset = SegmentationDataset(
                images=self.test.images,
                masks=self.test.masks,
                transforms=test_transforms,
            )
        elif dataset == 'anomaly':
            train_dataset = AnomalyDataset(
                images=self.train.images,
                masks=self.train.masks,
                transforms=train_transforms,
                input_transforms=ae_input_transforms,
            )
            valid_dataset = AnomalyDataset(
                images=self.valid.images,
                masks=self.valid.masks,
                transforms=valid_transforms,
            )
            test_dataset = AnomalyDataset(
                images=self.test.images,
                masks=self.test.masks,
                transforms=test_transforms,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_dict['train'],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size_dict['valid'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_dict['test'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory,
        )

        loaders = OrderedDict()
        loaders['train'] = train_loader
        loaders['valid'] = valid_loader
        loaders['test'] = test_loader

        return loaders
