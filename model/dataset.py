import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import albumentations as albu
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import crop_driver_image


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

    def __getitem__(self, idx: int) -> dict:
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

        return result


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
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image_pil = Image.open(image_path).convert('L')
        mask_pil = Image.open(self.masks[idx]).convert('L')

        image_pil = crop_driver_image(image_pil, image_path)
        mask_pil = crop_driver_image(mask_pil, image_path)

        image = np.array(image_pil).astype(np.float32)
        mask = (np.array(mask_pil) > 0).astype(np.float32)
        image *= mask  # Apply mask to the image
        image /= image.max()  # Normalize to [0, 1]

        result = {'image': image}

        if self.transforms is not None:
            result = self.transforms(**result)  # type: ignore

        result['filename'] = image_path.name  # type: ignore

        return result


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

    def __post_init__(self) -> None:
        self._validate_names()
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
            logging.info('Dataset paths validated successfully!')

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
            logging.info('Shapes of images and masks validated successfully!')

    def __str__(self) -> str:
        return f'Train:\n{self.train}\n\nValid:\n{self.valid}\n\nTest:\n{self.test}'

    def get_loaders(
        self,
        dataset: Literal['segmentation', 'anomaly'] = 'segmentation',
        batch_size: dict = {'train': 8, 'valid': 1, 'test': 1},
        num_workers: int = 4,
        train_transforms: albu.Compose | None = None,
        valid_transforms: albu.Compose | None = None,
        test_transforms: albu.Compose | None = None,
    ) -> dict:
        """Create dataloaders for train, valid, and test datasets."""
        if any(
            [
                train_transforms is None,
                valid_transforms is None,
                test_transforms is None,
            ]
        ):
            logging.warning('Transforms not provided. Loaders will return raw images.')

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
            batch_size=batch_size['train'],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size['valid'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size['test'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        loaders = OrderedDict()
        loaders['train'] = train_loader
        loaders['valid'] = valid_loader
        loaders['test'] = test_loader

        return loaders
