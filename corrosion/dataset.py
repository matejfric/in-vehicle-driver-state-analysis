from pathlib import Path
from collections import OrderedDict
from typing import Optional, Callable
from dataclasses import dataclass
import logging

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(
        self, images: list[Path], masks: list[Path] | None = None, transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image_pil = Image.open(image_path).convert("RGB")
        image = np.array(image_pil)

        result = {"image": image}

        if self.masks is not None:
            mask_pil = Image.open(self.masks[idx]).convert("L")
            mask = (np.array(mask_pil) > 0).astype(np.float32)
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)
            result["mask"] = np.expand_dims(result["mask"], 0)  # CWH
        else:
            # This is done by albu.ToTensorV2()
            result["mask"] = np.expand_dims(result["mask"], 0)  # CWH
            result["image"] = np.moveaxis(result["image"], -1, 0)  # CWH

        result["filename"] = image_path.name

        return result
    
@dataclass
class SegmentationDatasetSplit:
    images: list[Path]
    masks: list[Path]

@dataclass
class SegmentationDatasetLoader:
    train: SegmentationDatasetSplit
    valid: SegmentationDatasetSplit
    test: SegmentationDatasetSplit

    def __post_init__(self):
        self._validate_names()

    def _validate_names(self):
        def validate(images: list[Path], masks: list[Path]) -> bool:
            image_names = sorted([img.stem for img in images])
            mask_names = sorted([mask.stem for mask in masks])
            return image_names == mask_names

        train_validated = validate(self.train.images, self.train.masks)
        valid_validated = validate(self.valid.images, self.valid.masks)
        test_validated = validate(self.test.images, self.test.masks)

        if not all([train_validated, valid_validated, test_validated]):
            raise ValueError("Mismatch in names between images and masks.")
        else:
            logging.info("Dataset paths validated successfully!")

    def __str__(self):
        return f"Train:\n{self.train}\n\nValid:\n{self.valid}\n\nTest:\n{self.test}"

    def get_loaders(
        self,
        batch_size: dict = {"train": 8, "valid": 1, "test": 1},
        num_workers: int = 4,
        train_transforms_fn: Optional[Callable] = None,
        valid_transforms_fn: Optional[Callable] = None,
        test_transforms_fn: Optional[Callable] = None,
    ) -> dict:

        # Create train dataset
        train_dataset = SegmentationDataset(
            images=self.train.images,
            masks=self.train.masks,
            transforms = train_transforms_fn
        )

        # Create valid dataset
        valid_dataset = SegmentationDataset(
            images=self.valid.images,
            masks=self.valid.masks,
            transforms = valid_transforms_fn
        )

        # Create test dataset
        test_dataset = SegmentationDataset(
            images=self.test.images,
            masks=self.test.masks,
            transforms=test_transforms_fn
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size["train"],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size["valid"],
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size["test"],
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

        loaders = OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader
        loaders["test"] = test_loader

        return loaders