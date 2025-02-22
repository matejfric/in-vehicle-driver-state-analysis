import albumentations as albu
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def pre_transforms(image_size: int) -> list:
    return [
        albu.Resize(image_size, image_size, p=1, interpolation=cv2.INTER_NEAREST),
        # albu.ToGray(p=1),
    ]


def hard_transforms() -> list:
    """Apply a composition of hard augmentations.

    References
    ----------
    https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/
    https://albumentations.ai/docs/api_reference/full_reference/?h=horizon#albumentations.augmentations.geometric.transforms.HorizontalFlip
    """
    result = [
        albu.CoarseDropout(
            mask_fill_value=0,
            num_holes_range=(1, 3),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.3,
        ),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        albu.HorizontalFlip(p=0.3),
    ]

    return result


def post_transforms() -> list:
    # Image is normalized in CorrosionModel
    return [ToTensorV2(transpose_mask=True, p=1)]


def compose_transforms(transforms: list) -> albu.Compose:
    # Combine all augmentations into single pipeline
    return albu.Compose([item for sublist in transforms for item in sublist])
