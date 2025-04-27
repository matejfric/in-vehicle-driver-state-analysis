import albumentations as albu
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def pre_transforms(image_size: int) -> list:
    return [
        albu.Resize(image_size, image_size, p=1, interpolation=cv2.INTER_NEAREST),
    ]


def hard_transforms() -> list:
    """Apply a composition of hard augmentations.

    References
    ----------
    https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/
    https://albumentations.ai/docs/api_reference/full_reference/?h=horizon#albumentations.augmentations.geometric.transforms.HorizontalFlip
    """
    return [
        # albu.ToGray(p=0.05),
        albu.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.2,
        ),
        albu.RandomBrightnessContrast(p=0.2),
        albu.HorizontalFlip(p=0.2),
        albu.MotionBlur(p=0.1),
        albu.GaussNoise(p=0.1, std_range=(0.1, 0.2)),
        albu.RandomShadow(p=0.1),
        albu.RandomSunFlare(p=0.1),
    ]


def post_transforms() -> list:
    # Image is normalized in the model
    return [ToTensorV2(transpose_mask=True, p=1)]


def compose_transforms(transforms: list) -> albu.Compose:
    # Combine all augmentations into single pipeline
    return albu.Compose([item for sublist in transforms for item in sublist])
