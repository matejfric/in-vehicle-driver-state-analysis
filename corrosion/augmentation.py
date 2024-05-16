import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
import cv2


def pre_transforms(image_size: int = 256):
    return [albu.Resize(image_size, image_size, p=1, interpolation=cv2.INTER_NEAREST)]


def hard_transforms():
    """Apply a composition of hard augmentations.

    References
    ----------
    https://albumentations.ai/docs/api_reference/augmentations/dropout/coarse_dropout/
    """
    result = [
        albu.RandomRotate90(),
        albu.CoarseDropout(mask_fill_value=0),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        albu.GridDistortion(p=0.3),
    ]

    return result


def resize_transforms(image_size: int = 256, pre_size: int = 512) -> list:
    """Apply a standard resize or a random crop.

    Parameters:
    -----------
    image_size : int, optional
        The desired size of the output image after transformation.
        Default is `IMAGE_SIZE` (256).

    pre_size : int, optional
        The size of the image before transformation.
        Default is 512.

    Note
    ----
    Random crop will crop on an (image_size, image_size) area
    from a (pre_size, pre_size) area.
    """
    # pre_size = pre_size # input is a 2k image
    # pre_size = int(image_size * 1.5)

    random_crop = albu.Compose(
        [
            albu.SmallestMaxSize(pre_size, p=1, interpolation=cv2.INTER_NEAREST),
            albu.RandomCrop(image_size, image_size, p=1),
        ]
    )

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    # This would be the same as random_crop since we have square images
    # random_crop_big = albu.Compose([
    #   albu.LongestMaxSize(pre_size, p=1),
    #   albu.RandomCrop(
    #       image_size, image_size, p=1
    #   )
    # ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf(
            [
                random_crop,
                rescale,
                # random_crop_big
            ],
            p=1,
        )
    ]

    return result


def post_transforms():
    # Image is normalized in CorrosionModel
    return [ToTensorV2(transpose_mask=True, p=1)]


def compose_transforms(transforms):
    # Combine all augmentations into single pipeline
    result = albu.Compose(
        [item for sublist in transforms for item in sublist]
    )
    return result
