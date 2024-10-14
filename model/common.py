from pathlib import Path

from PIL import Image


def crop_driver_image(image: Image.Image, image_path: Path) -> Image.Image:
    """Crop from the left to create a square crop while maintaining the height."""
    if image_path.stem.startswith('2021_08_28'):
        # `2021_08_28_radovan_night_mask` different camera placement
        return image.crop((250, 0, 250 + image.size[1], image.size[1]))
    return image.crop((0, 0, image.size[1], image.size[1]))
