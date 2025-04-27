from collections.abc import Callable
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image

from .common import (
    crop_driver_image_contains,
    pad_to_square,
    pad_to_square_cv,
    preprocess_515,
    preprocess_515_cv,
    preprocess_depth_frame,
)


class PreprocessingFunctionBuilder:
    """Builder for creating preprocessing functions with various options.

    This builder creates functions with the signature:
    (image_path: Path, resize: tuple[int, int]) -> np.ndarray
    """

    def __init__(self) -> None:
        self._steps: list[Callable] = []
        self._input_format: str = 'pil'
        self._output_dtype: type = np.uint8
        self._normalize: bool = False
        self._resize_mode: str = 'pillow'
        self._interpolation = cv.INTER_AREA  # Default for OpenCV

    def from_pillow(self) -> 'PreprocessingFunctionBuilder':
        """Use PIL to load the image."""
        self._input_format = 'pil'
        return self

    def from_opencv(self, unchanged: bool = False) -> 'PreprocessingFunctionBuilder':
        """Use OpenCV to load the image."""
        self._input_format = 'cv'
        self._cv_unchanged = unchanged
        return self

    def crop_driver(self) -> 'PreprocessingFunctionBuilder':
        """Crop the image using driver_image_contains."""
        self._steps.append(lambda img, path: crop_driver_image_contains(img, path))
        return self

    def pad_square(self) -> 'PreprocessingFunctionBuilder':
        """Pad the image to a square shape."""
        if self._input_format == 'pil':
            self._steps.append(lambda img, _: pad_to_square(img))
        else:
            self._steps.append(lambda img, _: pad_to_square_cv(img))
        return self

    def preprocess_515(
        self, use_opencv: bool = False
    ) -> 'PreprocessingFunctionBuilder':
        """Apply Intel 515 preprocessing."""
        if use_opencv:
            self._steps.append(
                lambda img, _: preprocess_515_cv(img, opening_kernel_size=8)
            )
            self._input_format = 'cv'
            self._output_dtype = np.float32
            self._normalize = True
        else:
            self._steps.append(lambda img, _: preprocess_515(img))
            self._normalize = True
            self._output_dtype = np.float32
        return self

    def apply_mask(
        self,
        binary: bool = True,
        mask_in_parent: bool = True,
        ir: bool = False,
        ir_dir_name: str = 'ir',
    ) -> 'PreprocessingFunctionBuilder':
        """Apply a mask to the image.

        Args:
            binary: Whether to use binary masking (>0)
            mask_in_parent: If True, mask is in parent/masks/, otherwise in same directory
        """

        def _apply_mask(img: Image.Image | np.ndarray, path: Path) -> np.ndarray:
            if mask_in_parent:
                mask_path = (
                    path.parent.parent
                    / (f'masks_{ir_dir_name}' if ir else 'masks')
                    / path.with_suffix('.png').name
                )
            else:
                mask_path = path.with_suffix('.mask.png')

            mask_pil = Image.open(mask_path).convert('L')
            if isinstance(img, Image.Image):  # hasattr(img, 'size'):  # PIL Image
                mask_pil = mask_pil.resize(img.size)
                mask = np.array(mask_pil)
            else:  # NumPy array
                mask_pil = mask_pil.resize((img.shape[1], img.shape[0]))
                mask = np.array(mask_pil)

            if binary:
                mask = (mask > 0).astype(np.float32)

            img_array = np.array(img).astype(np.float32)

            if len(img_array.shape) == 3 and len(mask.shape) == 2:
                mask = mask[..., np.newaxis]

            return img_array * mask

        self._steps.append(_apply_mask)
        return self

    def multiply(self, value: float) -> 'PreprocessingFunctionBuilder':
        """Multiply the image by a value."""
        self._steps.append(lambda img, _: np.array(img) * value)
        return self

    def add_depth_channel(
        self,
        depth_in_parent: bool = True,
        depth_dir_name: str = 'depth',
        depth_threshold: int = 2000,
    ) -> 'PreprocessingFunctionBuilder':
        """Add a depth channel to the image.

        Parameters
        ----------
        depth_in_parent: bool
            If True, depth is in `.parent/depth/`, otherwise in same directory.
        depth_dir_name: str
            Name of the directory containing depth images.
        depth_threshold: int
            Threshold for depth values in milimeters. Anything above this value is set to 0.
        """

        def _add_depth(img: Image.Image | np.ndarray, path: Path) -> np.ndarray:
            if depth_in_parent:
                depth_path = (
                    path.parent.parent / depth_dir_name / path.with_suffix('.png').name
                )
            else:
                depth_path = path.with_suffix('.depth.png')

            if depth_dir_name == 'depth':
                # MDE (Depth Anything)
                depth_pil = Image.open(depth_path).convert('L')
            else:
                # Depth sensor
                depth_cv = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)
                depth_pil = Image.fromarray(
                    preprocess_depth_frame(
                        frame=depth_cv, depth_threshold=depth_threshold
                    )
                )

            if isinstance(img, np.ndarray):
                depth_pil = depth_pil.resize((img.shape[1], img.shape[0]))
                depth = np.array(depth_pil)[..., np.newaxis]
                return np.concatenate([img, depth], axis=-1)
            else:
                # If img is still PIL, convert to numpy first
                depth_pil = depth_pil.resize(img.size)
                depth = np.array(depth_pil)[..., np.newaxis]
                img_array = np.array(img)
                return np.concatenate([img_array, depth], axis=-1)

        self._steps.append(_add_depth)
        return self

    def add_mask_channel(
        self, binary: bool = True, mask_in_parent: bool = True
    ) -> 'PreprocessingFunctionBuilder':
        """Add a mask channel to the image instead of applying it.

        Args:
            binary: Whether to use binary masking (>0)
            mask_in_parent: If True, mask is in parent/masks/, otherwise in same directory
        """

        def _add_mask_channel(img: Image.Image | np.ndarray, path: Path) -> np.ndarray:
            if mask_in_parent:
                mask_path = path.parent.parent / 'masks' / path.with_suffix('.png').name
            else:
                mask_path = path.with_suffix('.mask.png')

            mask_pil = Image.open(mask_path).convert('L')

            if isinstance(img, np.ndarray):
                mask_pil = mask_pil.resize((img.shape[1], img.shape[0]))
            else:
                mask_pil = mask_pil.resize(img.size)
                img = np.array(img)

            mask = np.array(mask_pil)
            if binary:
                mask = (mask > 0).astype(np.float32)

            mask = mask[..., np.newaxis]
            return np.concatenate([img, mask], axis=-1)

        self._steps.append(_add_mask_channel)
        return self

    def normalize(self) -> 'PreprocessingFunctionBuilder':
        """Normalize the image to [0, 1] range."""
        self._normalize = True
        self._output_dtype = np.float32
        return self

    def output_as(self, dtype: type) -> 'PreprocessingFunctionBuilder':
        """Set the output data type."""
        self._output_dtype = dtype
        return self

    def use_opencv_resize(
        self, interpolation: int = cv.INTER_AREA
    ) -> 'PreprocessingFunctionBuilder':
        """Use OpenCV for resizing with specified interpolation."""
        self._resize_mode = 'opencv'
        self._interpolation = interpolation
        return self

    def build(self) -> Callable[[Path, tuple[int, int]], np.ndarray]:
        """Build the preprocessing function."""
        steps = self._steps.copy()
        input_format = self._input_format
        output_dtype = self._output_dtype
        normalize = self._normalize
        resize_mode = self._resize_mode
        interpolation = self._interpolation
        cv_unchanged = getattr(self, '_cv_unchanged', False)

        def preprocessing_function(
            image_path: Path, resize: tuple[int, int]
        ) -> np.ndarray:
            # Load the image
            if input_format == 'pil':
                img = Image.open(image_path)
            else:  # OpenCV
                if cv_unchanged:
                    img = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
                    if img.dtype == np.uint16:
                        img = (img / np.iinfo(np.uint16).max).astype(np.float32)
                else:
                    img = cv.imread(str(image_path))

            # Apply all preprocessing steps
            for step in steps:
                img = step(img, image_path)

            # Resize the image
            if resize_mode == 'pillow' and isinstance(img, Image.Image):
                img = img.resize(resize)
            elif resize_mode == 'pillow' and isinstance(img, np.ndarray):
                # Convert numpy to PIL for resizing
                if len(img.shape) == 3 and img.shape[2] == 4:
                    # Handle RGBA
                    temp_img = Image.fromarray(img.astype(np.uint8), 'RGBA')
                elif len(img.shape) == 3 and img.shape[2] == 3:
                    # Handle RGB
                    temp_img = Image.fromarray(img.astype(np.uint8), 'RGB')
                elif len(img.shape) == 3 and img.shape[2] == 1:
                    # Handle grayscale with channel dimension
                    temp_img = Image.fromarray(img[:, :, 0].astype(np.uint8), 'L')
                elif len(img.shape) == 2:
                    # Handle grayscale
                    temp_img = Image.fromarray(img.astype(np.uint8), 'L')
                else:
                    # Try to handle other formats
                    temp_img = Image.fromarray(img.astype(np.uint8))

                img = np.array(temp_img.resize(resize))
            elif resize_mode == 'opencv':
                if isinstance(img, Image.Image):
                    img = np.array(img)
                img = cv.resize(img, resize, interpolation=interpolation)

            # Ensure img is numpy array
            if isinstance(img, Image.Image):
                img = np.array(img)

            # Normalize if required
            if normalize:
                if img.max() > 0:
                    img = img / img.max()

            # Convert to desired output dtype
            return img.astype(output_dtype)

        return preprocessing_function

    @classmethod
    def get_recipe(cls, name: str) -> Callable[[Path, tuple[int, int]], np.ndarray]:
        """Get a predefined preprocessing function by name."""
        builder = cls()

        if name == 'resize_driver':
            return builder.from_pillow().build()

        elif name == 'intel_515_cv':
            return (
                builder.from_opencv(unchanged=True)
                .preprocess_515(use_opencv=True)
                .use_opencv_resize()
                .output_as(np.float32)
                .build()
            )

        elif name == 'intel_515':
            return (
                builder.from_pillow()
                .preprocess_515()
                .normalize()
                .output_as(np.float32)
                .build()
            )

        elif name == 'crop_resize_driver':
            return builder.from_pillow().crop_driver().build()

        elif name == 'mask_resize_driver':
            return builder.from_pillow().apply_mask().build()

        elif name == 'crop_mask_resize_driver':
            return builder.from_pillow().crop_driver().apply_mask().build()

        elif name == 'binary_mask_resize':
            return (
                builder.from_pillow()
                .crop_driver()
                .apply_mask()
                .multiply(255)
                .output_as(np.uint8)
                .build()
            )

        elif name == 'rgbd_resize_crop_driver':
            return (
                builder.from_pillow()
                .crop_driver()
                .apply_mask()
                .add_depth_channel()
                .build()
            )

        elif name == 'rgbdm_resize_crop_driver':
            return (
                builder.from_pillow()
                .crop_driver()
                .add_depth_channel()
                .add_mask_channel()
                .build()
            )

        else:
            raise ValueError(f'Unknown preprocessing recipe: {name}')
