import logging
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


class ONNXModel:
    def __init__(
        self, path: str | Path, device: str | torch.device | None = None
    ) -> None:
        """Initialize the ONNX model.

        Parameters
        ----------
        path: str | Path
            Path to the ONNX model file.

        Notes
        -----
        CUDA preferred, fallback to CPU.
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if isinstance(device, torch.device):
            device = device.type
        if device == 'cuda':
            logger.info('Using GPU for inference.')
            if not torch.cuda.is_available():
                msg = 'CUDA is not available. Please check your CUDA installation. Fallback to CPU.'
                warnings.warn(msg)
                logger.warning(msg)
        elif device == 'cpu':
            providers = ['CPUExecutionProvider']
            logger.info('Using CPU for inference.')

        self.model = ort.InferenceSession(
            path,
            providers=providers,
        )
        self.path = path
        self.input_name = self.model.get_inputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.input_type = self.model.get_inputs()[0].type
        self.device = device

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.model.run(None, {self.input_name: images})[0]

    def __repr__(self) -> str:
        """String representation of the ONNX model."""
        return f'ONNXModel(path={self.path}, input_shape={self.input_shape}, input_type={self.input_type}, device={self.device})'


def preprocess_depth_anything_v2(img: np.ndarray) -> np.ndarray:
    """Preprocess the input image for Depth Anything v2 model.

    Assumes the input image is in the format (H, W, C) with pixel
    values in [0, 255] with size (518, 518, 3).
    """
    img = np.expand_dims(img, axis=0)

    # Convert from (B, H, W, C) to (B, C, H, W)
    img = img.transpose(0, 3, 1, 2)
    img = img.astype(np.float32)

    # Rescale values from [0, 255] to [0, 1]
    img = img * 0.00392156862745098  # ~ 1/255

    # Normalize with mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    img = (img - mean) / std

    return img.astype(np.float32)


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize the input array to [0, 1] range."""
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min())


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function to the input array."""
    x = x.astype(np.float32)
    return 1 / (1 + np.exp(-x))
