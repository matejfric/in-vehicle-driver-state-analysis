import time
from io import BytesIO
from pathlib import Path

import dagshub
import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from model.augmentation import (
    compose_transforms,
    post_transforms,
    pre_transforms,
)


def init_dagshub(repo_name: str = 'corrosion', repo_owner: str = 'matejfric') -> None:
    dagshub.init(repo_name, repo_owner, mlflow=True)


def load_model_from_dagshub(model_name: str, model_version: int) -> torch.nn.Module:
    model_uri = f'models:/{model_name}/{model_version}'
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model


def preprocess_image(
    image: Image.Image, input_size: int = 256, apply_square_crop: bool = True
) -> torch.Tensor:
    # Crop from the left to create a square crop while maintaining the height
    # if apply_square_crop:
    #     image = image.crop((0, 0, image.size[1], image.size[1]))
    preprocess = compose_transforms(
        [
            pre_transforms(image_size=input_size),
            post_transforms(),
        ]
    )
    image = preprocess(image=np.asarray(image))['image']
    # Add batch dimension
    image = image.unsqueeze(0)  # type: ignore
    return image  # type: ignore


def load_image(image_path: str | Path, apply_square_crop: bool = True) -> Image.Image:
    image = Image.open(Path(image_path)).convert('RGB')
    # Crop from the left to create a square crop while maintaining the height
    if apply_square_crop:
        image = image.crop((0, 0, image.size[1], image.size[1]))
    return image


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image


def predict(
    model: torch.nn.Module,
    image: Image.Image,
    input_size: int = 256,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Perform inference on the given image using the model.
    
    Parameters
    ----------
    output_size : tuple[int, int] | None
        Optionally interpolate the output mask to the given size.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device (GPU/CPU)
    model = model.to(device)

    # Preprocess the image and move the tensor to the device
    tensor = preprocess_image(image, input_size).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(tensor)

    # Apply sigmoid activation to get the mask
    mask = prediction.sigmoid()

    if output_size is not None:
        mask = F.interpolate(
            mask, size=output_size, mode='bilinear', align_corners=False
        )

    return mask.to('cpu')


def measure_model_fps(
    model: torch.nn.Module, dataset_path: Path, input_size: int = 256
) -> None:
    """Measure the FPS of the model on the given dataset.

    Example
    -------
    >>> measure_fps(model, DATASET_PATH)

    Note
    ----
    Batch inference is not supported in this function, but it could be faster.
    """
    total_inference_time = 0
    total_loading_time = 0
    num_images = 0

    # Iterate through images in the dataset path
    for image_path in sorted(dataset_path.glob('*.jpg')):
        # Measure image loading time
        start_loading_time = time.time()
        image = Image.open(image_path)
        end_loading_time = time.time()

        # Measure inference time
        start_inference_time = time.time()
        mask = predict(model, image, input_size=input_size)
        end_inference_time = time.time()

        # Accumulate the times
        total_loading_time += end_loading_time - start_loading_time
        total_inference_time += end_inference_time - start_inference_time
        num_images += 1

    # Calculate FPS for inference and image loading
    inference_fps = num_images / total_inference_time if total_inference_time > 0 else 0
    loading_fps = num_images / total_loading_time if total_loading_time > 0 else 0

    # Print results
    print(f'Processed {num_images} images.')
    print(f'Total image loading time: {total_loading_time:.2f} seconds')
    print(f'Total inference time: {total_inference_time:.2f} seconds')
    print(f'Image loading FPS: {loading_fps:.2f}')
    print(f'Inference FPS: {inference_fps:.2f}')
