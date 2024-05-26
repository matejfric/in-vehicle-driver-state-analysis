from io import BytesIO
from pathlib import Path

import dagshub
import mlflow.pytorch
import numpy as np
import requests
import torch
from PIL import Image

from corrosion.augmentation import (
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


def preprocess_image(image: Image.Image, input_size: int = 256) -> torch.Tensor:
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


def load_image(image_path: str | Path) -> Image.Image:
    return Image.open(Path(image_path)).convert('RGB')


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image


def predict(
    model: torch.nn.Module, image: Image.Image, input_size: int = 256
) -> torch.Tensor:
    tensor = preprocess_image(image, input_size)
    with torch.no_grad():
        prediction = model(tensor)
    mask = prediction.sigmoid()
    return mask
