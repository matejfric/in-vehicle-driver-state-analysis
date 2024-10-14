import torch
import torch.nn as nn

from .common import Decoder, Encoder

# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html


class FCEncoder(Encoder):
    def __init__(self, image_size: int, hidden_layer_size: int = 512) -> None:
        super().__init__()
        divisibility_constraint = 8
        assert (
            (hls := hidden_layer_size) % divisibility_constraint == 0
        ), f'Hidden layer size must be divisible by {divisibility_constraint}.'
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(image_size**2, hls)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hls, hls // 2)
        self.fc3 = nn.Linear(hls // 2, hls // 4)
        self.fc4 = nn.Linear(hls // 4, hls // 8)
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.relu2(x)
        return x


class FCDecoder(Decoder):
    def __init__(self, image_size: int, hidden_layer_size: int = 512) -> None:
        super().__init__()
        divisibility_constraint = 8
        assert (
            (hls := hidden_layer_size) % divisibility_constraint == 0
        ), f'Hidden layer size must be divisible by {divisibility_constraint}.'
        self.fc1 = nn.Linear(hls // 8, hls // 4)
        self.fc2 = nn.Linear(hls // 4, hls // 2)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hls // 2, hls)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(hls, image_size**2)
        self.unflatten = nn.Unflatten(1, torch.Size([1, image_size, image_size]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu2(x)
        x = self.fc4(x)
        x = self.unflatten(x)
        x = self.sigmoid(x)
        return x
