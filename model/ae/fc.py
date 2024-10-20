import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Decoder, Encoder

# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html


class FCEncoder(Encoder):
    def __init__(self, image_size: int, hidden_layer_size: int = 512) -> None:
        super(Encoder, self).__init__()
        divisibility_constraint = 8
        assert (
            (hls := hidden_layer_size) % divisibility_constraint == 0
        ), f'Hidden layer size must be divisible by {divisibility_constraint}.'
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(image_size**2, hls)
        self.fc2 = nn.Linear(hls, hls // 2)
        self.fc3 = nn.Linear(hls // 2, hls // 4)
        self.fc4 = nn.Linear(hls // 4, hls // 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x


class FCDecoder(Decoder):
    def __init__(self, image_size: int, hidden_layer_size: int = 512) -> None:
        super(Decoder, self).__init__()
        divisibility_constraint = 8
        assert (
            (hls := hidden_layer_size) % divisibility_constraint == 0
        ), f'Hidden layer size must be divisible by {divisibility_constraint}.'
        self.fc1 = nn.Linear(hls // 8, hls // 4)
        self.fc2 = nn.Linear(hls // 4, hls // 2)
        self.fc3 = nn.Linear(hls // 2, hls)
        self.fc4 = nn.Linear(hls, image_size**2)
        self.sigmoid = nn.Sigmoid()
        self.unflatten = nn.Unflatten(1, torch.Size([1, image_size, image_size]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.sigmoid(x)
        x = self.unflatten(x)
        return x
