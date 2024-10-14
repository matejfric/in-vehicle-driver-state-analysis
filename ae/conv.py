import torch
import torch.nn as nn

from .common import Decoder, Encoder


class ConvEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256->128

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128->64

        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=5, padding=2
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64->32

        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32->16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        return x


class ConvDecoder(Decoder):
    def __init__(self) -> None:
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.relu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.relu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.relu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=7,
            stride=2,
            padding=3,
            output_padding=1,
        )
        self.relu4 = nn.ReLU()

        self.conv_out = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=3, padding='same'
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.relu4(x)

        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x
