import torch
import torch.nn as nn
import torch.optim

from .common import Decoder, Encoder

# TODO:
# class Conv3dAutoencoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encoder = Conv3dEncoder()
#         self.decoder = Conv3dDecoder()
#         self.predictor = Conv3dDecoder()

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         predicted = self.predictor(encoded)
#         return decoded, predicted


class Conv3dEncodeBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.LazyConv3d(out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Conv3dDecodeBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.LazyConvTranspose3d(
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3dEncoder(Encoder):
    def __init__(
        self,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        self.conv1 = Conv3dEncodeBlock(32)
        self.conv2 = Conv3dEncodeBlock(48)
        self.conv3 = Conv3dEncodeBlock(64)
        self.conv4 = Conv3dEncodeBlock(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Conv3dDecoder(Decoder):
    def __init__(
        self,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        self.deconv1 = Conv3dDecodeBlock(64)
        self.deconv2 = Conv3dDecodeBlock(48)
        self.deconv3 = Conv3dDecodeBlock(32)
        self.deconv4 = Conv3dDecodeBlock(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x
