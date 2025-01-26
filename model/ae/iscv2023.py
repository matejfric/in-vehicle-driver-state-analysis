from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim

from .common import Decoder, Encoder, TimeDistributed

# V1: LSTM
# V2: LSTM replaced by dense layers (+ conv layers)
# V3: LSTM with MLP (without conv layers)
# V4: Dense layers

# region V1


class ISVC23EncoderV1(Encoder):
    def __init__(
        self,
        image_size: int = 128,
        n_time_steps: int = 2,
        n_lstm_neurons: int = 128,
        latent_dim: int = 128,
        bidirectional: bool = True,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        assert image_size % 16 == 0, 'Image size must be divisible by 16.'

        self.c1 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(8, kernel_size=7, padding=3),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(8, kernel_size=7, padding=3),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c2 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c3 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c4 = deepcopy(self.c3)

        self.lstm = nn.LSTM(
            # When `batch_first=True`, the input shape is `(batch_size, n_time_steps, input_features)`,
            # and the `input_size` of the LSTM is expected to be `input_features`
            # (NOT multiplied by `n_time_steps`).
            64 * (image_size // 2**4) * (image_size // 2**4),
            n_lstm_neurons,
            batch_first=True,
            bidirectional=bidirectional,
            # Activation functions are applied in the LSTM cell (see LSTM design).
            # Output shape, when bidirectional=True is `(2, batch_size, n_lstm_neurons)`.
        )
        self.fc_latent = nn.Linear(
            n_lstm_neurons * (2 if bidirectional else 1), latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        # # Flatten the images
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size, time_steps, -1)

        # # `last_timestamp` shape is `(batch_size, 1, 2 * n_fc_neurons)` (2 for bidirectional)
        # # _whole_seq, (last_timestamp, _cell_state) = self.lstm(x)
        _, (x, _) = self.lstm(x)
        # # x = self.lstm_relu(last_timestamp)

        x = x.permute(1, 0, 2)  # Swap the batch and time steps
        x = x.reshape(batch_size, x.shape[1:].numel())
        x = self.fc_latent(x)
        return x


class ISVC23DecoderV1(Decoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        n_lstm_neurons: int = 128,
        latent_dim: int = 128,
        bidirectional: bool = True,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        assert image_size % 4 == 0, 'Image size must be divisible by 4.'

        self.n_time_steps = n_time_steps
        self.image_size = image_size
        self.n_image_channels = n_image_channels

        self.lstm = nn.LSTM(
            latent_dim,
            n_lstm_neurons,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.time_fc = nn.Sequential(
            TimeDistributed(
                nn.LazyLinear(128 * (image_size // 2**4) * (image_size // 2**4)),
                n_time_steps,
            ),
        )
        self.ct1 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    128, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(128, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct2 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    64, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct3 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    32, kernel_size=5, padding=2, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct4 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    16, kernel_size=7, padding=3, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            # 1x1 convolution
            TimeDistributed(
                nn.LazyConvTranspose2d(1, kernel_size=7, padding=3),
                n_time_steps,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the latent space.
        # The output shape is `(batch_size, n_time_steps, latent_dim)`.
        x = x.unsqueeze(1).repeat(1, self.n_time_steps, 1)

        x = self.lstm(x)[0]  # (batch_size, n_time_steps, n_lstm_neurons)
        x = self.time_fc(x)

        x = x.view(
            -1,
            self.n_time_steps,
            128,
            self.image_size // 16,
            self.image_size // 16,
        )

        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)

        return x


# endregion
# region V2


class ISVC23EncoderV2(Encoder):
    def __init__(
        self,
        image_size: int = 128,
        n_time_steps: int = 2,
        latent_dim: int = 128,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        assert image_size % 16 == 0, 'Image size must be divisible by 16.'

        self.c1 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(8, kernel_size=7, padding=3),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(8, kernel_size=7, padding=3),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c2 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c3 = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConv2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.c4 = deepcopy(self.c3)

        # No activations.
        self.mlp = nn.Sequential(
            nn.LazyLinear(n_time_steps * 128),
            nn.LazyLinear(latent_dim * 2),
            nn.LazyLinear(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)

        # Flatten
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x


class ISVC23DecoderV2(Decoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        latent_dim: int = 128,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        assert image_size % 4 == 0, 'Image size must be divisible by 4.'

        self.n_time_steps = n_time_steps
        self.image_size = image_size
        self.n_image_channels = n_image_channels

        # No activations.
        self.mlp = nn.Sequential(
            nn.LazyLinear(latent_dim * 2),
            nn.LazyLinear(n_time_steps * 128),
            nn.LazyLinear(n_time_steps * 256),
        )

        self.time_fc = nn.Sequential(
            TimeDistributed(
                nn.LazyLinear(128 * (image_size // 2**4) * (image_size // 2**4)),
                n_time_steps,
            ),
        )
        self.ct1 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    128, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(128, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct2 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    64, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(64, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct3 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    32, kernel_size=5, padding=2, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(32, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.ct4 = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    16, kernel_size=7, padding=3, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            # 1x1 convolution
            TimeDistributed(
                nn.LazyConvTranspose2d(1, kernel_size=7, padding=3),
                n_time_steps,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects input shape: (batch_size, latent_dim)
        batch_size = x.size(0)
        x = self.mlp(x)
        # x = x.unsqueeze(1).repeat(1, self.n_time_steps, 1)
        x = x.view(batch_size, self.n_time_steps, -1)
        x = self.time_fc(x)

        x = x.view(
            -1,
            self.n_time_steps,
            128,
            self.image_size // 16,
            self.image_size // 16,
        )

        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        x = self.ct4(x)

        return x


# endregion
# region V3


class ISVC23EncoderV3(Encoder):
    def __init__(
        self,
        image_size: int = 128,
        n_time_steps: int = 2,
        latent_dim: int = 128,
        n_lstm_neurons: int = 128,
        bidirectional: bool = True,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        self.time_mlp = nn.Sequential(
            TimeDistributed(nn.LazyLinear(1024), n_time_steps),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.LazyLinear(512), n_time_steps),
        )
        self.lstm = nn.LSTM(
            512,
            n_lstm_neurons,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_latent = nn.Linear(
            n_lstm_neurons * (2 if bidirectional else 1), latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (batch_size, n_time_steps, latent_dim)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.time_mlp(x)
        _, (x, _) = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], x.shape[1:].numel())
        x = self.fc_latent(x)
        return x


class ISVC23DecoderV3(Decoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        latent_dim: int = 128,
        n_lstm_neurons: int = 128,
        bidirectional: bool = True,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        assert image_size % 4 == 0, 'Image size must be divisible by 4.'

        self.n_time_steps = n_time_steps
        self.image_size = image_size
        self.n_image_channels = n_image_channels

        self.lstm = nn.LSTM(
            latent_dim,
            n_lstm_neurons,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # No activations.
        self.time_mlp = nn.Sequential(
            TimeDistributed(nn.LazyLinear(1024), n_time_steps),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(
                nn.LazyLinear(image_size**2 * n_image_channels), n_time_steps
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects input shape: (batch_size, latent_dim)
        x = x.unsqueeze(1).repeat(1, self.n_time_steps, 1)
        x = self.lstm(x)[0]
        x = self.time_mlp(x)

        x = x.view(
            -1,
            self.n_time_steps,
            self.n_image_channels,
            self.image_size,
            self.image_size,
        )

        return x


# endregion
# region V4


class ISVC23EncoderV4(Encoder):
    def __init__(
        self,
        image_size: int = 128,
        n_time_steps: int = 2,
        latent_dim: int = 128,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        self.time_mlp = nn.Sequential(
            TimeDistributed(nn.LazyLinear(1024), n_time_steps),  # D1
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.LazyLinear(512), n_time_steps),  # D2
        )
        self.mlp = nn.Sequential(
            nn.LazyLinear(n_time_steps * 128),  # D1
            nn.LazyLinear(latent_dim * 2),  # D2
            nn.LazyLinear(latent_dim),  # D3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape: (batch_size, n_time_steps, features)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.time_mlp(x)
        x = x.reshape(x.shape[0], x.shape[1:].numel())
        x = self.mlp(x)
        return x


class ISVC23DecoderV4(Decoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        latent_dim: int = 128,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        assert image_size % 4 == 0, 'Image size must be divisible by 4.'

        self.n_time_steps = n_time_steps
        self.image_size = image_size
        self.n_image_channels = n_image_channels

        self.mlp = nn.Sequential(
            nn.LazyLinear(latent_dim * 2),
            nn.LazyLinear(n_time_steps * 128),
            nn.LazyLinear(n_time_steps * 256),
        )

        self.time_mlp = nn.Sequential(
            TimeDistributed(nn.LazyLinear(1024), n_time_steps),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(
                nn.LazyLinear(image_size**2 * n_image_channels), n_time_steps
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects input shape: (batch_size, latent_dim)
        x = self.mlp(x)
        x = x.unsqueeze(1).repeat(1, self.n_time_steps, 1)
        x = self.time_mlp(x)

        x = x.view(
            -1,
            self.n_time_steps,
            self.n_image_channels,
            self.image_size,
            self.image_size,
        )

        return x


# endregion
