from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim

from .common import Decoder, Encoder, TimeDistributed


class ISVC23EncoderV1(Encoder):
    def __init__(
        self,
        image_size: int = 128,
        n_time_steps: int = 2,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
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
        bottleneck_input = n_lstm_neurons
        if bidirectional:
            bottleneck_input *= 2
        self.fc_latent = nn.Linear(bottleneck_input, latent_dim)

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
        n_filters: int = 8,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
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
