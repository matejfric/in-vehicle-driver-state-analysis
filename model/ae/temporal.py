import torch
import torch.nn as nn

from .common import Decoder, Encoder, TimeDistributed


class TemporalAutoencoder(nn.Module):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        n_filters: int = 8,
        n_fc_neurons: int = 128,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
    ) -> None:
        """Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super().__init__()

        self.encoder = LSTMEncoder(
            n_time_steps=n_time_steps,
            n_filters=n_filters,
            n_fc_neurons=n_fc_neurons,
            n_lstm_neurons=n_lstm_neurons,
            latent_dim=latent_dim,
        )
        self.decoder = LSTMDecoder(
            n_time_steps=n_time_steps,
            image_size=image_size,
            n_image_channels=n_image_channels,
            n_filters=n_filters,
            n_lstm_neurons=n_lstm_neurons,
            latent_dim=latent_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LSTMEncoder(Encoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        n_filters: int = 8,
        n_fc_neurons: int = 128,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            TimeDistributed(
                nn.LazyConv2d(n_filters, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
            TimeDistributed(
                nn.LazyConv2d(2 * n_filters, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
            TimeDistributed(
                nn.LazyConv2d(2 * n_filters, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )
        self.time_fc = nn.Sequential(
            TimeDistributed(nn.LazyLinear(n_fc_neurons), n_time_steps),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.lstm = nn.LSTM(
            n_fc_neurons,
            n_lstm_neurons,
            n_time_steps,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_relu = nn.ReLU()
        self.fc_latent = nn.LazyLinear(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)

        # Flatten the images
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size, time_steps, -1)

        x = self.time_fc(x)

        # The second element contains LSTM cell's memory and hidden states
        x = self.lstm(x)[0]
        x = self.lstm_relu(x)

        # Flatten the time dimension.
        # The output shape is `(batch_size, n_time_steps * n_lstm_neurons)`.
        x = x.reshape(-1, x.shape[1:].numel())
        x = self.fc_latent(x)
        return x


class LSTMDecoder(Decoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        image_size: int = 256,
        n_image_channels: int = 1,
        n_filters: int = 8,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
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
            n_time_steps,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_relu = nn.ReLU()
        self.time_fc = nn.Sequential(
            TimeDistributed(
                nn.LazyLinear(n_image_channels * image_size * image_size // 16),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )
        self.conv_layers = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    n_filters, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    n_filters, kernel_size=5, padding=2, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(
                nn.LazyConvTranspose2d(n_image_channels, kernel_size=5, padding=2),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the latent space.
        # The output shape is `(batch_size, n_time_steps, latent_dim)`.
        x = x.unsqueeze(1).repeat(1, self.n_time_steps, 1)

        x = self.lstm(x)[0]
        x = self.lstm_relu(x)
        x = self.time_fc(x)

        x = x.view(
            -1,
            self.n_time_steps,
            self.n_image_channels,
            self.image_size // 4,
            self.image_size // 4,
        )

        x = self.conv_layers(x)

        x = x.view(
            -1,
            self.n_time_steps,
            self.n_image_channels,
            self.image_size,
            self.image_size,
        )

        return x
