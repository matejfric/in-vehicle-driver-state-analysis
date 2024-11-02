from typing import Any, Literal

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from model.common import BatchSizeDict, ModelStages

from .common import Decoder, Encoder, TimeDistributed


class TemporalAutoencoderModel(L.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        batch_size_dict: BatchSizeDict,
        learning_rate: float = 0.0005,  # 5e-4 (Adam default is 1e-3)
        loss_function: Literal['mae', 'mse'] = 'mse',
        time_dim_index: Literal[1, 2] = 1,
        eps: float = 1e-07,  # Adam default is 1e-8
        train_noise_std_input: float = 0.0,
        train_noise_std_latent: float = 0.0,
        **kwargs: dict[Any, Any],
    ) -> None:
        """Temporal autoencoder model.

        Assumes that the input tensor has shape `(batch_size, temporal_dim, channels, height, width)` or
        `(batch_size, channels, temporal_dim, height, width)` depending on the `time_dim_index`.

        Parameters
        ----------
        encoder : Encoder
            Encoder model.
        decoder : Decoder
            Decoder model.
        batch_size_dict : dict
            Dictionary with batch sizes for `train`, `valid`, and `test` datasets.
        learning_rate : float, default=1e-4
        loss_function : {'fro', 'mse'}, default='mse'
        time_dim_index : int, default=1
            Index of the temporal dimension in the input tensor.
            If `1`, the input tensor has shape `(batch_size, temporal_dim, channels, height, width)`.
            If `2`, the input tensor has shape `(batch_size, channels, temporal_dim, height, width)`.
        train_noise_std_input: float, default=0.0
            Standard deviation of the Gaussian noise added to the input image.
        train_noise_std_latent: float, default=0.0
            Standard deviation of the Gaussian noise added to the latent space representation.
        kwargs : dict
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

        self.encoder = encoder
        self.decoder = decoder
        self.batch_size_dict = batch_size_dict
        self.lr = learning_rate
        self.eps = eps
        self.time_dim_index = time_dim_index
        self.train_noise_std_input = train_noise_std_input
        self.train_noise_std_latent = train_noise_std_latent

        loss_functions = {'mae': nn.SmoothL1Loss(), 'mse': nn.MSELoss()}
        self.loss_function_ = loss_functions[loss_function]
        self.metrics_ = dict(
            mae=mean_absolute_error,
            mse=mean_squared_error,
            fro=lambda x, y: torch.norm(x - y, dim=1, p='fro'),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Note
        ----
        Gaussian noise in torch: https://stackoverflow.com/a/59044860
        """
        # Add noise to the input image
        image += torch.randn_like(image) * self.train_noise_std_input
        encoded = self.encoder(image)

        # Add noise to the latent space representation
        encoded = encoded + torch.randn_like(encoded) * self.train_noise_std_latent
        decoded = self.decoder(encoded)

        return decoded

    def shared_step(self, batch: dict[str, torch.Tensor]) -> dict:
        images = batch['image']  # input images
        outputs = batch['mask']  # ground truth
        reconstructed = self.forward(images)  # reconstructed images

        # The tensors have shape `(batch_size, temporal_dim, channels, height, width)`
        # We need to compute the loss and metrics for each temporal dimension
        # and then average them.

        loss = 0
        metrics = {k: 0 for k in self.metrics_}
        n_time_steps = images.shape[self.time_dim_index]

        for t in range(n_time_steps):
            # This select one slice of the tensor along the temporal dimension.
            # The `slice(None)` is equivalent to `:` in numpy.
            slice_index = (slice(None),) * self.time_dim_index + (t,)  # e.g.: `:, t`

            # The `contiguous` call is not the most efficient, but `torchmetrics` are using
            # `view` internally in places where `reshape` should be used.
            recon_slice = reconstructed[slice_index].contiguous()
            output_slice = outputs[slice_index].contiguous()

            # Accumulate loss for each temporal slice
            loss += self.loss_function_(recon_slice, output_slice)

            # Accumulate metrics for each temporal slice
            for k, metric_fn in self.metrics_.items():
                metrics[k] += metric_fn(recon_slice, output_slice)  # type: ignore

        # Average loss and metrics across the temporal dimension
        loss /= n_time_steps
        metrics = {k: v / n_time_steps for k, v in metrics.items()}

        return {
            'loss': loss,
            **metrics,
        }

    def log_metrics(
        self,
        losses: dict[str, torch.Tensor],
        mode: ModelStages,
    ) -> None:
        metrics = {
            f'{mode}_loss': losses['loss'],
            **{f'{mode}_{k}': v.mean() for k, v in losses.items() if k != 'loss'},
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict[mode],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self.shared_step(batch)
        self.log_metrics(losses, 'train')
        return losses['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self.shared_step(batch)
        self.log_metrics(losses, 'valid')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        losses = self.shared_step(batch)
        self.log_metrics(losses, 'test')
        return losses

    def configure_optimizers(self) -> torch.optim.Optimizer:  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=self.eps)  # type: ignore
        return optimizer


class LSTMEncoder(Encoder):
    def __init__(
        self,
        n_time_steps: int = 2,
        n_filters: int = 8,
        n_image_channels: int = 1,
        n_fc_neurons: int = 128,
        n_lstm_neurons: int = 128,
        latent_dim: int = 48,
        bidirectional: bool = True,
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
            TimeDistributed(
                nn.LazyConv2d(2 * n_filters, kernel_size=3, padding=1),
                n_time_steps,
            ),
            TimeDistributed(nn.ReLU(), n_time_steps),
            TimeDistributed(nn.MaxPool2d(2), n_time_steps),
        )

        self.time_fc = TimeDistributed(nn.LazyLinear(n_fc_neurons), n_time_steps)
        self.time_fc_relu = TimeDistributed(nn.ReLU(), n_time_steps)

        self.lstm = nn.LSTM(
            # When `batch_first=True`, the input shape is `(batch_size, n_time_steps, input_features)`,
            # and the `input_size` of the LSTM is expected to be `input_features`
            # (NOT multiplied by `n_time_steps`).
            n_fc_neurons,
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
        x = self.conv_layers(x)

        # Flatten the images
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size, time_steps, -1)

        x = self.time_fc(x)
        x = self.time_fc_relu(x)

        # `last_timestamp` shape is `(batch_size, 1, 2 * n_fc_neurons)` (2 for bidirectional)
        # _whole_seq, (last_timestamp, _cell_state) = self.lstm(x)
        _, (x, _) = self.lstm(x)
        # x = self.lstm_relu(last_timestamp)

        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, x.shape[1:].numel())
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
                nn.LazyLinear(n_image_channels * image_size * image_size // 16),
                n_time_steps,
            ),
        )
        self.conv_layers = nn.Sequential(
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    n_filters, kernel_size=3, padding=1, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(
                    n_filters, kernel_size=5, padding=2, stride=2, output_padding=1
                ),
                n_time_steps,
            ),
            TimeDistributed(
                nn.LazyConvTranspose2d(n_image_channels, kernel_size=5, padding=2),
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
