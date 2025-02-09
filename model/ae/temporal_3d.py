from typing import Any, Literal, TypedDict

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from model.common import BatchSizeDict, ModelStages

from .common import Decoder, Encoder


class STAELosses(TypedDict):
    total_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    prediction_loss: torch.Tensor
    regularization_loss: torch.Tensor


class STAELoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-4) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        self.time_dim_index = 2

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructed: torch.Tensor,
        future_predictions: torch.Tensor,
        future_targets: torch.Tensor,
        model_parameters: list[torch.Tensor] | None = None,
    ) -> STAELosses:
        """The tensors have shape `(batch_size, channels, temporal_dim, height, width)`"""

        # Reconstruction loss
        L_recon = self.reconstruction_loss(inputs, reconstructed)

        # Prediction loss with weight-decreasing factor
        T = future_targets.size(self.time_dim_index)  # Number of future frames
        weights = torch.arange(  # e.g., [4, 3, 2, 1] for T=4
            T, 0, -1, dtype=torch.float32, device=future_targets.device
        )
        weights /= weights.sum()  # Normalize weights
        prediction_errors = ((future_predictions - future_targets) ** 2).mean(
            dim=[1, 3, 4]  # Mean over spatial dimensions
        )
        # Mean over the batch dimension and multiply by the weights
        L_pred = (weights * prediction_errors.mean(dim=0)).sum()

        # Regularization term
        L_reg = (
            sum(param.norm(2) for param in model_parameters)
            if model_parameters
            else 0.0
        )

        # Combined loss
        total_loss = L_recon + L_pred + self.lambda_reg * L_reg
        return STAELosses(
            total_loss=total_loss,
            reconstruction_loss=L_recon,
            prediction_loss=L_pred,
            regularization_loss=L_reg,  # type: ignore
        )


class STAEModel(L.LightningModule):
    def __init__(
        self,
        batch_size_dict: BatchSizeDict,
        learning_rate: float = 0.0005,  # 5e-4 (Adam default is 1e-3)
        time_dim_index: Literal[2] = 2,
        eps: float = 1e-07,  # Adam default is 1e-8
        lambda_reg: float = 1e-4,
        use_2d_bottleneck: bool = False,
        regularization: Literal['l2_model_weights', 'l2_encoder_weights'] | None = None,
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

        self.encoder = Conv3dEncoder(use_2d_bottleneck=use_2d_bottleneck)
        self.decoder = Conv3dDecoder(use_2d_bottleneck=use_2d_bottleneck)
        # Decoder is the same as the predictor
        self.predictor = Conv3dDecoder(use_2d_bottleneck=use_2d_bottleneck)

        self.batch_size_dict = batch_size_dict
        self.lr = learning_rate
        self.eps = eps
        self.time_dim_index = time_dim_index
        self.lambda_reg = lambda_reg
        self.regularization = regularization

        self.loss_function = STAELoss(lambda_reg)
        self.metrics_ = dict(
            mae=mean_absolute_error,
            mse=mean_squared_error,
            fro=lambda x, y: torch.norm(x - y, dim=1, p='fro'),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        predicted = self.predictor(encoded)

        return reconstructed, predicted

    def shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        images = batch['image']
        future_images = batch['mask']
        expected_outputs = images.clone()
        reconstructed, predicted = self.forward(images)

        match self.regularization:
            case None:
                regularization = None
            case 'l2_encoder_weights':
                regularization = self.encoder.parameters()
            case 'l2_model_weights':
                regularization = self.parameters()

        losses = self.loss_function(
            expected_outputs, reconstructed, predicted, future_images, regularization
        )

        # The tensors have shape `(batch_size, channels, temporal_dim, height, width)`
        # We need to compute the loss and metrics for each temporal dimension
        # and then average them.

        metrics = {k: 0 for k in self.metrics_}
        n_time_steps = images.shape[self.time_dim_index]

        for t in range(n_time_steps):
            # This select one slice of the tensor along the temporal dimension.
            # The `slice(None)` is equivalent to `:` in numpy.
            slice_index = (slice(None),) * self.time_dim_index + (t,)  # e.g.: `:, t`

            # The `contiguous` call is not the most efficient, but `torchmetrics` are using
            # `view` internally in places where `reshape` should be used.
            recon_slice = reconstructed[slice_index].contiguous()
            output_slice = expected_outputs[slice_index].contiguous()

            # Accumulate metrics for each temporal slice
            for k, metric_fn in self.metrics_.items():
                metrics[k] += metric_fn(recon_slice, output_slice)  # type: ignore

        # Average loss and metrics across the temporal dimension
        metrics = {k: v / n_time_steps for k, v in metrics.items()}

        return losses, metrics  # type: ignore

    def log_metrics(
        self,
        losses: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
        mode: ModelStages,
    ) -> None:
        metrics = {f'{mode}_{k}': v.mean() for k, v in metrics.items()}
        losses = {f'{mode}_{k}': v for k, v in losses.items()}
        self.log_dict(
            metrics | losses,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict[mode],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses, metrics = self.shared_step(batch)
        self.log_metrics(losses, metrics, 'train')
        return losses['total_loss']

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses, metrics = self.shared_step(batch)
        self.log_metrics(losses, metrics, 'valid')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        losses, metrics = self.shared_step(batch)
        self.log_metrics(losses, metrics, 'test')
        return losses | metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=self.eps)  # type: ignore
        return optimizer


class Conv3dEncodeBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.LazyConv3d(out_channels, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()  # nn.ReLU()
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Conv2dEncodeBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.LazyConv2d(out_channels, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()  # nn.ReLU()
        self.pool = nn.MaxPool2d(2)

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
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU()  # nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2dDecodeBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.LazyConvTranspose2d(
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()  # nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv3dEncoder(Encoder):
    def __init__(
        self,
        use_2d_bottleneck: bool = False,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Encoder, self).__init__()

        self.use_2d_bottleneck = use_2d_bottleneck

        self.conv1 = Conv3dEncodeBlock(32)
        self.conv2 = Conv3dEncodeBlock(48)
        self.conv3 = Conv3dEncodeBlock(64)
        self.conv4 = Conv3dEncodeBlock(64)
        self.conv5 = Conv2dEncodeBlock(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if self.use_2d_bottleneck:
            x = x.squeeze(2)
            x = self.conv5(x)
        return x


class Conv3dDecoder(Decoder):
    def __init__(
        self,
        use_2d_bottleneck: bool = False,
    ) -> None:
        """LSTM Encoder for Temporal Autoencoder. Input shape: `(batch_size, n_time_steps, n_image_channels, image_size, image_size)`"""
        super(Decoder, self).__init__()

        self.use_2d_bottleneck = use_2d_bottleneck

        self.deconv0 = Conv2dDecodeBlock(64)  # number of filters???
        self.deconv1 = Conv3dDecodeBlock(64)
        self.deconv2 = Conv3dDecodeBlock(48)
        self.deconv3 = Conv3dDecodeBlock(32)
        self.deconv4 = Conv3dDecodeBlock(16)
        self.conv_out = nn.LazyConv3d(1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_2d_bottleneck:
            x = self.deconv0(x)
            x = x.unsqueeze(2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x
