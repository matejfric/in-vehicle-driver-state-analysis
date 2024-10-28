from typing import Any, Literal

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from .ae.common import Decoder, Encoder
from .common import BatchSizeDict, ModelStages


class AutoencoderModel(L.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        batch_size_dict: BatchSizeDict,
        learning_rate: float = 1e-4,
        loss_function: Literal['mae', 'mse'] = 'mse',
        train_noise_std_input: float = 0.0,
        train_noise_std_latent: float = 0.0,
        **kwargs: dict[Any, Any],
    ) -> None:
        """Autoencoder model.

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
        train_noise_std_input: float, default=0.0
            Standard deviation of the Gaussian noise added to the input image.
        train_noise_std_latent: float, default=0.0
            Standard deviation of the Gaussian noise added to the latent space representation.
        kwargs : dict
        """
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

        self.encoder = encoder
        self.decoder = decoder

        self.batch_size_dict = batch_size_dict
        self.lr = learning_rate

        self.train_noise_std_input = train_noise_std_input
        self.train_noise_std_latent = train_noise_std_latent

        loss_functions = {'mae': nn.SmoothL1Loss(), 'mse': nn.MSELoss()}
        self.loss_function = loss_functions[loss_function]
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
        image = batch['image']  # input image
        output = batch['mask']  # ground truth
        reconstructed = self.forward(image)  # reconstructed image
        return {
            'loss': self.loss_function(reconstructed, output),
            **{k: v(reconstructed, output) for k, v in self.metrics_.items()},
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

    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    #     return optimizer

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:  # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  # type: ignore

        # https://hphuongdhsp.github.io/ml-blog/pytorchlightning/semanticsegmentation/deeplearning/2022/08/04/segmentation-model-part3.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,  # type: ignore
        )

        return [optimizer], [scheduler]
