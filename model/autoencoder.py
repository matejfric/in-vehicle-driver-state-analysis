from typing import Literal

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim


class AutoencoderModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Sequential,
        decoder: nn.Sequential,
        batch_size_dict: dict,
        learning_rate: float = 1e-4,
        loss_function: Literal['fro', 'mse'] = 'mse',
        train_noise_std_input: float = 0.0,
        train_noise_std_latent: float = 0.0,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

        self.encoder = encoder
        self.decoder = decoder

        self.batch_size_dict = batch_size_dict
        self.lr = learning_rate

        self.train_noise_std_input = train_noise_std_input
        self.train_noise_std_latent = train_noise_std_latent

        self.loss_function = loss_function
        self.mse = nn.MSELoss()
        self.fro = lambda x, y: torch.norm(x - y, dim=1, p='fro')
        self.metrics = ['losss', 'mse', 'fro']

        # Buffers for intermediate results (per batch)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Note
        ----
        https://stackoverflow.com/a/59044860
        """
        # Add noise to the input image
        # image += torch.randn_like(image) * self.train_noise_std_input

        print(f'Image shape: {image.shape}')

        encoded = self.encoder(image)

        print("Can't get here ¯\\_(ツ)_/¯")

        # assert encoded.shape == (32, 16, 16, 16), f'encoded.shape: {encoded.shape}'

        # Add noise to the latent space representation
        # encoded += torch.randn_like(encoded) * self.train_noise_std_latent

        decoded = self.decoder(encoded)

        assert decoded.shape == image.shape, f'decoded.shape: {decoded.shape}'

        return decoded

    def shared_step(self, batch: dict[str, torch.Tensor]) -> dict:
        image = batch['image']

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4, f'image.ndim: {image.ndim}'

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, f'h: {h}, w: {w}'

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        mse = self.mse(logits_mask, image)
        fro = self.fro(logits_mask, image)

        print(f'mse: {mse.mean()}, fro: {fro.mean()}')

        loss = mse if self.loss_function == 'mse' else fro

        return {
            'loss': loss,
            'mse': mse,
            'fro': fro,
        }

    def log_metrics(
        self,
        losses: dict[str, torch.Tensor],
        mode: Literal['train', 'val', 'test'],
    ) -> None:
        metrics = {
            f'{mode}_loss': losses['loss'],
            f'{mode}_mse': losses['mse'].mean(),
            f'{mode}_fro': losses['fro'].mean(),
        }
        # `log_dict` is a method from LightningModule
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict[mode],
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self.shared_step(batch)
        # self.training_step_outputs.append(losses["loss"])
        self.log_metrics(losses, 'train')
        return losses['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self.shared_step(batch)
        self.log_metrics(losses, 'val')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        losses = self.shared_step(batch)
        self.log_metrics(losses, 'test')
        return losses

    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    #     return optimizer

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # https://hphuongdhsp.github.io/ml-blog/pytorchlightning/semanticsegmentation/deeplearning/2022/08/04/segmentation-model-part3.html
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs
        )

        return [optimizer], [scheduler]
