import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch

from .common import BatchSizeDict, ModelStages


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        batch_size_dict: BatchSizeDict,
        freeze_encoder: bool = True,
        encoder_weights: str | None = 'imagenet',
        learning_rate: float = 1e-4,
        **kwargs: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # saves HPs in each checkpoint (arch, encoder_name, in_channels, out_classes, **kwargs)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.batch_size_dict = batch_size_dict
        self.lr = learning_rate

        if freeze_encoder:
            # https://github.com/qubvel/segmentation_models.pytorch/issues/793
            # https://github.com/qubvel/segmentation_models.pytorch/issues/79
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch: dict) -> dict:
        image = batch['image']
        mask = batch['mask']

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(),  # type: ignore
            mask.long(),
            mode='binary',
        )

        # F1-score also happens to be Sørensen–Dice coefficient
        return {
            'loss': loss,
            'jaccard_index': (tp).float() / (tp + fp + fn),
            'f1_score': 2 * (tp).float() / (2 * tp + fp + fn),
            'precision': tp.float() / (tp + fp),
            'recall': tp.float() / (tp + fn),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def log_metrics(
        self,
        losses: dict[str, torch.Tensor],
        mode: ModelStages,
    ) -> None:
        metrics = {
            f'{mode}_loss': losses['loss'],
            **{
                f'{mode}_{k}': v.mean(dtype=torch.float)
                for k, v in losses.items()
                if k != 'loss'
            },
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)  # type: ignore
