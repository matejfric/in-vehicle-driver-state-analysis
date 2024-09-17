import pytorch_lightning as L
import segmentation_models_pytorch as smp
import torch

# Consider changing the evaluation...
# https://hphuongdhsp.github.io/ml-blog/pytorchlightning/semanticsegmentation/deeplearning/2022/08/04/segmentation-model-part3.html#:~:text=valid_losses%20%3D%20torch.stack(%5Bx%5B%22valid_loss%22%5D%20for%20x%20in%20outputs%5D).mean()%0A%20%20%20%20%20%20%20%20valid_dices%20%3D%20torch.stack(%5Bx%5B%22valid_dice%22%5D%20for%20x%20in%20outputs%5D).mean()%0A%20%20%20%20%20%20%20%20valid_ious%20%3D%20torch.stack(%5Bx%5B%22valid_iou%22%5D%20for%20x%20in%20outputs%5D).mean()


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        batch_size_dict: dict,
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

        # Buffers for intermediate results (per batch)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        # assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        # h, w = image.shape[2:]
        # assert h % 32 == 0 and w % 32 == 0

        mask = batch['mask']

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        # assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        # assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode='binary'
        )

        # F1-score also happens to be Sørensen–Dice coefficient
        return {
            'loss': loss,
            'jaccard_index': tp / (tp + fp + fn),
            'f1_score': 2 * tp / (2 * tp + fp + fn),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        losses = self.shared_step(batch)
        # self.training_step_outputs.append(losses["loss"])
        metrics = {
            'train_loss': losses['loss'],
            'train_f1s': losses['f1_score'].mean(),
            'train_jaccard': losses['jaccard_index'].mean(),
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict['train'],
        )
        return losses['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        losses = self.shared_step(batch)
        metrics = {
            'val_loss': losses['loss'],
            'val_f1s': losses['f1_score'].mean(),
            'val_jaccard': losses['jaccard_index'].mean(),
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict['valid'],
        )

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        losses = self.shared_step(batch)
        metrics = {
            'test_loss': losses['loss'],
            'test_f1s': losses['f1_score'].mean(),
            'test_jaccard': losses['jaccard_index'].mean(),
        }
        self.log_dict(
            metrics,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size_dict['test'],
        )
        return losses

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
