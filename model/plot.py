import random
from pathlib import Path

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from .common import crop_driver_image


def show_examples(name: str, image: np.ndarray, mask: np.ndarray) -> None:
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Image: {name}')

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f'Mask: {name}')


def show(
    index: int,
    images: list[Path],
    masks: list[Path],
    transforms: albu.Compose | None = None,
) -> None:
    image_path = images[index]
    name = image_path.name

    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(masks[index]))

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp['image']
        mask = temp['mask']

    show_examples(name, image, mask)


def show_random(
    images: list[Path], masks: list[Path], transforms: albu.Compose | None = None
) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)


def plot_single_prediction(
    image: Image.Image,
    mask: torch.Tensor,
    figsize: tuple[int, int] = (16, 5),
    threshold: float = 0.5,
    apply_square_crop: bool = True,
) -> None:
    if apply_square_crop:
        # Crop from the left to create a square crop while maintaining the height
        image = image.crop((0, 0, image.size[1], image.size[1]))

    n_cols = 4
    plt.figure(figsize=figsize)

    # Squeeze batch and class dimension (binary classification)
    pr_mask_squeeze = mask.squeeze().numpy()
    image_resized = image.resize(pr_mask_squeeze.shape)  # type: ignore

    plt.subplot(1, n_cols, 1)
    plt.imshow(image_resized)
    plt.title('Input image')
    plt.axis('off')

    plt.subplot(1, n_cols, 2)

    plt.imshow(pr_mask_squeeze, cmap='gray', vmin=0, vmax=1)
    plt.title('Probabilities')
    plt.axis('off')

    plt.subplot(1, n_cols, 3)
    plt.imshow(np.where(pr_mask_squeeze > 0.5, 1, 0), cmap='gray', vmin=0, vmax=1)
    plt.title('Binary Prediction')
    plt.axis('off')

    plt.subplot(1, n_cols, 4)
    plt.imshow(image_resized)
    plt.title('Overlay')
    plt.axis('off')
    overlay = pr_mask_squeeze.copy()
    overlay[overlay < threshold] = 0
    alpha = 0.4
    plt.imshow(overlay, cmap='jet', alpha=alpha * (overlay > 0))

    plt.show()


def plot_predictions(
    model: LightningModule,
    data_loader: DataLoader,
    *,
    figsize: tuple[int, int] = (16, 5),
) -> None:
    n_cols = 5
    for batch in data_loader:
        with torch.no_grad():
            model.eval()
            logits = model(batch['image'])
        pr_masks = logits.sigmoid()

        for image, gt_mask, pr_mask, img_file in zip(
            batch['image'], batch['mask'], pr_masks, batch['filename']
        ):
            plt.figure(figsize=figsize)

            original_image = image.numpy().transpose(1, 2, 0)  # convert CHW -> HWC

            plt.subplot(1, n_cols, 1)
            plt.imshow(original_image)
            plt.title(f'Image {img_file}')
            plt.axis('off')

            plt.subplot(1, n_cols, 2)
            # Squeeze classes dim, because we have only one class
            plt.imshow(gt_mask.numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
            plt.title('Ground truth')
            plt.axis('off')

            plt.subplot(1, n_cols, 3)
            # Squeeze classes dim, because we have only one class
            pr_mask_squeeze = pr_mask.numpy().squeeze()
            plt.imshow(pr_mask_squeeze, cmap='gray', vmin=0, vmax=1)
            plt.title('Probabilities')
            plt.axis('off')

            plt.subplot(1, n_cols, 4)
            plt.imshow(
                np.where(pr_mask_squeeze > 0.5, 1, 0), cmap='gray', vmin=0, vmax=1
            )
            plt.title('Binary Prediction')
            plt.axis('off')

            plt.subplot(1, n_cols, 5)
            plt.imshow(original_image)
            plt.title('Overlay')
            plt.axis('off')
            alpha = 0.4
            plt.imshow(pr_mask_squeeze, cmap='jet', alpha=alpha)
            # plt.colorbar() # TODO

            plt.show()


def plot_predictions_compact(
    model: LightningModule,
    data_loader: DataLoader,
    n_cols: int = 5,
    threshold: float = 0.5,
    save_path: str | Path | None = None,
    cmap: str = 'winter',
    limit: int | None = None,
) -> None:
    """Plot predictions from a model on a dataset.

    Parameters
    ----------
    model : LightningModule
        PyTorch Lightning model.
    data_loader : DataLoader
        PyTorch DataLoader.
    n_cols : int, default=5
        Number of columns in the plot, by default 5.
    threshold : float, default=0.5
        Threshold to apply to the predicted masks.
        Everything below the threshold is set to 0.
    save_path : str or Path or None, optional
        Path to save the plot.
    cmap : str, default='winter'
        Colormap to use for the overlay.
    """
    batch_size = data_loader.batch_size if data_loader.batch_size else 1
    n_data = len(data_loader) * batch_size if limit is None else limit

    # Calculate the number of rows needed
    n_rows = (n_data + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()  # type: ignore

    idx = 0

    for batch in data_loader:
        with torch.no_grad():
            model.eval()
            logits = model(batch['image'])
        pr_masks = logits.sigmoid()

        for image, gt_mask, pr_mask, img_file in zip(
            batch['image'], batch['mask'], pr_masks, batch['filename']
        ):
            if idx >= n_data:  # Ensure we don't go beyond the total number of images
                break

            ax = axes[idx]
            original_image = image.numpy().transpose(1, 2, 0)  # convert CHW -> HWC
            overlay = pr_mask.numpy().squeeze()
            overlay[overlay < threshold] = 0

            ax.imshow(original_image)
            ax.imshow(overlay, cmap=cmap, alpha=0.4 * (overlay > 0))
            ax.set_title(img_file)
            ax.axis('off')
            idx += 1
            if limit and idx >= limit:
                break

        if limit and idx >= limit:
            break

    # Hide any remaining empty subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_autoencoder_reconstruction(
    model: LightningModule,
    data_loader: DataLoader,
    path_to_jpg: str | Path,
    save_path: str | Path | None = None,
    limit: int | None = None,
) -> None:
    """Plot predictions from a model on a dataset.

    Parameters
    ----------
    model : LightningModule
        PyTorch Lightning model.
    data_loader : DataLoader
        PyTorch DataLoader.
    path_to_jpg : str | Path
        Path to the directory containing the original images in JPG format.
    save_path : str | Path | None, default=None
        Path to save the plot.
    limit : int | None, default=None
        Limit the number of images to plot.
    """
    batch_size = data_loader.batch_size if data_loader.batch_size else 1

    # Calculate the number of rows needed
    n_cols = 3
    n_rows = len(data_loader) * batch_size if limit is None else limit

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()  # type: ignore

    idx = 0

    if not path_to_jpg or not (path_to_jpg := Path(path_to_jpg)).exists():
        raise FileNotFoundError(f'Path `{path_to_jpg}` does not exist.')

    for batch in data_loader:
        with torch.no_grad():
            model.eval()
            results = model(batch['image'])

        for image, result, img_file in zip(batch['image'], results, batch['filename']):
            # if (
            #     idx >= n_cols * n_data
            # ):  # Ensure we don't go beyond the total number of images
            #     break

            original_image = Image.open(
                jpg_path := (path_to_jpg / img_file).with_suffix('.jpg')
            )
            original_image = crop_driver_image(original_image, jpg_path).resize(
                (image.shape[2], image.shape[1])
            )
            input_image = image.numpy().transpose(1, 2, 0)  # convert CHW -> HWC
            reconst_image = result.numpy().transpose(1, 2, 0)  # convert CHW -> HWC

            axes[idx + 1].set_title(img_file)

            axes[idx].imshow(original_image)
            axes[idx + 1].imshow(input_image, cmap='gray')
            axes[idx + 2].imshow(reconst_image, cmap='gray')

            idx += n_cols
            if limit and idx >= n_cols * limit:
                break

        if limit and idx >= n_cols * limit:
            break

    # Hide any remaining empty subplots
    for i in range(0, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_learning_curves(
    csv_log_path: str | Path,
    save_path: str | Path | None = None,
    metrics: dict[str, str] = {'jaccard': 'Jaccard Index', 'f1s': 'F1 Score'},
) -> None:
    """Plot learning curves from a CSV log file generated by PyTorch Lightning.

    Parameters
    ----------
    csv_log_path : str or Path
        Path to the CSV log file.
    """

    path = Path(csv_log_path)
    if not path.exists():
        raise FileNotFoundError(f'File {path} not found')

    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if '_epoch' in col] + ['step'])
    df = df.rename(columns={col: col.replace('_step', '') for col in df.columns})
    df.head()

    # Group by epoch ignoring NaN values
    df_epoch = df.groupby('epoch').mean().reset_index()

    # Plot learning curves
    fig, axes = plt.subplots(1, 1 + len(metrics), figsize=(15, 5))
    ax = axes.flatten()  # type: ignore

    # Plot loss
    df_epoch.plot(
        x='epoch', y=['train_loss', 'valid_loss'], ax=ax[0], title='Loss', linewidth=2
    )
    ax[0].legend(['Training loss', 'Validation loss'], frameon=True)
    ax[0].set_ylabel('Loss')

    # Plot additional metrics
    for i, metric in enumerate(metrics):
        axes_index = 1 + i
        metric_label = metrics[metric]
        df_epoch.plot(
            x='epoch',
            y=[f'train_{metric}', f'valid_{metric}'],
            ax=ax[axes_index],
            title=metric_label,
            linewidth=2,
        )
        ax[axes_index].legend(
            [f'Training {metric_label}', f'Validation {metric_label}'], frameon=True
        )
        ax[axes_index].set_ylabel(metrics[metric])

    for a in ax:
        a.set_xlabel('Epoch')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
