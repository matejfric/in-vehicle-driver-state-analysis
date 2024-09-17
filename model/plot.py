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
    image_resized = image.resize(pr_mask_squeeze.shape)

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
    axes = axes.flatten()  # Flatten to easily iterate over the axes

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


def plot_learning_curves(
    csv_log_path: str | Path, save_path: str | Path | None = None
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
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    df_epoch.plot(
        x='epoch', y=['train_loss', 'val_loss'], ax=ax[0], title='Loss', linewidth=2
    )
    ax[0].legend(['Training loss', 'Validation loss'], frameon=True)
    ax[0].set_ylabel('Loss')
    df_epoch.plot(
        x='epoch',
        y=['train_jaccard', 'val_jaccard'],
        ax=ax[1],
        title='Jaccard index',
        linewidth=2,
    )
    ax[1].legend(['Training Jaccard index', 'Validation Jaccard index'], frameon=True)
    ax[1].set_ylabel('Jaccard index')
    df_epoch.plot(
        x='epoch', y=['train_f1s', 'val_f1s'], ax=ax[2], title='F1 score', linewidth=2
    )
    ax[2].legend(['Training F1 score', 'Validation F1 score'], frameon=True)
    ax[2].set_ylabel('F1 score')

    for a in ax:
        a.set_xlabel('Epoch')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
