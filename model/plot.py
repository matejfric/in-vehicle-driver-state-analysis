import random
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import albumentations as albu
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningModule
from sklearn.metrics import auc, precision_recall_curve, roc_curve
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


def plot_predictions_compact(
    model: LightningModule,
    data_loader: DataLoader,
    n_cols: int = 5,
    threshold: float = 0.5,
    save_path: str | Path | None = None,
    cmap: str = 'winter',
    limit: int = 10,
    seed: int | None = None,
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
    rng = random.Random(seed)
    indices = range(len(data_loader.dataset))  # type: ignore
    random_indices = rng.sample(indices, limit)

    # Calculate the number of rows needed
    n_rows = (limit + n_cols - 1) // n_cols  # Ceiling division

    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()  # type: ignore

    idx = 0
    for item in [data_loader.dataset[idx] for idx in random_indices]:
        with torch.no_grad():
            model.eval()
            logits = model(item['image'])
        pr_mask = logits.sigmoid()
        image = item['image']
        ax = axes[idx]
        original_image = image.numpy().transpose(1, 2, 0)  # convert CHW -> HWC
        overlay = pr_mask.numpy().squeeze()
        overlay[overlay < threshold] = 0

        ax.imshow(original_image)
        ax.imshow(overlay, cmap=cmap, alpha=0.4 * (overlay > 0))
        ax.axis('off')
        idx += 1

    # Hide any remaining empty subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_temporal_autoencoder_reconstruction(
    model: LightningModule,
    data_loader: DataLoader,
    save_path: str | Path | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    random_shuffle: bool = False,
    time_dim_index: Literal[0, 1] = 0,
    show_heatmap: bool = True,
    show_metrics: bool = True,
    legend_font_size: int = 19,
    model_type: Literal['tae', 'stae'] = 'tae',
) -> None:
    """Plot predictions from a model on a dataset.

    Parameters
    ----------
    model : LightningModule
        PyTorch Lightning model.
    data_loader : DataLoader
        PyTorch DataLoader for STAEDataset, TemporalAutoencoderDataset, or TemporalAutoencoderDatasetDMD
    save_path : str | Path | None, default=None
        Path to save the plot.
    limit : int | None, default=None
        Limit the number of sequences to plot.
    indices : list[int] | None, default=None
        List of indices to plot.
    random_shuffle : bool, default=False
        Shuffle the batches before plotting.
    time_dim_index : int, default=0
        Index of the time dimension in the input tensor.
    show_heatmap : bool, default=True
        Show the difference heatmap.
    show_metrics : bool, default=True
        Show the MSE and Frobenius norm metrics.
    """
    if not limit and not indices:
        raise ValueError('Either `limit` or `indices` must be provided.')
    if limit and indices:
        raise ValueError('Only one of `limit` or `indices` can be provided.')
    if indices and random_shuffle:
        raise ValueError('`random_shuffle` is only for `limit`.')

    model.eval()
    device = model.device
    dataset = data_loader.dataset
    n_cols = 3 if show_heatmap else 2
    n_rows = limit or len(indices)  # type: ignore (either limit or indices is provided)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), layout='compressed'
    )

    if limit:
        indices = (
            random.sample(range(len(data_loader.dataset)), limit)  # type: ignore
            if random_shuffle
            else list(range(limit))
        )

    for row_idx, idx in enumerate(indices):  # type: ignore
        sequence = dataset[idx]['image']
        with torch.no_grad():
            reconstruction = model(sequence.unsqueeze(0).to(device))[0]  # type: ignore
        if model_type == 'stae':
            # (C, T, H, W)
            reconstruction = reconstruction[0]
        reconstruction = reconstruction.cpu()

        t = 0  # temporal index (plot the first frame only)
        if time_dim_index == 0:
            input_img = sequence[t].numpy().squeeze()
            reconst_img = reconstruction[t].numpy().squeeze()
        else:
            input_img = sequence[:, t].numpy().squeeze()  # type: ignore
            reconst_img = reconstruction[:, t].numpy().squeeze()

        if len(input_img.shape) == 3:
            # Convert CHW -> HWC
            input_img = input_img.transpose(1, 2, 0)
            reconst_img = reconst_img.transpose(1, 2, 0)

        mse = np.mean((input_img - reconst_img) ** 2)
        fro_norm = np.sqrt(np.sum((input_img - reconst_img) ** 2))
        diff = np.abs(input_img - reconst_img)

        if len(diff.shape) == 3:
            if diff.shape[-1] > 3:
                # RGBD or RGBDM images
                # NOTE: We might want to plot RGB and depth images separately
                input_img = input_img[..., :3]
                reconst_img = reconst_img[..., :3]
            diff = diff.mean(axis=-1)

        # Plot original and reconstructed images
        axes[row_idx, 0].imshow(input_img, cmap='gray')  # type: ignore
        axes[row_idx, 1].imshow(reconst_img, cmap='gray')  # type: ignore
        if show_metrics:
            axes[row_idx, 1].text(  # type: ignore
                # (0, 0) is lower-left and (1, 1) is upper-right
                0.972,
                0.972,
                f'MSE={mse:.3f}\nFRO={fro_norm:.3f}',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'pad': 0.5},
                ha='right',
                va='top',
                transform=axes[row_idx, 1].transAxes,  # type: ignore
                fontsize=legend_font_size,
            )
        if show_heatmap:
            heatmap = axes[row_idx, 2].imshow(diff, cmap='jet')  # type: ignore

        for i in range(n_cols):
            axes[row_idx, i].axis('off')  # type: ignore

    # Add column titles
    col_titles = ['Original', 'Reconstructed']
    if show_heatmap:
        col_titles.append('Difference')
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, pad=20)  # type: ignore

    # Add a colorbar for the heatmap
    if show_heatmap:
        plt.colorbar(heatmap, ax=axes, label='Absolute Difference', shrink=1.6 / n_rows)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_stae_reconstruction(
    model: LightningModule,
    data_loader: DataLoader,
    save_path: str | Path | None = None,
    limit: int | None = None,
    indices: list[int] | None = None,
    random_shuffle: bool = False,
    show_heatmap: bool = False,
) -> None:
    """Plot predictions from a model on a dataset.

    Parameters
    ----------
    model : LightningModule
        PyTorch Lightning model.
    data_loader : DataLoader
        PyTorch DataLoader.
    save_path : str | Path | None, default=None
        Path to save the plot.
    limit : int | None, default=None
        Limit the number of sequences to plot.
    indices : list[int] | None, default=None
        List of indices to plot.
    random_shuffle : bool, default=False
        Shuffle the batches before plotting.
    """
    from .dataset import STAEDataset

    if not isinstance(data_loader.dataset, STAEDataset):
        raise ValueError(
            f'DataLoader must be using `STAEDataset`. Actual: `{type(data_loader.dataset)}`.'
        )
    if not limit and not indices:
        raise ValueError('Either `limit` or `indices` must be provided.')
    if limit and indices:
        raise ValueError('Only one of `limit` or `indices` can be provided.')
    if indices and random_shuffle:
        raise ValueError('`random_shuffle` is only for `limit`.')

    model.eval()
    device = model.device
    dataset = data_loader.dataset
    n_cols = 3 if show_heatmap else 2
    n_rows = limit or len(indices)  # type: ignore (either limit or indices is provided)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    if limit:
        indices = (
            random.sample(range(len(data_loader.dataset)), limit)
            if random_shuffle
            else list(range(limit))
        )

    for row_idx, idx in enumerate(indices):  # type: ignore
        sequence = dataset[idx]['image']
        with torch.no_grad():
            reconstruction = model(sequence.unsqueeze(0).to(device))[0]  # type: ignore
        reconstruction = reconstruction.squeeze().cpu().detach()

        # Take the first frame of the sequence
        input_img = sequence[:, 0].numpy().squeeze()  # type: ignore
        reconst_img = reconstruction[0].numpy().squeeze()

        mse = np.mean((input_img - reconst_img) ** 2)
        fro_norm = np.linalg.norm(input_img - reconst_img, ord='fro')
        diff = np.abs(input_img - reconst_img)

        # Plot original and reconstructed images
        axes[row_idx, 0].imshow(input_img, cmap='gray')  # type: ignore
        axes[row_idx, 0].axis('off')  # type: ignore
        axes[row_idx, 1].imshow(reconst_img, cmap='gray')  # type: ignore
        axes[row_idx, 1].axis('off')  # type: ignore
        axes[row_idx, 1].text(  # type: ignore
            # (0, 0) is lower-left and (1, 1) is upper-right
            0.98,
            0.98,
            f'MSE={mse:.3f}\nFRO={fro_norm:.3f}',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'pad': 0.5},
            ha='right',
            va='top',
            transform=axes[row_idx, 1].transAxes,  # type: ignore
            fontsize=9,
        )
        if show_heatmap:
            heatmap = axes[row_idx, 2].imshow(diff, cmap='jet')  # type: ignore
            axes[row_idx, 2].axis('off')  # type: ignore

    # Add column titles
    col_titles = ['Original', 'Reconstructed']
    if show_heatmap:
        col_titles.append('Difference')
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, pad=20)  # type: ignore

    # Add a colorbar for the heatmap
    if show_heatmap:
        height_factor = 1 / n_rows  # Dynamically scale the colorbar height
        cbar_ax = fig.add_axes([0.92, (1 - height_factor) / 2, 0.02, height_factor])  # type: ignore
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        plt.colorbar(heatmap, cax=cbar_ax, label='Absolute Difference')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_autoencoder_reconstruction(
    model: LightningModule,
    data_loader: DataLoader,
    dataset_path: str | Path,
    save_path: str | Path | None = None,
    limit: int | None = None,
    random_shuffle: bool = False,
) -> None:
    """Plot predictions from a model on a dataset.

    Parameters
    ----------
    model : LightningModule
        PyTorch Lightning model.
    data_loader : DataLoader
        PyTorch DataLoader.
    dataset_path : str | Path
        Path to the root directory containing the original images in JPG format.
    save_path : str | Path | None, default=None
        Path to save the plot.
    limit : int | None, default=None
        Limit the number of images to plot.
    random_shuffle : bool, default=False
        Shuffle the batches before plotting.
    """
    batch_size = data_loader.batch_size if data_loader.batch_size else 1

    # Calculate the number of rows needed
    n_cols = 3
    n_rows = len(data_loader) * batch_size if limit is None else limit

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()  # type: ignore

    idx = 0

    if not dataset_path or not (dataset_path := Path(dataset_path)).exists():
        raise FileNotFoundError(f'Path `{dataset_path}` does not exist.')

    if random_shuffle:
        batches = list(data_loader)
        random.shuffle(batches)
    else:
        batches = data_loader

    for batch in batches:
        with torch.no_grad():
            model.eval()
            results = model(batch['image'])

        for image, result, img_file in zip(batch['image'], results, batch['filename']):
            for stage in ['train', 'validation', 'test']:
                # Try to load the original image in JPG format
                try:
                    original_image = Image.open(
                        jpg_path := (
                            dataset_path / stage / 'images' / img_file
                        ).with_suffix('.jpg')
                    )
                except FileNotFoundError:
                    continue
                else:
                    break  # Found the original image

            original_image = crop_driver_image(original_image, jpg_path).resize(
                (image.shape[2], image.shape[1])
            )
            input_image = image.numpy().transpose(1, 2, 0)  # convert CHW -> HWC
            reconst_image = result.numpy().transpose(1, 2, 0)  # convert CHW -> HWC

            frobenius_norm = np.linalg.norm(
                (input_image - reconst_image).squeeze(), ord='fro'
            )
            mse = np.mean((input_image - reconst_image).squeeze() ** 2)

            axes[idx + 1].set_title(img_file)

            axes[idx].imshow(original_image)
            axes[idx + 1].imshow(input_image, cmap='gray')
            axes[idx + 2].imshow(reconst_image, cmap='gray')
            axes[idx + 2].text(
                # (0, 0) is lower-left and (1, 1) is upper-right
                1.03,
                0.97,
                f'MSE={mse:.3f}\nFRO={frobenius_norm:.3f}',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'pad': 1},
                horizontalalignment='center',
                verticalalignment='center',
                transform=axes[idx + 2].transAxes,
            )

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
    figsize: tuple[int | float, int | float] = (15, 5),
    loss_name: str = 'loss',
    linewidth: float = 2.0,
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
    fig, axes = plt.subplots(1, 1 + len(metrics), figsize=figsize)
    ax = axes.flatten()  # type: ignore

    # Plot loss
    df_epoch.plot(
        x='epoch',
        y=[f'train_{loss_name}', f'valid_{loss_name}'],
        ax=ax[0],
        title='Loss',
        linewidth=linewidth,
        style=['--', '-'],  # Dashed for train, solid for valid
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
            linewidth=linewidth,
            style=['--', '-'],  # Dashed for train, solid for valid
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


def plot_error_and_anomalies(
    y_true: Sequence[int],
    y_pred: Sequence[float],
    y_max: float = 1.0,
    threshold: float | None = None,
    figsize: tuple[int, int] = (20, 6),
    save_path: str | Path | None = None,
) -> None:
    x = np.arange(len(y_pred))

    plt.figure(figsize=figsize)
    plt.plot(x, y_pred, color='brown', alpha=0.9, linewidth=1.5, label='Error')

    # Highlight anomaly regions
    start_indices = np.where(np.diff(y_true, prepend=0) == 1)[0]
    end_indices = np.where(np.diff(y_true, append=0) == -1)[0]
    for start, end in zip(start_indices, end_indices, strict=True):
        plt.axvspan(
            start,
            end,
            color='red',
            alpha=0.1,
            label='Anomaly' if start == start_indices[0] else '',
        )

    plt.ylim(0.0 - 0.05, y_max + 0.05)

    if threshold:
        plt.axhline(
            y=threshold, color='black', linestyle='--', linewidth=2.0, label='Threshold'
        )

    plt.ylabel('Error')
    plt.xlabel('Frame')

    plt.legend(
        fancybox=True,  # rounded corners
        facecolor='white',
        framealpha=1,  # no transparency
        frameon=1,
        borderpad=0.5,
        edgecolor='black',
        fontsize=plt.rcParams['font.size'] - 4,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_pr_chart(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = 'rainbow',
    cbar_text: str = 'Thresholds',
    title: str = 'Precision-Recall Chart',
) -> tuple[float, float]:
    """
    Plot Precision-Recall curve with optimal threshold.

    Parameters
    ----------
    y_true : np.ndarray or list[int]
        True binary labels.
    y_pred_proba : np.ndarray or list[float]
        Predicted probabilities.
    save_path : str or Path, optional
        If provided, the plot is saved to this path.
    figsize : tuple, default=(8, 6)
        Figure size.
    cmap : str, default='rainbow'
        Colormap for the thresholds.
    cbar_text : str, default='Thresholds'
        Label for the colorbar.

    Returns
    -------
    pr_auc : float
        Area under the Precision-Recall curve.
    optimal_threshold : float
        Optimal threshold based on the maximum F1 score.

    Note
    ----
    The optimal threshold is selected by maximizing the F1 score computed from the
    precision and recall values. Additionally, the no-skill baseline (i.e. the ratio
    of positive samples) is plotted for reference.
    """
    plt.figure(figsize=figsize)

    # Compute precision, recall and thresholds.
    # Note: precision and recall have one more element than thresholds.
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    # Compute F1 scores for each threshold (skip the first element).
    f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:])
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Plot the precision-recall curve.
    plt.step(recall, precision, c='black', label=f'AUC={pr_auc:.3f}')
    scatter = plt.scatter(recall[1:], precision[1:], c=thresholds, cmap=cmap)

    # Plot the no-skill baseline (precision equals the ratio of positive samples).
    no_skill = np.sum(y_true) / len(y_true)
    plt.hlines(no_skill, 0, 1, colors='gray', linestyles='dashed', label='Baseline')

    # Mark the optimal threshold point.
    optimal_recall = recall[optimal_idx + 1]
    optimal_precision = precision[optimal_idx + 1]
    plt.plot(
        [optimal_recall, optimal_recall], [0, optimal_precision], 'k', linestyle='solid'
    )
    plt.text(
        # + 0.01 or - 0.25 (if <0 place on the left, but depends on font size),
        optimal_recall + 0.01,
        optimal_precision / 2 + 0.05,
        f'T={optimal_threshold:.2f}',
        fontsize=plt.rcParams['font.size'] - 4,
    )

    # Add a colorbar to indicate threshold values.
    cbar = plt.colorbar(scatter, shrink=0.7)
    cbar.set_label(cbar_text)

    # Set chart details.
    if title:
        plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('square')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

    return float(pr_auc), optimal_threshold


def plot_roc_chart(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = 'rainbow',
    cbar_text: str = 'Thresholds',
    title: str = 'ROC Chart',
) -> tuple[float, float]:
    """Plot ROC curve with optimal threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities.
    figsize : tuple, default=(8, 6)
        Figure size.
    cmap : str, default='rainbow'
        Colormap for the thresholds. One of `matplotlib.pyplot.colormaps()`.

    Returns
    -------
    roc_auc : float
        Area under the ROC curve.
    optimal_threshold : float
        Optimal threshold based on Youden's J statistic.

    Note
    ----
    Inspired by Peter Barrett Bryan: "You deserve a better ROC curve"
    https://towardsdatascience.com/you-deserve-a-better-roc-curve-970617528ce8
    """

    plt.figure(figsize=figsize)

    fpr, tpr, thresholds = roc_curve(y_true[: len(y_pred_proba)], y_pred_proba)
    roc_auc = auc(fpr, tpr)
    j_scores = tpr - fpr
    optimal_idx = j_scores.argmax()
    if optimal_idx == 0:
        warnings.warn(
            'Optimal threshold is 0.0, which may indicate that the model is not performing well.'
        )
        optimal_threshold = 0.0
    else:
        optimal_threshold = thresholds[optimal_idx]

    plt.plot(fpr, tpr, c='black', label=f'AUC={roc_auc:.3f}')
    scatter = plt.scatter(fpr, tpr, c=thresholds, cmap=cmap)

    # Random predictions curve:
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Baseline')

    # Maximum value of Youden's index for the ROC curve (optimal threshold):
    # https://en.wikipedia.org/wiki/Youden%27s_J_statistic
    # https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3
    plt.plot(
        [fpr[optimal_idx], fpr[optimal_idx]],
        [fpr[optimal_idx], tpr[optimal_idx]],
        'k',
        linestyle='solid',
        # label=f'J={optimal_threshold:.2f}',
    )
    plt.text(
        fpr[optimal_idx] + 0.01,
        (fpr[optimal_idx] + tpr[optimal_idx]) / 2,
        f'J={optimal_threshold:.2f}',
        fontsize=plt.rcParams['font.size'] - 4,
    )
    cbar = plt.colorbar(scatter, shrink=0.7)
    cbar.set_label(cbar_text)

    if title:
        plt.title(title)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('square')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return roc_auc, optimal_threshold  # type: ignore


def plot_roc_charts(
    y_true_list: list[np.ndarray | list[int]],
    y_pred_proba_list: list[np.ndarray | list[float]],
    titles: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] | None = None,
    cmap: str = 'rainbow',
    cbar_text: str = 'Thresholds',
) -> list[tuple[float, float]]:
    """Plot multiple ROC curves with optimal thresholds side by side.

    Parameters
    ----------
    y_true_list : list of arrays
        List of true binary labels arrays.
    y_pred_proba_list : list of arrays
        List of predicted probabilities arrays.
    titles : list of str, optional
        List of titles for each subplot. If None, will use default titles.
    figsize : tuple, optional
        Figure size. If None, will be calculated based on number of plots.
    cmap : str, default='rainbow'
        Colormap for the thresholds. One of `matplotlib.pyplot.colormaps()`.
    cbar_text : str, default='Thresholds'
        Label for the colorbar.

    Returns
    -------
    metrics : list of tuples
        List of (roc_auc, optimal_threshold) tuples for each plot.
    """
    n_plots = len(y_true_list)
    if not n_plots == len(y_pred_proba_list):
        raise ValueError('Length of y_true_list and y_pred_proba_list must match')

    if titles is None:
        titles = [f'ROC Chart {i + 1}' for i in range(n_plots)]
    elif len(titles) != n_plots:
        raise ValueError('Length of titles must match number of plots')

    # Calculate default figsize if not provided
    if figsize is None:
        figsize = (7 * n_plots, 7)

    # Create figure with gridspec to accommodate colorbar
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, n_plots + 1, width_ratios=[1] * n_plots + [0.05])  # type: ignore
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_plots)]

    # Initialize list to store metrics
    metrics = []

    # Use fixed colormap range from 0 to 1
    norm = plt.Normalize(vmin=0, vmax=1)  # type: ignore

    # Create plots
    for idx, (ax, y_true, y_pred_proba, title) in enumerate(
        zip(axes, y_true_list, y_pred_proba_list, titles)
    ):
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true[: len(y_pred_proba)], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        optimal_idx = j_scores.argmax()
        optimal_threshold = thresholds[optimal_idx]

        # Store metrics
        metrics.append((roc_auc, optimal_threshold))

        # Plot ROC curve
        ax.plot(fpr, tpr, c='black', label=f'AUC={roc_auc:.3f}')
        scatter = ax.scatter(fpr, tpr, c=thresholds, cmap=cmap, norm=norm)

        # Random predictions curve
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # Optimal threshold line and text
        ax.plot(
            [fpr[optimal_idx], fpr[optimal_idx]],
            [fpr[optimal_idx], tpr[optimal_idx]],
            'k',
            linestyle='solid',
        )
        ax.text(
            fpr[optimal_idx] + 0.01,
            (fpr[optimal_idx] + tpr[optimal_idx]) / 2,
            f'J={optimal_threshold:.2f}',
            # fontsize=12,
        )

        # Set title and limits
        ax.set_title(title)
        ax.set_xlim([0, 1])  # type: ignore
        ax.set_ylim([0, 1])  # type: ignore
        ax.axis('square')

        # Handle axis labels and ticks
        if idx == 0:
            ax.set_ylabel('True positive rate')
        else:
            # Remove y-axis labels for all but the first plot
            ax.set_yticklabels([])

        # Add x-label to all plots
        ax.set_xlabel('False positive rate')
        ax.legend(loc='lower right')

    # Add colorbar in the last column of gridspec
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(cbar_text)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return metrics


def _plot_roc_curve(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    ax: plt.Axes,  # type: ignore
    source_type: str,
    cmap: str,
    norm: plt.Normalize,  # type: ignore
    justification: int,
    plot_thresholds: bool,
    plot_kwargs: dict[str, Any],
) -> mpl.collections.PathCollection | None:  # type: ignore
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true[: len(y_pred_proba)], y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        label=f'{source_type.ljust(justification)} AUC={roc_auc:.3f}',
        **plot_kwargs,
    )

    # Random predictions curve
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    if plot_thresholds:
        scatter = ax.scatter(fpr, tpr, c=thresholds, cmap=cmap, norm=norm)
        return scatter


def _plot_pr_curve(
    y_true: np.ndarray | list[int],
    y_pred_proba: np.ndarray | list[float],
    ax: plt.Axes,  # type: ignore
    source_type: str,
    cmap: str,
    norm: plt.Normalize,  # type: ignore
    justification: int,
    plot_thresholds: bool,
    plot_kwargs: dict[str, Any],
) -> mpl.collections.PathCollection | None:  # type: ignore
    precision, recall, thresholds = precision_recall_curve(
        y_true[: len(y_pred_proba)], y_pred_proba
    )
    precision[precision > 1.0] = 1.0
    recall[recall > 1.0] = 1.0
    pr_auc = auc(recall, precision)

    # Compute F1 scores for each threshold (skip the first element).
    with warnings.catch_warnings():
        # Suppress warnings for division by zero
        warnings.simplefilter('ignore')
        f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:])
    optimal_idx = np.array(f1_scores).argmax()
    optimal_threshold = thresholds[optimal_idx]  # noqa

    # Plot PR curve
    ax.step(
        recall,
        precision,
        label=f'{source_type.ljust(justification)} AUC={pr_auc:.3f}',
        **plot_kwargs,
    )

    if plot_thresholds:
        scatter = plt.scatter(
            recall[1:], precision[1:], c=thresholds, cmap=cmap, norm=norm
        )
        return scatter


def plot_results(
    which: Literal['roc', 'pr'],
    data: dict[str, dict[str, dict[str, list[float | int] | float]]],
    cmap: str = 'rainbow',
    figsize: tuple[int, int] | None = None,
    plot_thresholds: bool = False,
    cbar_text: str = 'Threshold',
    save_path: str | Path | None = None,
    justification: int = 5,
    linewidth: int = 3,
    fig_height_multiplier: int = 9,
    fig_width_multiplier: int = 7,
    n_rows: int = 1,
    n_cols: int | None = None,
    legend_outside: bool = False,
    source_type_color_map: dict[str, str] | None = None,
    source_type_linestyle_map: dict[str, str] | None = None,
    driver_name_mapping: dict[str, Any] | None = None,
) -> None:
    n_plots = len(data)

    # Calculate number of columns if not provided
    if n_cols is None:
        n_cols = int(np.ceil(n_plots / n_rows))
    else:
        # Adjust n_rows if both n_rows and n_cols are provided but insufficient
        n_rows = int(np.ceil(n_plots / n_cols))

    # Calculate default figsize if not provided
    if figsize is None:
        figsize = (fig_width_multiplier * n_cols, fig_height_multiplier * n_rows)

    # Create figure with gridspec to accommodate colorbar
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [0.05])  # type: ignore

    # Use fixed colormap range from 0 to 1
    norm = plt.Normalize(vmin=0, vmax=1)  # type: ignore
    scatter = None

    # Create axes for each plot
    axes = []
    for i in range(min(n_plots, n_rows * n_cols)):
        row = i // n_cols
        col = i % n_cols
        axes.append(fig.add_subplot(gs[row, col]))

    for idx, (ax, (driver_name, driver_results)) in enumerate(zip(axes, data.items())):
        margin = 0.05
        ax.set_xlim([0 - margin, 1 + margin])  # type: ignore
        ax.set_ylim([0 - margin, 1 + margin])  # type: ignore
        ax.axis('square')
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        for source_type, results in driver_results.items():
            y_true: list[int] = results['y_true']  # type: ignore
            y_pred_proba: list[float] = results['y_proba']  # type: ignore

            plot_kwargs = {}
            plot_kwargs['linewidth'] = linewidth
            if source_type_color_map:
                plot_kwargs['c'] = source_type_color_map[source_type]
            if source_type_linestyle_map:
                plot_kwargs['linestyle'] = source_type_linestyle_map[source_type]

            if which == 'roc':
                scatter = _plot_roc_curve(
                    y_true,
                    y_pred_proba,
                    ax,
                    source_type,
                    cmap,
                    norm,
                    justification,
                    plot_thresholds,
                    plot_kwargs,
                )
            elif which == 'pr':
                scatter = _plot_pr_curve(
                    y_true,
                    y_pred_proba,
                    ax,
                    source_type,
                    cmap,
                    norm,
                    justification,
                    plot_thresholds,
                    plot_kwargs,
                )

        # Set title and limits
        driver_name = (
            driver_name_mapping[driver_name] if driver_name_mapping else driver_name
        )
        if driver_name == 'All':
            title = 'All Drivers'
        else:
            title = f'Driver {driver_name}'
        ax.set_title(title)

        # Handle axis labels and ticks
        row = idx // n_cols
        col = idx % n_cols

        # Only add y-labels to leftmost plots
        if col == 0:
            ax.set_ylabel('True positive rate' if which == 'roc' else 'Precision')
        else:
            ax.set_yticklabels([])

        # Only add x-labels to bottom plots
        if row == n_rows - 1 or idx == n_plots - 1:
            ax.set_xlabel('False positive rate' if which == 'roc' else 'Recall')
        else:
            ax.set_xticklabels([])

        if not legend_outside:
            ax.legend(
                loc='lower right' if which == 'roc' else 'lower left',
                fancybox=True,  # rounded corners
                facecolor='white',
                framealpha=0.75,  # 3/4 opaque
                frameon=1,
                borderpad=0.5,
                edgecolor='black',
            )
        else:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
            )

    # Add colorbar in the last column of gridspec
    if plot_thresholds and scatter is not None:
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.set_label(cbar_text)

    plt.tight_layout()
    if n_rows > 1:
        plt.subplots_adjust(bottom=0.3)  # Pad bottom for the legend
        plt.subplots_adjust(hspace=0.1)  # Pad vertically between subplots

    if save_path:
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pdf')
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
