from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def get_y_proba_from_errors(
    errors: list[float], iqr: tuple[float, float] = (0.0, 1.0)
) -> list[float]:
    """Normalize errors to [0, 1] range and get probabilistic `y_pred`.
    Values close to 1 mean a higher error and may therefore represent an anomaly.

    Outliers are clipped to the interquartile range (IQR), by default min and max values (i.e., no clipping).
    The range is then normalized to [0, 1].

    Parameters
    ----------
    errors : list[float]
        List of errors to be normalized.
    iqr : tuple[float, float], default=(0.0, 1.0)
        Tuple of floats representing the quantiles to clip the errors.
    """
    q_low_err = float(np.quantile(errors, iqr[0]))
    q_up_err = float(np.quantile(errors, iqr[1]))

    # Clip values to be within the IQR
    errors_clipped = [max(min(x, q_up_err), q_low_err) for x in errors]

    # Normalize using the quantile range
    y_proba = [(x - q_low_err) / (q_up_err - q_low_err) for x in errors_clipped]
    return y_proba


def compute_best_roc_auc(
    y_true: Sequence[float | int],
    errors: dict[str, list[float]],
    iqr: tuple[float, float] = (0.0, 1.0),
) -> dict[str, str | float | list[float]]:
    """Evaluate ROC AUC score for each metric, choose the best one."""
    best_metric = 'mse'
    best_roc_auc_score = 0.0
    for key, value in errors.items():
        y_proba = get_y_proba_from_errors(value, iqr=iqr)
        if len(y_proba) != len(y_true):
            print(
                f'Ground truth and predictions have different lengths! Truncating from {len(y_true)} to {len(y_proba)}.'
            )
            y_true = y_true[: len(y_proba)]
        roc_auc = roc_auc_score(y_true, y_proba)
        if roc_auc > best_roc_auc_score:
            best_metric = key
            best_roc_auc_score = roc_auc

    return {
        'best_metric': best_metric,
        'roc_auc': float(best_roc_auc_score),
        'y_proba': get_y_proba_from_errors(errors[best_metric], iqr=iqr),
    }
