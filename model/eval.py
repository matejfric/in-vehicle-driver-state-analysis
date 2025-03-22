import numpy as np


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
