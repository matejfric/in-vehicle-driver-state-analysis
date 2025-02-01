from itertools import chain


def get_y_proba_from_errors(errors: list[float]) -> list[float]:
    """Normalize errors to [0, 1] range and get probabilistic `y_pred`.
    Values close to 1 mean a higher error and may therefore represent an anomaly."""
    min_err = min(errors)
    max_err = max(errors)
    errors_norm = [(x - min_err) / (max_err - min_err) for x in errors]
    y_proba = list(chain.from_iterable([[x] for x in errors_norm]))
    return y_proba
