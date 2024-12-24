import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import mlflow
import pytorch_lightning as L


def get_early_stopping_epoch(experiment_dir: Path) -> int | None:
    checkpoint = list(experiment_dir.glob('*.ckpt'))[0].stem
    pattern = r'epoch=(\d+)'
    match = re.search(pattern, checkpoint)
    if match:
        return int(match.group(1))
    else:
        return None


def log_dict_to_mlflow(
    dictionary: Mapping[str, int | float], type: Literal['metric', 'param']
) -> None:
    for k, v in dictionary.items():
        if type == 'metric':
            mlflow.log_metric(k, v)
        elif type == 'param':
            mlflow.log_param(k, v)


def get_submodule_param_count(model: L.LightningModule) -> dict[str, int]:
    """Get the number of parameters for each submodule in the model."""
    param_counts = {}
    for name, submodule in model.named_children():
        num_params = sum(p.numel() for p in submodule.parameters())
        if num_params > 0:
            param_counts[f'{name}_parameters'] = num_params
    param_counts['total_parameters'] = sum([x for x in param_counts.values()])
    return param_counts
