from collections import defaultdict
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm


# Newtype Design Pattern
class Encoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# Newtype Design Pattern
class Decoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


def evaluate_model_parallel(
    model: nn.Module,
    test_dataloader: DataLoader,
    batch_size: int = 16,
    num_workers: int = 4,
    model_type: Literal['tae', 'stae'] = 'tae',
) -> dict[str, list[float]]:
    """Evaluate a model in parallel."""
    if torch.cuda.is_available():
        # Get the number of available GPUs
        n_gpu = torch.cuda.device_count()
        # Use DataParallel if multiple GPUs are available
        if n_gpu > 1:
            model = nn.DataParallel(model)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        n_gpu = 0

    # Print device information
    print(f'Using {n_gpu} GPU(s)' if n_gpu > 0 else 'Using CPU')

    # Set model to evaluation mode and move to device
    model.eval()
    model.to(device)

    # Initialize error container
    errors = defaultdict(list)

    # Use a proper DataLoader with batching
    loader = DataLoader(
        test_dataloader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Process batches
    with torch.no_grad():
        for batch in tqdm(loader):
            input_seq = batch['image']
            input_seq = input_seq.to(device)

            # Forward pass
            output = model(input_seq)
            if model_type == 'stae':
                # For STAE, we only take the reconstruction during
                # inference and discard the prediction.
                output = output[0]

            # Move results to CPU for error calculation
            output = output.cpu()
            input_seq = input_seq.cpu()

            # Calculate errors for each sample in the batch
            for i in range(input_seq.size(0)):
                sample_output = output[i]
                sample_input = input_seq[i]

                # MSE
                mse = (
                    F.mse_loss(sample_output, sample_input, reduction='none')
                    .mean(dim=1)
                    .squeeze()
                    .mean()
                    .item()
                )
                # MAE
                mae = (
                    F.l1_loss(sample_output, sample_input, reduction='none')
                    .mean(dim=1)
                    .squeeze()
                    .mean()
                    .item()
                )
                # Frobenius norm
                fro = (
                    torch.norm((sample_output - sample_input).squeeze(), p='fro', dim=0)
                    .mean()
                    .item()
                )

                # Store errors
                errors['mse'].append(mse)
                errors['mae'].append(mae)
                errors['fro'].append(fro)

    return errors


class TimeDistributed(torch.nn.Module):
    """Inspired by: https://discuss.pytorch.org/t/timedistributed-cnn/51707/2"""

    def __init__(
        self,
        layer: torch.nn.Module,
        n_time_steps: int,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([layer for _ in range(n_time_steps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps = x.size()[1]
        output = torch.tensor([], device=x.device, dtype=x.dtype)
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


def summarize_model(
    modules: torch.nn.Module | list[torch.nn.Module],
    header: list[str] = ['Layer', 'Hyperparameters', 'Parameters'],
    intfmt: str = ',',
    tablefmt: str = 'simple',
    use_separating_line: bool = True,
) -> str:
    """Get model summary with tabulate.

    Parameters
    ----------
    tablefmt: str, default='simple'
        The table format, e.g. 'plain', 'simple', 'grid', 'pipe', 'orgtbl', 'jira', 'presto', 'pretty', 'html', 'latex', 'mediawiki', 'moinmoin', 'rst', 'tsv', 'textile'.
    use_separating_line: bool, default=True
        Whether to use a separating line between the module summary and the total number of parameters. This is not supported in all table formats.

    Returns
    -------
    str
        The model summary as a formatted string.

    Examples
    --------
    >>> print(summarize_model(encoder))
    >>>
    >>> # Concatenate multiple modules
    >>> print(summarize_model([encoder, decoder]))
    >>>
    >>> # Use any `tabulate` formatter (see `tablefmt` parameter in `tabulate` package).
    >>> print(summarize_model(encoder, tablefmt='latex_booktabs', use_separating_line=False))
    """
    from tabulate import SEPARATING_LINE, tabulate

    def get_module_params(module: torch.nn.Module) -> int:
        """Get the number of parameters of the module."""
        return sum(p.numel() for p in module.parameters())

    def get_module_hyperparameters(module: torch.nn.Module) -> str:
        """Get the hyperparameters of the module."""
        str_module = str(module)
        left = str_module.find('(')
        right = str_module.rfind(')')
        parameters = str_module[left + 1 : right]
        return parameters.replace('in_features=', '').replace('out_features=', '')

    def collect_module_summary(module: torch.nn.Module, name_prefix: str = '') -> list:
        """Recursively collect summary information for each module."""
        summary = []

        for sub_module in module.children():
            module_name = module.__class__.__name__
            # Ignore `Sequential` and `TimeDistributed`, directly collect its children.
            if (
                isinstance(sub_module, torch.nn.Sequential)
                or sub_module.__class__.__name__ == 'TimeDistributed'
            ):
                # Recurse
                summary.extend(collect_module_summary(sub_module, name_prefix))
                continue

            if isinstance(sub_module, torch.nn.modules.container.ModuleList):
                # If this is a `ModuleList`, iterate over the submodules.
                # Example output: "TimeDistributed 4x Conv2d"
                for sub_sub_module in sub_module.children():
                    n_params = get_module_params(sub_sub_module)
                    summary.append(
                        [
                            f'{module_name} {len(sub_module)}x {sub_sub_module.__class__.__name__}',
                            get_module_hyperparameters(sub_sub_module),
                            n_params,
                        ]
                    )
            else:
                # Regular submodule

                if list(sub_module.children()):
                    # Recurse if submodule has children
                    summary.extend(
                        collect_module_summary(
                            sub_module,
                            name_prefix=name_prefix,
                        )
                    )
                else:
                    # Example output: "Conv2d"
                    summary.append(
                        [
                            sub_module.__class__.__name__,
                            get_module_hyperparameters(sub_module),
                            get_module_params(sub_module),
                        ]
                    )

        return summary

    # Initialize table data
    table_data = []
    modules = modules if isinstance(modules, list) else [modules]
    for module in modules:
        # Recursively traverse modules and collect summary information.
        table_data.extend(collect_module_summary(module))

    # Calculate total parameters and add to table
    total_params = sum([row[2] for row in table_data])
    if use_separating_line:
        table_data.append(SEPARATING_LINE)
    table_data.append(['', '', total_params])

    # Format the table with tabulate
    table = tabulate(table_data, headers=header, intfmt=intfmt, tablefmt=tablefmt)

    return table
