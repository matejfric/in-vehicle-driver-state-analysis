from typing import Any

import torch


# Newtype Design Pattern
class Encoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


# Newtype Design Pattern
class Decoder(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(args, kwargs)


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
        output = torch.tensor([])
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
    """Get model summary.

    Parameters
    ----------
    tablefmt: str, default='simple'
        The table format, e.g. 'plain', 'simple', 'grid', 'pipe', 'orgtbl', 'jira', 'presto', 'pretty', 'html', 'latex', 'mediawiki', 'moinmoin', 'rst', 'tsv', 'textile'.

    Examples
    --------
    >>> print(summarize_model(encoder))
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

    table_data = []
    for module in modules if isinstance(modules, list) else [modules]:
        for submodule in module.children():
            # Iterate over the submodules in the Sequential container
            children = list(submodule.children())
            if len(children) == 0:
                if (n_params := get_module_params(submodule)) > 0:
                    table_data.append(
                        [
                            submodule.__class__.__name__,
                            get_module_hyperparameters(submodule),
                            n_params,
                        ]
                    )
                else:
                    table_data.append(
                        [
                            submodule.__class__.__name__,
                            get_module_hyperparameters(submodule),
                            n_params,
                        ]
                    )
            for subsubmodule in children:
                subname = subsubmodule.__class__.__name__
                for subsubsubmodule in subsubmodule.children():
                    if isinstance(
                        subsubsubmodule, torch.nn.modules.container.ModuleList
                    ):
                        # If this is a ModuleList, iterate over the submodules
                        for subsubsubsubmodule in subsubsubmodule.children():
                            if (n_params := get_module_params(subsubsubsubmodule)) > 0:
                                table_data.append(
                                    [
                                        f'{subname} {len(subsubsubmodule)}x {subsubsubsubmodule.__class__.__name__}',
                                        get_module_hyperparameters(subsubsubsubmodule),
                                        n_params,
                                    ]
                                )
                            else:
                                table_data.append(
                                    [
                                        f'{subname} {len(subsubsubmodule)}x {subsubsubsubmodule.__class__.__name__}',
                                        get_module_hyperparameters(subsubsubsubmodule),
                                        n_params,
                                    ]
                                )
                    else:
                        if (n_params := get_module_params(subsubsubmodule)) > 0:
                            table_data.append(
                                [
                                    subname,
                                    get_module_hyperparameters(subsubsubmodule),
                                    n_params,
                                ]
                            )
                        else:
                            table_data.append(
                                [
                                    subname,
                                    get_module_hyperparameters(subsubsubmodule),
                                    n_params,
                                ]
                            )

    total_params = sum([x[2] for x in table_data])
    if use_separating_line:
        table_data.append(SEPARATING_LINE)
    table_data.append(['', '', total_params])

    table = tabulate(table_data, headers=header, intfmt=intfmt, tablefmt=tablefmt)

    return table
