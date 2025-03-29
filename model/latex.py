import re
from pathlib import Path

import pandas as pd

# This file contains functions to process and format LaTeX tables for model evaluation results.


def pivotize_drivers(
    df: pd.DataFrame, source_type_map: dict, driver_name_mapping: dict
) -> pd.DataFrame:
    df = df.copy()
    df['source_type'] = df['source_type'].replace(source_type_map)
    pivot_df = pd.pivot_table(
        df, index='source_type', columns='driver', values='metric.roc_auc'
    )
    pivot_df = pivot_df.rename(
        columns={k: f'Driver {v}' for k, v in driver_name_mapping.items() if k != 'all'}
    )
    if 'all' not in pivot_df.columns:
        pivot_df['all'] = None
    pivot_df['Mean'] = pivot_df.drop(columns=['all'], errors='ignore').mean(axis=1)
    # Move 'all' to the end
    pivot_df = pivot_df[[col for col in pivot_df.columns if col != 'all'] + ['all']]
    pivot_df.columns.name = None
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.rename(columns={'all': 'All', 'source_type': 'Source'})
    return pivot_df


def get_caption(model: str, dataset: str) -> str:
    return f'{model} AU-ROC scores for different {dataset} drivers and source types.'


def pivot_table_to_latex(
    df: pd.DataFrame,
    path: Path,
    label: str,
    caption: str,
) -> None:
    """Convert a DataFrame pivot table to a LaTeX table with highlighted max values."""
    df = df.copy()
    tmp_file = path.with_suffix('.tmp')
    df.to_latex(
        tmp_file,
        index=False,
        caption=(caption, caption.removesuffix('.')),
        label=label,
        position='htb',
        na_rep='',
        float_format='%.3f',
    )
    process_latex_table(tmp_file, path)


def process_latex_table(input_file: Path, output_file: Path | None = None) -> None:
    """Highlight max values in columns of a LaTeX table and add centering. Assumes standard LaTeX table format."""
    with open(input_file) as f:
        latex_content = f.read()

    lines = latex_content.split('\n')
    modified_lines = []
    data_rows = []
    in_data = False

    # Collect data rows and other lines
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith(r'\begin{table}'):
            modified_lines.append(line)
            modified_lines.append(r'\centering')  # Add centering
            continue
        if stripped_line == r'\midrule':
            in_data = True
            modified_lines.append(line)
            continue
        if stripped_line == r'\bottomrule':
            in_data = False
            # Process collected data rows
            processed_rows = _process_data_rows(data_rows)
            modified_lines.extend(processed_rows)
            modified_lines.append(line)
            data_rows = []
            continue
        if in_data:
            data_rows.append(line)
        else:
            modified_lines.append(line)

    # write back the modified content
    with open(output_file or input_file, 'w') as f:
        f.write('\n'.join(modified_lines))


def _parse_float(s: str) -> float | None:
    """Parse a string to a float, returning None if it fails."""
    value = re.match(r'[\d\.]+', s)
    if value:
        value = float(value.group())
    return value


def _process_data_rows(data_rows: list[str]) -> list[str]:
    # Collect numerical values per column
    n_cols = len(data_rows[0].split('&'))
    columns = [[] for _ in range(n_cols)]
    for row in data_rows:
        cells = [cell.strip() for cell in row.split('&')]
        while len(cells) < n_cols:
            cells.append('')
        for i in range(n_cols):
            cell_content = cells[i]
            if cell_content:
                try:
                    value = _parse_float(cell_content)
                    columns[i].append(value)
                except ValueError:
                    pass

    # Find max values for each column
    max_values = [
        max((x for x in col if x is not None), default=None) for col in columns
    ]

    # Process each row to add \textbf{} around max values
    processed_rows = []
    for row in data_rows:
        cells = [cell.strip() for cell in row.split('&')]
        while len(cells) < n_cols:
            cells.append('')
        modified_cells = []
        # Process numerical cells
        for i in range(n_cols):
            cell_content = cells[i]
            modified_cell = cell_content
            if cell_content:
                try:
                    value = _parse_float(cell_content)
                    if max_values[i] is not None and value == max_values[i]:
                        modified_cell = r'\textbf{' + cell_content + '}'
                except ValueError:
                    pass
            modified_cells.append(modified_cell)
        # Rebuild the row
        new_row = ' & '.join(modified_cells)
        processed_rows.append(new_row)

    return processed_rows
