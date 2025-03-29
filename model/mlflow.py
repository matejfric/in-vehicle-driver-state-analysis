import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mlflow.client import MlflowClient
from mlflow.entities import ViewType
from mlflow.entities.experiment import Experiment
from tqdm import tqdm


def download_all_runs(
    client: MlflowClient, experiments: list[Experiment]
) -> pd.DataFrame:
    """Get all runs from the experiments."""
    all_runs = []
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string='',
            run_view_type=ViewType.ACTIVE_ONLY,
        )
        all_runs.extend(runs)

    # Create a DataFrame from the runs
    runs_df = pd.DataFrame(
        [
            {
                'run_id': r.info.run_id,
                'experiment_id': r.info.experiment_id,
                'experiment_name': client.get_experiment(r.info.experiment_id).name,
                'status': r.info.status,
                'start_time': pd.to_datetime(r.info.start_time, unit='ms'),
                'end_time': pd.to_datetime(r.info.end_time, unit='ms')
                if r.info.end_time
                else None,
                'artifact_uri': r.info.artifact_uri,
                **r.data.params,  # Add all parameters
                **{
                    # Add all metrics with "metric." prefix
                    f'metric.{k}': v
                    for k, v in r.data.metrics.items()
                },
                **{
                    # Add all tags with "tag." prefix
                    f'tag.{k}': v
                    for k, v in r.data.tags.items()
                },
            }
            for r in all_runs
        ]
    )
    return runs_df


def download_predictions(
    client: MlflowClient,
    df: pd.DataFrame,
    artificat_dir: str | Path = 'outputs',
    local_root_dir: str | Path = 'outputs/mlflow_artifacts',
) -> pd.DataFrame:
    """Download the predictions from MLflow artifacts and save them to local paths."""

    if 'run_id' not in df.columns:
        raise Exception('DataFrame must contain `run_id` column.')

    df = df.assign(local_path=None)
    local_root = Path(local_root_dir)
    artifact_dir = Path(artificat_dir)

    # Loop through each row in the dataframe
    for index, row in tqdm(df.iterrows(), total=len(df)):
        run_id = row['run_id']
        # Download artifacts and store the path
        local_dir = local_root / str(run_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = client.download_artifacts(
            run_id, str(artifact_dir / 'predictions.json'), str(local_dir)
        )
        # Save the local path to the dataframe
        df.at[index, 'local_path'] = local_path

    return df


def load_predictions(
    df: pd.DataFrame,
    source_type_map: dict[str, str] | None = None,
) -> dict:
    """Load the predictions from the local paths."""

    if 'local_path' not in df.columns:
        raise Exception(
            'DataFrame must contain `local_path` column. Run `download_predictions` first.'
        )

    data = defaultdict(dict)
    for _index, row in df.iterrows():
        with open(row['local_path']) as f:
            results = json.load(f)
        source_type = (
            source_type_map[row['source_type']]
            if source_type_map
            else row['source_type']
        )
        data[row['driver']][source_type] = results
    return data
