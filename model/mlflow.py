import pandas as pd
from mlflow.client import MlflowClient
from mlflow.entities import ViewType
from mlflow.entities.experiment import Experiment


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
                    f'metric.{k}': v for k, v in r.data.metrics.items()
                },  # Add all metrics with "metric." prefix
            }
            for r in all_runs
        ]
    )
    return runs_df
