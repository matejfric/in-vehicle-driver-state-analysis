# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm
import tqdm

# %%
root_dir = Path().home() / 'source/driver-dataset/dmd/'
session_dirs: list[Path] = list(root_dir.glob('*'))
session_subdirs: list[Path] = [
    ses_dir / class_dir
    for ses_dir in session_dirs
    for class_dir in ['normal', 'anomal']
]
all_subdirs: list[list[Path]] = [
    list(subdir.glob('*')) for subdir in session_subdirs if subdir.is_dir()
]
all_dirs = sorted([dir for subdirs in all_subdirs for dir in subdirs])
job_root = Path(f'jobs/dmd/{datetime.now().strftime("%Y-%m-%d")}')

pprint(all_dirs)

# %%
for dir in (pbar := tqdm.tqdm(all_dirs)):
    sequence = dir.name
    class_name = dir.parent.name
    session = dir.parent.parent.name
    job_dir = job_root / session / class_name / sequence
    pbar.set_postfix_str(f'{session}/{class_name}/{sequence}')

    job_dir.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        'job_batch_inference.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            input_dir=str(dir / 'rgb'),
            dataset='dmd',
        ),
    )

# COMMAND ----------
