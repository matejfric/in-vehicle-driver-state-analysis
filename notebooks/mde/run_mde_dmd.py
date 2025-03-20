# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm
from tqdm import tqdm

from model.dmd import CATEGORIES, ROOT

# %%
session_dirs: list[Path] = [
    x
    for x in ROOT.glob('*')
    if x.is_dir()
    and x.name != 'gA_1_s1_2019-03-08T09;31;15+01;00'
    and x.name.startswith('gA_1')
]
pprint(session_dirs)

session_subdirs: list[Path] = [
    ses_dir / class_dir for ses_dir in session_dirs for class_dir in CATEGORIES
]
all_subdirs: list[list[Path]] = [
    list(subdir.glob('*')) for subdir in session_subdirs if subdir.is_dir()
]
all_dirs = sorted(
    [
        dir
        for subdirs in all_subdirs
        for dir in subdirs
        if (dir / 'rgb').is_dir() and len(list((dir / 'rgb').glob('*.jpg'))) > 0
    ]
)
job_root = Path(f'jobs/dmd/{datetime.now().strftime("%Y-%m-%d")}')

pprint(all_dirs[:3])

# %%

for dir in (pbar := tqdm(all_dirs)):
    sequence = dir.name
    class_name = dir.parent.name
    session = dir.parent.parent.name
    job_dir = job_root / session / class_name / sequence
    pbar.set_postfix_str(f'{session}/{class_name}/{sequence}')

    job_dir.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        'job_mde.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            input_dir=str(dir / 'rgb'),
            dataset='dmd',
        ),
    )

# %%
