# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

from model.dmd import CATEGORIES, ROOT

# %%
session = 'gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body'
sequencies: list[list[Path]] = [
    list((ROOT / session / cat_dir).glob('*')) for cat_dir in CATEGORIES
]
all_dirs: list[Path] = [subdir for sublist in sequencies for subdir in sublist]
job_root = Path(f'jobs/depth_dmd/{session}')

pprint(all_dirs)

# %%

for dir in all_dirs:
    path_parts = str(dir).split('/')
    output_subdir = path_parts[-2]
    output_name = path_parts[-1]

    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: {output_subdir}/{output_name}'
    )

    (job_root / output_subdir).mkdir(parents=True, exist_ok=True)

    job_dir = job_root / output_subdir / output_name
    pm.execute_notebook(
        'job_depth.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            input_dir=str(dir / 'rgb'),
            dataset='dmd',
        ),
    )

# %%
