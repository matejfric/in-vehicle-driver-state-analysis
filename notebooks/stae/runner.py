# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Literal

import papermill as pm

# %%
job_dir_root = Path(f'jobs/{datetime.now().strftime("%Y-%m-%d-%H%M")}')
drivers = ['geordi']  # [1, 2, 3, 4, 5, 'geordi', 'poli', 'michal', 'dans', 'jakub']
source_types = ['depth']  # ['masks', 'images', 'depth']
dataset: Literal['mrl', 'dmd'] = 'mrl'
pprint(
    dict(
        job_dir_root=job_dir_root,
        drivers=drivers,
        source_types=source_types,
        dataset=dataset,
    )
)

# %%
for driver in drivers:
    for source_type in source_types:
        print(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: driver={driver}, source_type={source_type}'
        )
        job_dir = (
            job_dir_root
            / (driver if isinstance(driver, str) else f'driver_{driver}')
            / source_type
        )
        job_dir.parent.mkdir(parents=True, exist_ok=True)
        pm.execute_notebook(
            'train.ipynb',
            f'{str(job_dir)}.ipynb',
            parameters=dict(
                driver=driver,
                source_type=source_type,
                dataset=dataset,
            ),
        )

# %%
