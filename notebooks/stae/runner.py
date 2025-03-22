# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

# %%
job_dir_root = Path(f'jobs/{datetime.now().strftime("%Y-%m-%d-%H%M")}')
drivers = ['geordi', 'poli', 'michal', 'dans', 'jakub']
pprint(drivers)

# %%
for driver in drivers:
    print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: driver={driver}')
    job_dir = job_dir_root / driver
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        'train.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            driver=driver,
        ),
    )

# %%
