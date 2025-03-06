# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

# %%
root_dir = Path().home() / 'source/driver-dataset/2024-10-28-driver-all-frames/'
people_dirs: list[Path] = list(root_dir.glob('*'))
people_dirs_and_subdirs: list[list[Path]] = [
    list(person_dir.glob('*')) for person_dir in people_dirs
]
all_dirs: list[Path] = [
    subdir for sublist in people_dirs_and_subdirs for subdir in sublist
]

now = datetime.now().strftime('%Y-%m-%d-%H%M')
job_root = Path('jobs/depth') / now

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
        parameters={'input_dir': str(dir / 'images'), 'dataset': 'mrl-515'},
    )

# %%
