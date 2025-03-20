# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

# %%
job_dir_root = Path(f'jobs/{datetime.now().strftime("%Y-%m-%d-%H%M")}')

# ['geordi', 'poli', 'michal', 'dans', 'jakub', 1, 2, 3, 4, 5]
drivers = [1]

# Some choices are only available for DMD and MRL respectively.
# ['masks', 'depth', 'source_depth', 'rgb_source_depth', 'rgb', 'rgbd', 'rgbdm']
source_types = ['rgb_source_depth']

image_sizes = [64]
latent_dims = [128]
dataset = 'dmd'

pprint(
    dict(
        dataset=dataset,
        drivers=drivers,
        source_types=source_types,
        image_sizes=image_sizes,
        latent_dims=latent_dims,
    )
)

# %%
for driver in drivers:
    for source_type in source_types:
        for image_size in image_sizes:
            for latent_dim in latent_dims:
                print(
                    f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: driver={driver} | type={source_type} | size={image_size} | latent={latent_dim}'
                )
                job_dir = job_dir_root / (
                    driver if isinstance(driver, str) else f'driver_{driver}'
                )
                job_dir.parent.mkdir(parents=True, exist_ok=True)
                pm.execute_notebook(
                    'train.ipynb',
                    f'{str(job_dir)}_{source_type}_{image_size}_{latent_dim}.ipynb',
                    parameters=dict(
                        driver=driver,
                        source_type=source_type,
                        image_size=image_size,
                        latent_dim=latent_dim,
                        dataset=dataset,
                    ),
                )

# %%
