# %%
from datetime import datetime
from pathlib import Path

import papermill as pm

# %%
job_dir_root = Path(f'jobs/{datetime.now().strftime("%Y-%m-%d-%H%M")}')
drivers = ['geordi', 'poli', 'michal', 'dans', 'jakub']
source_types = ['rgbd', 'rgbdm']
image_sizes = [64]
latent_dims = [128]

# %%
for driver in drivers:
    for source_type in source_types:
        for image_size in image_sizes:
            for latent_dim in latent_dims:
                print(
                    f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: {driver} | {source_type} | {image_size} | {latent_dim}'
                )
                job_dir = job_dir_root / driver
                job_dir.parent.mkdir(parents=True, exist_ok=True)
                pm.execute_notebook(
                    'mrl.ipynb',
                    f'{str(job_dir)}_{source_type}_{image_size}_{latent_dim}.ipynb',
                    parameters=dict(
                        driver=driver,
                        source_type=source_type,
                        image_size=image_size,
                        latent_dim=latent_dim,
                    ),
                )

# %%
