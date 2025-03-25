# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

# %%
job_root = Path(f'jobs/{datetime.now().strftime("%Y-%m-%d")}-semantic-segmentation')
encoders = ['mit_b1']  # ['mit_b0', 'efficientnet-b0', 'efficientnet-b1', 'resnet18']
decoders = ['unet']  # , 'unetplusplus']
pprint(
    dict(
        job_root=job_root,
        encoders=encoders,
        decoders=decoders,
    )
)

# %%
for encoder in encoders:
    for decoder in decoders:
        print(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: encoder={encoder} | decoder={decoder}'
        )
        job_dir = job_root / f'{encoder}-{decoder}'
        job_dir.mkdir(parents=True, exist_ok=True)
        pm.execute_notebook(
            'train.ipynb',
            f'{str(job_dir)}.ipynb',
            parameters=dict(
                encoder=encoder,
                decoder=decoder,
            ),
        )

# COMMAND ----------
