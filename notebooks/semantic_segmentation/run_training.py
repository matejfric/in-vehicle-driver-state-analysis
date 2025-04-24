# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm

# %%
job_root = Path(
    f'jobs/{datetime.now().strftime("%Y-%m-%d-%H%M")}-semantic-segmentation'
)
encoders = [
    'efficientnet-b0',
    'efficientnet-b1',
    'efficientnet-b2',
    'efficientnet-b3',
    'resnet18',
    'mit_b0',
    'mit_b1',
]
decoders = ['unet', 'unetplusplus']
pprint(
    dict(
        job_root=job_root,
        encoders=encoders,
        decoders=decoders,
    )
)

# %%
job_root.mkdir(parents=True, exist_ok=True)
for encoder in encoders:
    for decoder in decoders:
        if encoder.startswith('mit') and decoder == 'unetplusplus':
            # Skip this combination as it is not supported.
            continue
        print(
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: encoder={encoder} | decoder={decoder}'
        )
        job_path = job_root / f'{encoder}-{decoder}'
        pm.execute_notebook(
            'train.ipynb',
            f'{str(job_path)}.ipynb',
            parameters=dict(
                encoder=encoder,
                decoder=decoder,
            ),
        )

# COMMAND ----------
