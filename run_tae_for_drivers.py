# %%
from datetime import datetime
from pathlib import Path

import papermill as pm

# %%
drivers = ['geordi', 'poli', 'michal', 'dans', 'jakub']
for driver in drivers:
    print(f'{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} processing: {driver}')
    job_dir = Path(f"jobs/{datetime.now().strftime('%Y-%m-%d')}") / driver
    job_dir.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        'temporal_anomaly_detection.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            driver=driver,
        ),
    )

# %%
