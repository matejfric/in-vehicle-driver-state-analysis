# %%
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm
import tqdm

from model.dmd import ROOT, get_all_sessions

# %%
sessions = get_all_sessions()
all_dirs = [ROOT / s for s in sessions if (ROOT / s).is_dir()]
job_root = Path(f'jobs/dmd/{datetime.now().strftime("%Y-%m-%d")}')
job_root.mkdir(parents=True, exist_ok=True)

print(f'Job root: {job_root}')
print(f'Found {len(all_dirs)} sessions. Example:')
pprint(all_dirs)

# %%
for dir in (pbar := tqdm.tqdm(all_dirs)):
    session_name = dir.name.split(';')[0]
    job_dir = job_root / session_name
    pbar.set_postfix_str(session_name)
    pm.execute_notebook(
        'job_batch_inference.ipynb',
        f'{str(job_dir)}.ipynb',
        parameters=dict(
            model_name='pytorch-sem-seg',
            model_version=26,
            input_dir=str(dir),
            dataset='dmd',
            source_type='ir',
        ),
    )

# COMMAND ----------
