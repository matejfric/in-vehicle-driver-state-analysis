# COMMAND ----------
from datetime import datetime
from pathlib import Path
from pprint import pprint

import papermill as pm


# COMMAND ----------
def list_directories(root_dir: Path) -> list[Path]:
    return [dir for dir in root_dir.glob('*') if dir.is_dir()]


root_dir = Path.home() / 'source/driver-dataset/images/'
driver_dirs: list[Path] = list_directories(root_dir)
driver_dirs_and_subdirs: list[list[Path]] = [
    list_directories(person_dir) for person_dir in driver_dirs
]
all_dirs: list[Path] = [
    subdir for sublist in driver_dirs_and_subdirs for subdir in sublist
]
# sessions = [
#     'gA_1_s1_2019-03-08T09;31;15+01;00',
#     'gA_1_s2_2019-03-08T09;21;03+01;00',
#     'gA_1_s3_2019-03-14T14;31;08+01;00',
#     'gA_1_s4_2019-03-22T11;49;58+01;00',
#     'gA_1_s5_2019-03-14T14;26;17+01;00',  # drowsiness
#     'gA_1_s6_2019-03-08T09;15;15+01;00',
#     'gA_1_s6_2019-03-22T11;59;56+01;00',
#     'gA_2_s1_2019-03-08T10;01;44+01;00',
#     'gA_2_s2_2019-03-08T09;50;49+01;00',
#     'gA_3_s1_2019-03-08T10;27;38+01;00',
#     'gA_3_s2_2019-03-08T10;16;48+01;00',
#     'gA_4_s1_2019-03-13T10;36;15+01;00',
#     'gA_4_s2_2019-03-13T10;43;06+01;00',
#     'gA_5_s1_2019-03-08T10;57;00+01;00',
#     'gA_5_s2_2019-03-08T10;46;46+01;00',
# ]
# all_dirs = [dir for dir in all_dirs if any(session in str(dir) for session in sessions)]

pprint(all_dirs)

# COMMAND ----------
# ~2 hours for 14383 images

for dir in all_dirs:
    path_parts = str(dir).split('/')
    output_subdir = path_parts[-2]
    output_name = path_parts[-1]

    print(
        f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processing: {output_subdir}/{output_name}'
    )

    job_dir = Path(f'jobs/{output_subdir}')
    job_dir.mkdir(parents=True, exist_ok=True)

    pm.execute_notebook(
        'job_sam2.ipynb',
        job_dir / f'{output_name}.ipynb',
        parameters=dict(
            image_dir=str(dir),
        ),
    )

# COMMAND ----------
