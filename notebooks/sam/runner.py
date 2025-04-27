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
print(f'Found {len(all_dirs)} directories')
pprint(all_dirs)

# COMMAND ----------
# # Rename files as needed
# for dir in all_dirs:
#     path = dir / 'sam2_output/masks'

#     # Sort files in reverse order to avoid overwriting
#     imgs = sorted(path.glob('*.png'), reverse=True)

#     for img in imgs:
#         num = int(img.stem)  # '00000' -> 0
#         new_name = f'{num + 1:05d}.png'  # 0+1=1 -> '00001.png'
#         img.rename(img.with_name(new_name))

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
