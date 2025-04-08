# %%
from pathlib import Path

import dagshub
import requests
from mlflow.artifacts import download_artifacts
from tqdm import tqdm

# %%
USER_NAME = 'matejfric'
LOCAL_DIR = Path().cwd() / 'models2'

seg_model_uri = 'mlflow-artifacts:/7631752f9ba344e58f10d59f972ac342/03d2f6c0e7794ff0b3e2c6b70af93a49/artifacts/model'
ae_model_uri = 'mlflow-artifacts:/34294c9653fc48309a2302eb44b3be4b/175a88ca24ff4f7f8b265cf07a8d85e4/artifacts/model'

seg_model_local_path = LOCAL_DIR / 'seg'
ae_model_local_path = LOCAL_DIR / 'ae'

# %%
# Segmentation model
dagshub.init('driver-seg', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=seg_model_uri,
    dst_path=str(seg_model_local_path),
)

# %%
# Autoencoder model
dagshub.init('driver-tae', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=ae_model_uri,
    dst_path=str(ae_model_local_path),
)

# %%
# Depth Anything v2
url = 'https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx'
local_filename = LOCAL_DIR / url.split('/')[-1]
with requests.get(url, stream=True) as response:
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(local_filename, 'wb') as file:
        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=local_filename.name,
            initial=0,
            miniters=1,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
                    pbar.update(len(chunk))
