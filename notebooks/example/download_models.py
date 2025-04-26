# %%
from pathlib import Path

import dagshub
import requests
from mlflow.artifacts import download_artifacts
from tqdm import tqdm

# %%
USER_NAME = 'matejfric'
LOCAL_DIR = Path().cwd() / 'models'

seg_model_uri = 'mlflow-artifacts:/7631752f9ba344e58f10d59f972ac342/a3084291e6b74d0fa39fa52c0801ce8c/artifacts/model'
tae_model_uri = 'mlflow-artifacts:/34294c9653fc48309a2302eb44b3be4b/f6aef4dd035340648ec48e696ca46c33/artifacts/model'
stae_model_uri = 'mlflow-artifacts:/59d0b3cc656e4fada574028be15d7bc9/5c05f9761b274acfbfcfbbc585a3a0f2/artifacts/model'

seg_model_local_path = LOCAL_DIR / 'seg'
tae_model_local_path = LOCAL_DIR / 'tae'
stae_model_local_path = LOCAL_DIR / 'stae'

# %%
# Segmentation model
dagshub.init('driver-seg', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=seg_model_uri,
    dst_path=str(seg_model_local_path),
)

# %%
# TAE model
dagshub.init('driver-tae', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=tae_model_uri,
    dst_path=str(tae_model_local_path),
)

# %%
# STAE model
dagshub.init('driver-stae', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=stae_model_uri,
    dst_path=str(stae_model_local_path),
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
