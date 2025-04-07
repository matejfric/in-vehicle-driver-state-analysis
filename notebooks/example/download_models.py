from pathlib import Path

import dagshub
from mlflow.artifacts import download_artifacts

USER_NAME = 'matejfric'

seg_model_uri = 'mlflow-artifacts:/7631752f9ba344e58f10d59f972ac342/03d2f6c0e7794ff0b3e2c6b70af93a49/artifacts/model'
ae_model_uri = 'mlflow-artifacts:/34294c9653fc48309a2302eb44b3be4b/175a88ca24ff4f7f8b265cf07a8d85e4/artifacts/model'

seg_model_local_path = Path().cwd() / 'models' / 'seg'
ae_model_local_path = Path().cwd() / 'models' / 'ae'

dagshub.init('driver-seg', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=seg_model_uri,
    dst_path=str(seg_model_local_path),
)

dagshub.init('driver-tae', USER_NAME, mlflow=True)  # type: ignore
download_artifacts(
    artifact_uri=ae_model_uri,
    dst_path=str(ae_model_local_path),
)

# https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx
