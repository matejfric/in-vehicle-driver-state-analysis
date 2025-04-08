# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: torch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check ONNX Model Accuracy Compared to Transformers Model

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from common import ONNXModel, normalize, preprocess_depth_anything_v2
from transformers import pipeline

# %%
mde_model_onnx = ONNXModel('models/depth_anything_v2_vits.onnx')

# %%
mde_model_transformers = pipeline(
    task='depth-estimation',
    model='depth-anything/Depth-Anything-V2-Small-hf',
    device=0,
)

# %%
img_rgb = cv2.cvtColor(cv2.imread('images/000000.jpg'), cv2.COLOR_BGR2RGB)
img = preprocess_depth_anything_v2(img_rgb)

# %%
out_onnx = mde_model_onnx.predict(img)
out_onnx = normalize(out_onnx).squeeze()

# %%
out_transformers = mde_model_transformers.predict('images/000000.jpg')[  # type: ignore
    'predicted_depth'
]
out_transformers = normalize(out_transformers.numpy())

# %%
print(
    np.linalg.norm(out_onnx - out_transformers)
    / np.linalg.norm(out_onnx + out_transformers)
)

# %%
# Plot differences
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(out_transformers)
plt.title('onnx')
plt.subplot(1, 3, 2)
plt.imshow(out_onnx)
plt.title('transformers')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(out_transformers - out_onnx))
plt.title('diff')
plt.colorbar(shrink=0.5)
plt.tight_layout()
plt.show()
