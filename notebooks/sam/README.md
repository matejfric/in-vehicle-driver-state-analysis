# SAM-2 Driver Segmentation Mask Generation

The goal of this submodule is to generate "pseudo-ground-truth" segmentation masks for the MRL Driver dataset.

## SAM-2 Instalation

- Ubuntu 24.04 (WSL-2)
- NVIDIA-SMI 560.27
- CUDA 12.5
- Python 3.12.3

```sh
git clone https://github.com/facebookresearch/segment-anything-2.git
git checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4

# Install CUDA
sudo sh cuda_12.5.1_555.42.06_linux.run
sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
source switch_cuda 12.5

# Create virtual environment
python3 -m venv .venv
. .venv/bin/activate

pip install wheel setuptools ninja
cd segment-anything-2 && CC=gcc-11 pip install -e ".[notebooks]"
```

## Additional Requirements

```sh
sudo apt install ffmpeg
pip install -r requirements.txt
```

## Run

We use [`papermill`](https://github.com/nteract/papermill) to run the notebook `job_sam2.ipynb` for all videos.

```sh
python runner.py
```
