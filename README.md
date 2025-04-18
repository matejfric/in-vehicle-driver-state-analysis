# In-Vehicle Driver State Analysis

[![DOI](https://zenodo.org/badge/858760757.svg)](https://doi.org/10.5281/zenodo.15242002)

![PipelineAnimation](https://github.com/user-attachments/assets/de14f020-1b32-4931-8702-5cd302d5a9de)

(Model results on the Driver Monitoring Dataset [[1]](#ref1).)

---

- [1. Project Organization](#1-project-organization)
- [2. Development Environment](#2-development-environment)
- [3. What can we do with an RGB image?](#3-what-can-we-do-with-an-rgb-image)
- [4. References](#4-references)

## 1. Project Organization

- [`model`](model/) - library code
- [`notebooks`](notebooks/) - runnable code, primarily Jupyter notebooks, and Python scripts used to orchestrate notebook execution with different parameters
  - [`notebooks/example`](notebooks/example/) - demonstration of an inference pipeline on sample data. Instructions for running the pipeline are provided [here](notebooks/example/README.md)
  - [`notebooks/datasets`](notebooks/datasets/) - dataset pre-processing
  - [`notebooks/eda`](notebooks/eda/) - exploratory data analysis
  - [`notebooks/sam`](notebooks/sam/) - Segment Anything Model 2 (SAM 2) inference (generation of groud truth masks). This module uses a different virtual environment than the rest of the project, as described [here](notebooks/sam/README.md)
  - [`notebooks/semantic_segmentation`](notebooks/semantic_segmentation/) - semantic segmentation model training, evaluation, and inference
  - [`notebooks/tae`](notebooks/tae/) - training and evaluation of Temporal Autoencoder (TAE)
  - [`notebooks/stae`](notebooks/stae/) - training and evaluation of Spatio-Temporal Autoencoder (STAE)
  - [`notebooks/memory_map`](notebooks/memory_map/) - export training data to continuous memory-mapped files (`np.memmap`) to speed up training and reduce CPU usage, at the cost of increased disk space
  - [`notebooks/clip`](notebooks/clip/) - OpenAI CLIP model (this is excluded from the thesis)

## 2. Development Environment

- Ubuntu 24.04
- CUDA 12.5
- Python 3.12

```bash
conda env create -f environment.yml -n driver-state-analysis
pip install -e .
pre-commit install
```

## 3. What can we do with an RGB image?

The following diagram illustrates a (non-exhaustive) list of different tasks that can be performed using an RGB image as input:

```mermaid
mindmap
    root)RGB Image)
        (GT Mask)
            YOLOv8x
                Segment Anything 2
        (Semantic Segmentation)
            EfficientNet
            ResNet
            U-Net
            UNet++
        (Monocular Depth Estimation)
            Image
                MiDaS
                Marigold
                Depth Anything
                    Depth Anything 2
            Video
                Video Depth Anything
                DepthCrafter
        (Edge Detection)
            Canny
            Sobel
            Laplacian
        (Pose Estimation)
            Graph NN
            YOLOv11
        LBP
        HOG
        CLIP
        Text
        Skin Segmentation
```

## 4. References

1. <a name="ref1"></a> Ortega, J., Kose, N., Cañas, P., Chao, M.a., Unnervik, A., Nieto, M., Otaegui, O., & Salgado, L. (2020). DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Alertness Analysis. In: A. Bartoli & A. Fusiello (eds), Computer Vision -- ECCV 2020 Workshops (pg. 387–405). Springer International Publishing.
