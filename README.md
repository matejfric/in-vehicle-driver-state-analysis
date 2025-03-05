# Driver State Analysis

## 1. Installation

- Ubuntu 24.04

```bash
conda env create -f environment.yml -n driver-state-analysis
pip install -e .
```

## 2. High Level Overview

### 2.1. Input Data Processing

```mermaid
mindmap
    root((RGB Image))
        (GT Mask)
            YOLOv8x
                Segment Anything 2
        (Semantic Segmentation)
            EfficientNetB0
                U-Net
        (Monocular Depth Estimation)
            Depth Anything
                Depth Anything 2
            Intel MiDaS ...worse
        (Edge Detection)
            Canny
            Sobel
            Laplacian
        (Pose Estimation)
            Graph NN
            YOLOv11
        LBP
        OpenAI CLIP
        Text
```

### 2.2. Anomaly Detection

#### 2.2.1. Autoencoders

```mermaid
mindmap
    root(("`AE`"))
        Dense
        Convolutional
        Sequence
            Conv LSTM
            Conv 3D
            Spatio-Temporal 3D
        Single Image
```

#### 2.2.2. Loss Functions

```mermaid
mindmap
    root(("`AE Loss`"))
        MAE - L1
        MSE - L2
        (Regularized)
            Encoder Outputs
            Model Weights
            Prediction
            Contractive
```
