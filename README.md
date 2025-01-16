# Driver State Analysis

## 1. Installation

```bash
conda env create -f environment.yml -n driver-state-analysis
```

## 2. High Level Overview

### 2.1. Input Data Processing

```mermaid
mindmap
    root((RGB Image))
        (GT Mask)
            YOLOv8x
                Segment Anything 2
        (Segmentation)
            EfficientNetB0
                U-Net
        (Monocular Depth Estimation)
            Depth Anything 2
            Intel MiDaS ...worse
        (Edge Detection)
            Canny
            Sobel
            Laplacian
        LBP
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
        (Regularized)
            Encoder Outputs
            Model Weights
            Prediction
        Contractive
        MAE - L1
        MSE - L2
```
