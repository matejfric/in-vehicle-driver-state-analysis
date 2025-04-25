# Inference Example

## Installation

### Docker

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Install [just](https://github.com/casey/just).
3. Build the Docker image and run the container:
   1. For CPU inference:

      ```sh
      # 2.27GB
      cd notebooks/example/ && sudo just jupyter
      ```

   2. For GPU-accelarated inference:
      1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
      2. Run

      ```sh
      # 14.8GB
      cd notebooks/example/ && sudo just jupyter --cuda
      ```

4. Open the Jupyter Notebook [`pipeline.ipynb`](./pipeline.ipynb) in your browser at [http://localhost:8888](http://localhost:8888). You will be asked to enter the access token that you will find in the terminal.

### Local

Note: This method is not recommended.

1. Install [uv](https://docs.astral.sh/uv/) package manager:
   - Linux and macOS:

      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

   - Windows:

      ```sh
      powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
      ```

2. Create a new Python environment:

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   # Windows:
   # .venv\Scripts\activate
   ```

   - CPU inference

      ```sh
      uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
      uv pip install -r requirements.txt
      ```

   - GPU-accelarated inference

      ```sh
      uv pip install torch==2.5.1
      uv pip install onnxruntime-gpu==1.21.0
      uv pip install -r requirements.txt
      ```

3. Run [`pipeline.py`](./pipeline.py)

   ```bash
   python pipeline.py
   ```

4. Or alternatively, start a Jupyter server and open the notebook [`pipeline.ipynb`](./pipeline.ipynb) in your browser.

   ```bash
   jupyter lab --ip=0.0.0.0 --allow-root
   ```

## Notes

The autoencoder model is not identical to the one described in the thesis, because the original model is using a different implementation of the [`TimeDistributed`](../../model/ae/common.py) layer which is not compatible with ONNX. The newer ONNX-compatible model uses `TimeDistributedV2` and achieves ~0.2% higher AU-ROC on the test set while being significantly faster thanks to ONNX Runtime.

## References

- Depth Anything 2 ONNX export: [https://github.com/fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)
