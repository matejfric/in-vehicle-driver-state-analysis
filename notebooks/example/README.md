# Inference Example

## Create Python Environment

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
   - Linux and macOS:

        ```bash
        uv venv --python 3.12
        source .venv/bin/activate
        # only CPU version
        uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install -r requirements.txt --no-deps
        # or GPU version (assuming you have CUDA 12.x installed)
        # UV_HTTP_TIMEOUT=120 uv pip install -r requirements.txt
        ```

   - Windows:

        ```sh
        uv venv --python 3.12
        .venv\Scripts\activate
        # only CPU version
        uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
        uv pip install -r requirements.txt --no-deps
        # or GPU version (assuming you have CUDA 12.x installed)
        # set UV_HTTP_TIMEOUT=120
        # uv pip install -r requirements.txt
        ```

## Credits

- Depth Anything 2 ONNX export: [https://github.com/fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)
