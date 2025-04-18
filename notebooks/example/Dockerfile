FROM python:3.12-slim

# Use system Python
ENV UV_SYSTEM_PYTHON=1
ENV UV_HTTP_TIMEOUT=120
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install curl build-essential ca-certificates -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed uv binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

COPY requirements.txt .

RUN uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -r requirements.txt

# Start Jupyter server
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
