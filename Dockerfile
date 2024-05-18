FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
# WORKDIR /workspace

# Install dependencies
RUN pip install jupyterlab==4.0.13 \
                dagshub==0.3.27 \
                mlflow==2.12.2 \
                albumentations==1.4.7 \
                pytorch-lightning==2.2.4 \
                segmentation-models-pytorch==0.3.3 \
                tensorboard==2.16.2

# Install git and configure it
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean && \
    git config --global --add safe.directory '*'

# Start Jupyter server
CMD jupyter lab --no-browser --ip=0.0.0.0 --allow-root
