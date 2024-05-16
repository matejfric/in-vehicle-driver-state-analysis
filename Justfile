IMAGE := "corrosion:latest"

# List all available commands
default:
    @just --list

# Install pre-commit hooks
pre-commit:
    pre-commit install

build:
    docker buildx build --load -f Dockerfile -t {{ IMAGE }} .

# Run a command in the docker container
_docker_run *args: build
    #!/usr/bin/env sh
    docker run --rm --runtime=nvidia --gpus all -it -e PYTHONPATH="${PWD}" --workdir "${PWD}" -v /mnt:/mnt -v "${PWD}":"${PWD}" {{ IMAGE }} {{ args }}

# Run Jupyter server
notebook: build
    #!/usr/bin/env sh
    docker run --rm --runtime=nvidia --gpus all -it -e PYTHONPATH="${PWD}" -p 8888:8888 --workdir "${PWD}" -v /mnt:/mnt -v "${PWD}":"${PWD}" {{ IMAGE }}
    # Notebook needs port forwarding.

python: (_docker_run "python")

bash: (_docker_run "bash")
