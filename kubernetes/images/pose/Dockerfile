FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ARG GITHUB_TOKEN

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y git python3.10 python3-pip python3-venv 
# RUN apt-get install -y curl \
#     libgl1-mesa-glx \
#     libgtk2.0-dev \
#     # ffmpeg \
#     git

RUN echo "machine github.com login x-oauth-basic password $GITHUB_TOKEN" > ~/.netrc && \
    chmod 600 ~/.netrc && \
    git clone https://github.com/EPFL-EMPlus/emplusvault.git

RUN cd emplusvault && \
    python3 -m venv .venv && \
    source .venv/bin/activate && \
    .venv/bin/pip install -U pip setuptools

RUN cd emplusvault && \
    source .venv/bin/activate && \
    .venv/bin/pip install poetry

RUN cd emplusvault && \
    source .venv/bin/activate && \
    .venv/bin/poetry install && \
    .venv/bin/pip3 install torch torchvision torchaudio

RUN apt-get clean && apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

RUN cd emplusvault && \
    source .venv/bin/activate && \
    pip3 install openpifpaf-vita && \
    poe download-pose-model && \
    pip3 install pika && \
    rm ~/.netrc

# CMD tail -f /dev/null

CMD cd emplusvault && source .venv/bin/activate && python3 scripts/consumer.py pose
