# ------------------------------------
# Stage 1: Build environment
# ------------------------------------
    FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS builder

    ARG GITHUB_TOKEN
    SHELL ["/bin/bash", "-c"]

    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            git \
            python3.10 \
            python3-pip \
            python3-venv && \
        rm -rf /var/lib/apt/lists/*
    
    RUN echo "machine github.com login x-oauth-basic password $GITHUB_TOKEN" > ~/.netrc && \
        chmod 600 ~/.netrc && \
        git clone --depth=1 --branch main https://github.com/EPFL-EMPlus/emplusvault.git && \
        rm ~/.netrc && \
        cd emplusvault && rm -rf .git
    
    RUN cd emplusvault && \
        python3 -m venv .venv && \
        source .venv/bin/activate && \
        pip install --no-cache-dir -U pip setuptools && \
        pip install --no-cache-dir poetry && \
        poetry install --no-cache
    
    RUN cd emplusvault && \
        source .venv/bin/activate && \
        pip install --no-cache-dir opencv-python imutils ultralytics pika && \
        rm -rf ~/.cache/pip
    
    # ------------------------------------
    # Stage 2: Production environment
    # ------------------------------------
    FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
    
    SHELL ["/bin/bash", "-c"]
    
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            python3.10 \
            python3-pip \
            libgl1-mesa-glx \
            libglib2.0-0 && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy pre-built env + code from 'builder' stage
    COPY --from=builder /emplusvault /emplusvault
    
    ENV VIRTUAL_ENV=/emplusvault/.venv
    ENV PATH="$VIRTUAL_ENV/bin:$PATH"
    WORKDIR /emplusvault
    
    CMD ["bash", "-c", "source .venv/bin/activate && python scripts/consumer.py pose_binary"]
    