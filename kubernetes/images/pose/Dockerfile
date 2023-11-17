FROM python:3.10

ARG GITHUB_TOKEN

RUN echo "machine github.com login x-oauth-basic password $GITHUB_TOKEN" > ~/.netrc && \
    chmod 600 ~/.netrc && \
    git clone https://github.com/EPFL-EMPlus/emplusvault.git && \
    cd emplusvault && \
    python3 -m venv .venv && \
    .venv/bin/pip install -U pip setuptools && \
    .venv/bin/pip install poetry && \
    .venv/bin/poetry install && \
    .venv/bin/pip3 install torch torchvision torchaudio && \
    .venv/bin/poe download-pose-model && \
    pip3 install openpifpaf-vita && \
    pip3 install pika && \
    rm ~/.netrc

CMD ["./emplusvault/.venv/bin/python3 /emplusvault/scripts/consumer.py pose"]
