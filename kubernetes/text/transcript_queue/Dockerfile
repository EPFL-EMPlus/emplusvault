FROM transcription-whisperx:latest

ARG GITHUB_TOKEN

RUN echo "machine github.com login x-oauth-basic password $GITHUB_TOKEN" > ~/.netrc && \
    chmod 600 ~/.netrc && \ 
    cd emplusvault && \
    source .venv/bin/activate && \
    git fetch && \
    git checkout 7126bb6

CMD cd emplusvault && source .venv/bin/activate && python3 scripts/producer.py transcript

