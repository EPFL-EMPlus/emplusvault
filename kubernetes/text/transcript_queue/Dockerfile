FROM transcription-whisperx:latest

ARG GITHUB_TOKEN

RUN apt-get update && apt-get install dnsutils iputils-ping curl -y

RUN echo "machine github.com login x-oauth-basic password $GITHUB_TOKEN" > ~/.netrc && \
    chmod 600 ~/.netrc && \ 
    cd emplusvault && \
    source .venv/bin/activate && \
    git fetch && \
    git checkout feb2cf5

CMD cd emplusvault && source .venv/bin/activate && python3 scripts/producer.py transcript

# CMD ["sh", "-c", "while true; do sleep 1000; done"]