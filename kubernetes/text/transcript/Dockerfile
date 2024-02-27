FROM pose-openpifpaf:latest

RUN cd emplusvault && \
    source .venv/bin/activate && \
    apt-get install -y ffmpeg && \
    pip3 install pyannote.audio==2.1.1 && \
    pip3 install whisperx==3.1.1 && \
    pip3 install lightning-fabric && \
    # pip3 uninstall nvidia-cudnn-cu11 && \
    pip3 install nvidia-cudnn-cu11==8.9.6.50

RUN cd emplusvault && \
    source .venv/bin/activate && \
    python -m spacy download fr_core_news_lg
# COPY download_models.py /download_models.py

# RUN cd emplusvault && \
#     source .venv/bin/activate && \
#     python /download_models.py

CMD cd emplusvault && source .venv/bin/activate && python3 scripts/consumer.py transcript
# CMD ["sh", "-c", "while true; do sleep 1000; done"]
