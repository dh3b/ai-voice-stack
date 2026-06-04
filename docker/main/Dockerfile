FROM python:3.11-slim-bookworm

# libportaudio2: shared lib sounddevice binds to (pulls libasound2 for ALSA/dev/snd).
# libsndfile1: audio file support. ca-certificates: HTTPS for the model download below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/main/requirements.txt ./requirements.main.txt
RUN pip install --no-cache-dir -r requirements.main.txt

# I wish I knew why it needs that
RUN python -c "import openwakeword.utils as u; u.download_models()"

COPY config.py ./config.py
COPY pipeline.py ./pipeline.py
COPY modules ./modules

CMD ["python", "pipeline.py"]
