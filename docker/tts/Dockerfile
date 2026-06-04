FROM python:3.11-slim-bookworm

# libgomp1: OpenMP runtime used by onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/tts/requirements.txt ./requirements.tts.txt
RUN pip install --no-cache-dir -r requirements.tts.txt

COPY config.py ./config.py
COPY modules/__init__.py ./modules/__init__.py
COPY modules/server/__init__.py ./modules/server/__init__.py
COPY modules/server/tts_server.py ./modules/server/tts_server.py

CMD ["python", "-m", "modules.server.tts_server"]
