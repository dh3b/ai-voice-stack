ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG SIMULSTREAMING_REF=

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA < 13.0
RUN pip install --no-cache-dir "numpy<2" "pydantic-settings>=2.4,<3"

RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch torchaudio; \
    else \
        pip install --no-cache-dir torch torchaudio; \
    fi

RUN git clone --depth 1 https://github.com/dh3b/SimulStreaming-lite simulstreaming_lib \
    && if [ -n "$SIMULSTREAMING_REF" ]; then \
         git -C simulstreaming_lib fetch --depth 1 origin "$SIMULSTREAMING_REF" \
         && git -C simulstreaming_lib checkout FETCH_HEAD; \
       fi \
    && pip install --no-cache-dir -r simulstreaming_lib/requirements_whisper.txt

COPY config.py ./config.py
COPY modules/__init__.py ./modules/__init__.py
COPY modules/server/__init__.py ./modules/server/__init__.py
COPY modules/server/stt_server.py ./modules/server/stt_server.py

CMD ["python", "-m", "modules.server.stt_server"]
