FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user with audio group membership
RUN useradd -m -s /bin/bash assistant && usermod -aG audio assistant

COPY --chown=assistant:assistant . .

RUN mkdir -p /app/assets/confirmations && chown -R assistant:assistant /app/assets

USER assistant

CMD ["python3", "-m", "app.main"]
