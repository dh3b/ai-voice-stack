FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopenblas-dev \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN set -e; \
    LLAMA_CMAKE_ARGS=""; \
    if command -v nvcc >/dev/null 2>&1; then \
        echo "CUDA detected — building llama-cpp-python with CUDA support"; \
        LLAMA_CMAKE_ARGS="-DLLAMA_CUDA=ON -DLLAMA_NATIVE=ON"; \
    elif pkg-config --exists openblas 2>/dev/null || [ -f /usr/lib/aarch64-linux-gnu/libopenblas.so ] || [ -f /usr/lib/x86_64-linux-gnu/libopenblas.so ]; then \
        echo "OpenBLAS detected — building llama-cpp-python with OpenBLAS support"; \
        LLAMA_CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_METAL=OFF"; \
    else \
        echo "No accelerator detected — building llama-cpp-python with baseline"; \
        LLAMA_CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_METAL=OFF"; \
    fi; \
    CMAKE_ARGS="$LLAMA_CMAKE_ARGS" pip install llama-cpp-python --no-binary llama-cpp-python

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["python3", "-m", "app.server"]
