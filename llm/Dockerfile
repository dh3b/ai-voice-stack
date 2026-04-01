FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Make nvcc findable if CUDA toolkit is mounted by the NVIDIA container runtime
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# ── System deps ──────────────────────────────────────────────────────────────
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
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

# ── llama-cpp-python (CUDA-aware) ─────────────────────────────────────────────
# On Jetson with --runtime nvidia, /usr/local/cuda is bind-mounted.
# We probe that path rather than `command -v nvcc`.
RUN set -e; \
    if [ -d /usr/local/cuda ] && [ -f /usr/local/cuda/bin/nvcc ]; then \
    echo "CUDA detected — building llama-cpp-python with CUDA support"; \
    CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_NATIVE=ON" \
    pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir; \
    else \
    echo "CUDA not found — falling back to OpenBLAS (CPU) build"; \
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_NATIVE=ON" \
    FORCE_CMAKE=1 \
    pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir; \
    fi

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["python3", "-m", "app.server"]
