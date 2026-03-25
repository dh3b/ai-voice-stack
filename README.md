# AI Voice Assistant Stack

Asynchronous AI voice assistant with microservices architecture, optimized for ARM64 (Jetson Orin Nano).

## Architecture

The system is split into 5 microservices:

1. **Wakeword Service** - Detects wake phrases using OpenWakeWord
2. **STT Service** - Speech-to-text using Faster Whisper
3. **LLM Service** - Response generation using llama-cpp-python
4. **TTS Service** - Text-to-speech using Piper
5. **Orchestrator** - Main pipeline coordinator

## Features

- Asynchronous pipeline with streaming between services
- Wakeword detection with instant cancellation
- Audio confirmation after wakeword detection
- Chat continuation (optional)
- Tool/function calling for agent mode
- Auto-detection of audio devices
- Optimized llama-cpp build (CUDA/OpenBLAS/baseline)

## Quick Start

1. Create `.env` file from template:
```bash
cp .env.example .env
```

2. Download models:
```bash
# Create model directories
mkdir -p models/{llm,stt,tts,wakeword}

# Download your models to respective directories
# LLM: Place GGUF model in models/llm/
# STT: Faster-whisper downloads automatically
# TTS: Download Piper voice to models/tts/
# Wakeword: OpenWakeWord downloads automatically
```

3. Build and run:
```bash
docker-compose up --build
```

## Configuration

Edit `.env` file to customize:

- Model paths and configurations
- Audio device selection
- Pipeline behavior (chat continuation, timeouts)
- LLM parameters (temperature, context size, etc.)
- TTS voice selection

## Adding Custom Tools

Edit `orchestrator/app/tool_registry.py`:

```python
@registry.register(description="Your tool description")
def my_tool(param1: str, param2: int) -> str:
    # Your implementation
    return result
```

Tools are automatically available to the LLM agent.

## Volume Mounts

- `./models/` - Model storage
- `./orchestrator/app/assets/confirmations/` - Generated confirmation audio

## Ports

- 8001: Wakeword service
- 8002: STT service
- 8003: LLM service
- 8004: TTS service

## Requirements

- Docker & Docker Compose
- Audio devices (microphone & speaker)
- Models downloaded to appropriate directories

## Notes

- First run generates confirmation audio files
- llama-cpp-python builds with optimal acceleration (CUDA/OpenBLAS)
- Streaming reduces latency between pipeline stages
- Wakeword detection runs continuously and can interrupt ongoing responses
