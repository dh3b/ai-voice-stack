# Architecture Documentation

## System Overview

The AI Voice Assistant uses a microservices architecture where each component runs in its own Docker container and communicates via HTTP APIs. This design enables:

- Independent scaling of services
- Easy replacement of components
- Streaming data between services for low latency
- Fault isolation

## Pipeline Flow

```
┌─────────────┐
│  Wakeword   │ ◄─── Continuous audio monitoring
│  Detection  │
└──────┬──────┘
       │ Detected!
       ▼
┌─────────────┐
│Confirmation │ ──► Play random "mhm" sound
│   Audio     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Record    │ ──► Capture until silence
│   Speech    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     STT     │ ──► Transcribe to text
│  (Whisper)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │ ──► Generate response (streaming)
│   (Llama)   │
└──────┬──────┘
       │ Stream tokens
       ▼
┌─────────────┐
│     TTS     │ ──► Synthesize speech (streaming)
│   (Piper)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Speaker   │ ──► Play audio
│   Output    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Chat     │ ──► Optional: Wait for continuation
│Continuation │
└─────────────┘
```

## Service Details

### Orchestrator
- **Port**: N/A (client)
- **Role**: Main coordinator
- **Key Features**:
  - Manages audio I/O
  - Coordinates pipeline flow
  - Handles wakeword interruption
  - Implements chat continuation
  - Tool registry for agent functions

### Wakeword Service
- **Port**: 8001
- **Technology**: OpenWakeWord
- **Endpoints**:
  - `POST /detect` - Detect wakeword in audio chunk
  - `GET /health` - Health check
- **Input**: Raw audio bytes (int16)
- **Output**: JSON with detection result

### STT Service
- **Port**: 8002
- **Technology**: Faster Whisper
- **Endpoints**:
  - `POST /transcribe` - Transcribe audio to text
  - `GET /health` - Health check
- **Input**: Raw audio bytes (int16)
- **Output**: JSON with transcribed text

### LLM Service
- **Port**: 8003
- **Technology**: llama-cpp-python
- **Endpoints**:
  - `POST /generate` - Generate response
  - `GET /health` - Health check
- **Features**:
  - Streaming token generation
  - Tool/function calling
  - Agent mode support
- **Input**: JSON with text, tools, stream flag
- **Output**: Streaming tokens or complete response

### TTS Service
- **Port**: 8004
- **Technology**: Piper
- **Endpoints**:
  - `POST /synthesize` - Synthesize complete text
  - `POST /synthesize_stream` - Stream synthesis
  - `GET /health` - Health check
- **Input**: JSON with text or streaming text
- **Output**: WAV audio data

## Streaming Implementation

### LLM → TTS Streaming
1. LLM generates tokens one at a time
2. Tokens sent via Server-Sent Events (SSE)
3. TTS accumulates tokens until sentence boundary
4. TTS synthesizes complete sentences
5. Audio chunks streamed to orchestrator
6. Orchestrator plays audio immediately

This reduces perceived latency from seconds to milliseconds.

## Cancellation Mechanism

The orchestrator uses an `asyncio.Event` to signal cancellation:

1. Wakeword detected during active pipeline
2. `cancel_event.set()` called
3. All pipeline tasks check event periodically
4. Tasks exit gracefully
5. New pipeline starts immediately

## Tool Registry

Tools are registered using decorators:

```python
@registry.register(description="Get current time")
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")
```

The registry:
- Extracts function signatures automatically
- Generates JSON schema for LLM
- Handles async and sync functions
- Executes tools when LLM requests them

## Chat Continuation

After TTS completes:
1. Wait for configurable timeout (default 5s)
2. Monitor audio for speech
3. If detected, record and process
4. Continue conversation without wakeword
5. Recursive continuation support

## Audio Device Auto-Detection

When `AUDIO_INPUT_DEVICE=auto`:
- Uses system default input device
- Falls back to first available device

When specific device ID provided:
- Uses that device directly

List devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

## Environment Variables

All services configured via environment variables:
- Model paths and configurations
- Service URLs (for inter-service communication)
- Audio parameters
- Pipeline behavior
- Performance tuning

See `.env.example` for complete list.

## Build Optimization

### llama-cpp-python
The Dockerfile detects available acceleration:
1. Check for CUDA (NVIDIA GPUs)
2. Check for OpenBLAS (CPU optimization)
3. Fallback to baseline build

This ensures optimal performance on any hardware.

## Volume Management

- `models/`: Persistent model storage
- `confirmations/`: Generated confirmation audio
## Networking

All services on `voice_assistant` bridge network:
- Internal DNS resolution
- Service-to-service communication
- Isolated from host network
- Orchestrator has host audio device access

## Error Handling

Each service:
- Logs errors with context
- Returns appropriate HTTP status codes
- Continues operation on non-fatal errors
- Health endpoints for monitoring

## Performance Considerations

- Wakeword: ~80ms chunks, low latency
- STT: Batch processing, VAD filtering
- LLM: Streaming reduces wait time
- TTS: Sentence-level synthesis
- Audio: Minimal buffering

## Future Enhancements

- Multi-user support
- Voice activity detection in orchestrator
- Model hot-swapping
- Metrics and monitoring
- Web UI for configuration
- Multi-language support
- Custom wakeword training
