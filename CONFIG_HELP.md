Every variable can be set within the `.env` file after copying the example.
Here are some comments to help you understand what each setting does.

```env
APP_ENABLE_EARCONS=true    # audio ack/nack sound cues
APP_WARMUP_ON_INIT=true    # warm the models on startup
APP_CONTINUATION_ENABLED=true    # keep listening for a follow-up after each reply
APP_LOGGING_LEVEL=INFO    # DEBUG, INFO, WARNING, ERROR; DEBUG also prints per-turn latency timings
APP_LOGGING_FORMAT=%(asctime)s %(levelname)s [%(name)s]: %(message)s    # python logging format string
APP_DISABLE_HTTP_LOGGING=true    # disable httpx and similar logs, they clutter up the output

LLM_EXECUTABLE_PATH=llama_cpp_bin/llama-server.exe
LLM_BIND_HOST=127.0.0.1    # address the server binds to; 0.0.0.0 to expose it (docker/LAN), 127.0.0.1 for local only
LLM_SERVER_PORT=43001
LLM_MODEL_PATH=./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
LLM_CONTEXT_WINDOW=8192    # context windows, set lower if history is disabled
LLM_N_GPU_LAYERS=0    # layers offloaded to the GPU; 0 is CPU-only, 99 is all (needs a CUDA build)
LLM_FLASH_ATTN=false    # flash attention, faster inference on GPU builds

LLM_SERVER_HOST=127.0.0.1    # address the client connects to; the server host, or its service name under docker
LLM_SYSTEM_INSTRUCTIONS=    # base system prompt; memory guidance is appended automatically when memory_tools is on
LLM_TEMPERATURE=0.6    # sampling randomness, 0 is deterministic, higher is more creative
LLM_MAX_ITERATIONS=10    # max tool-call rounds per turn
LLM_MODE=agent    # "agent" (tool-calling) or "chatbot"
LLM_RESPONSE_TIMEOUT=60.0    # max seconds to wait for a response before aborting the turn. mostly for when the llm gets stuck
LLM_HISTORY_ENABLED=true    # Whether to enable chat history (current session only)
LLM_HISTORY_MAX_TURNS=3    # How much messages (ai-user pairs) aback should the history store
LLM_HISTORY_IDLE_TIMEOUT_S=120.0    # After how much seconds should the earliest message be wiped from memory

TOOLS_ENABLED_MODULES=math_tools,datetime_tools,random_tools,memory_tools    # which tool modules to load
TOOLS_MEMORY_SYSTEM_INSTRUCTIONS= # system instructions to append when memory is on, to guide the agent
TOOLS_MEMORY_DB_PATH=./assets/memory.db    # SQLite file backing persistent memory

OWW_MODEL_PATHS=./models/hey_jarvis_v0.1.onnx    # wake word .onnx model
OWW_FRAMEWORK=onnx    # wakeword inference backend, onnx or tflite
OWW_THRESHOLD=0.5    # detection sensitivity, 0-1
OWW_CHUNK_SIZE=1280    # samples per prediction, 1280 is 80ms at 16kHz
OWW_SAMPLE_RATE=16000    # mic capture rate; openwakeword models expect 16kHz, leave as is
OWW_CHANNELS=1    # mono
OWW_DTYPE=int16    # audio sample format, only int16 is supported

STT_BIND_HOST=127.0.0.1    # address the server binds to (0.0.0.0 to expose it)
STT_SERVER_PORT=43002
STT_MODEL_PATH=./models/whisper-base.pt
STT_LANGUAGE=auto    # set to auto or specific iso code
STT_MIN_CHUNK_SIZE=1    # seconds of audio processed per step, lower is snappier but heavier
STT_WARMUP_AUDIO_PATH=./assets/stt_warmup.wav    # clip used to warm the model when warmup is on

STT_SERVER_HOST=127.0.0.1    # address the client connects to (server host or docker service name)
STT_SAMPLE_RATE=16000    # mic capture rate; Whisper expects 16kHz, leave as is
STT_CHANNELS=1    # mono
STT_DTYPE=int16    # audio sample format, only int16 is supported
STT_BLOCK_SIZE=4000    # mic block size in samples, 4000 is 250ms at 16kHz
STT_RESPONSE_TIMEOUT=5.0    # default window, seconds
STT_CONTINUATION_TIMEOUT=3.0    # follow-up window, seconds

TTS_BIND_HOST=127.0.0.1    # address the server binds to (0.0.0.0 to expose it)
TTS_SERVER_PORT=43003
TTS_MODEL_PATH=./models/en_US-lessac-medium.onnx

TTS_SERVER_HOST=127.0.0.1    # address the client connects to (server host or docker service name)
TTS_LENGTH_SCALE=1.5    # 1.0 is normal speech, higher is slower, lower is faster
TTS_NOISE_SCALE=1.0    # voice variability, higher is more expressive but less stable
TTS_NOISE_W_SCALE=0.5    # speaking-cadence variability (phoneme duration jitter)
TTS_CHUNK_MODE=hybrid    # how reply text is split for streaming: hybrid, sentence, chars, or words
TTS_CHUNK_SIZE=3    # units per chunk for chars/words modes
TTS_FIRST_CHUNK_MAX_WORDS=5    # hybrid mode: cap on the opening chunk so audio starts sooner
TTS_PLAYBACK_CHUNK_MS=20    # speaker is fed audio in slices this small; smaller reacts to barge-in faster
```
