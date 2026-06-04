# ai-voice-stack

## What it is

A local voice assistant. It listens for a wake word, transcribes what you say, runs it through a language model that can call tools, and speaks the reply. Wake word, speech-to-text, LLM, and text-to-speech all run on your own machine - no cloud, nothing leaves the device - and it is built to run on edge hardware such as Jetson boards.

## How it works

The stack is four components, each a client talking to a local server, wired together by `pipeline.py`.

![ai-voice-stack architecture](demo/aistack_diagram1.svg)

A turn runs left to right: the wake word listener (openwakeword) triggers on loaded wakeword models, speech-to-text (Whisper, via SimulStreaming) produces a transcript, the LLM (llama.cpp) generates a reply and may call tools while doing so, and text-to-speech (Piper) streams the audio back. Saying the wake word again interrupts a reply in progress (mutable via config), and after each reply it keeps listening briefly so you can follow up without the wake word (also mutable via config). Tools, the backend servers, and the utilities (audio cues, latency timing) sit around that main flowchart.

## Compatibility

- Linux and Windows. Targets edge devices such as NVIDIA Jetson. Also tested specifically on RPI4/5, but mind the speed is not that great.
- Python 3.10+.
- A microphone and a speaker.
- A CUDA-capable GPU is recommended; CPU works but the LLM and Whisper will be slow.
- A llama.cpp build that provides `llama-server` (OpenAI-compatible API). The STT and ML dependencies (torch, etc.) come from the SimulStreaming repo below, not from this one.

The LLM server invokes `llama_cpp_bin/llama-server.exe`; on Linux use your platform's binary name and change it in config (see [installation](#installation)).

## <a name="installation"></a>Installation
> [!WARNING]
> Although a manual installation is still maintaned possible, I highly recommend a much simpler installation with [docker](#docker)

1. Clone and install the app layer:

   ```
   git clone https://github.com/dh3b/ai-voice-stack
   cd ai-voice-stack
   python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -e .
   pip install -r requirements.txt
   ```

2. Speech-to-text engine. Clone one of these into `simulstreaming_lib/` and install its requirements - this supplies torch, Whisper, and the streaming server the STT server launches:

   ```
   git clone https://github.com/ufal/SimulStreaming simulstreaming_lib
   # or the trimmed fork:
   git clone https://github.com/dh3b/SimulStreaming-lite simulstreaming_lib
   ```
    And make sure to download needed requirements for the `simulstreaming_lib` with:
   ```
   pip install -r requirements_whisper.txt
   ```

3. llama.cpp. Build `llama-server` from source or download a prebuilt binary, and place it in `llama_cpp_bin/`. (`modules/server/llm_server.py` calls `llama_cpp_bin/llama-server.exe` - adjust the name for your platform). For more installation instructions just visit the [llama.cpp](https://github.com/ggml-org/llama.cpp) repo.

4. Models. Download into `models/`, see [models reference](#mdl_ref)

### Run

Start the three servers (each supervises its backend), then the assistant:

```
python -m modules.server.llm_server
python -m modules.server.stt_server
python -m modules.server.tts_server
python pipeline.py
```

## <a name="docker"></a>Running with Docker 

The whole stack runs as four containers wired over a private Compose network:

| Service | Image builds | Port | Notes |
|---|---|---|---|
| `llm_server` | llama.cpp `llama-server` | `:43001` |  the llama.cpp build |
| `stt_server` | torch + SimulStreaming-lite | `:43002` | the Whisper/torch stack |
| `tts_server` | piper-tts | `:43003` | |
| `main` | the pipeline | — | wakeword + orchestration; the only container that uses the mic/speaker |

Models and `assets/` are **bind-mounted**, not baked into the images, so images stay small and you can swap models without rebuilding.

### Prerequisites
- Docker Engine + Compose v2.
- The models in `models/` and `assets/`. See [models reference](#mdl_ref)]
- A viable config located in `.env`, following the `.env.example` file

### Quick start (CPU, any platform)

```
docker compose up --build
```

The first build compiles llama.cpp and installs torch, so it takes a while; later rebuilds are cached. `main` waits (via healthchecks) until the three servers are ready.

### NVIDIA GPU (x86 + discrete GPU)

Needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Layer the GPU override:

```
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

This swaps `llm_server` to the CUDA build (offloading all layers, `LLM_N_GPU_LAYERS=99`), builds the STT torch wheels from the CUDA index, and reserves the GPU. Set the GPU architecture with the `CUDA_ARCH` build arg in `docker-compose.cuda.yml` (`89` Ada, `86` Ampere, `80` A100, `75` Turing).

### Jetson (Orin)

Jetson needs JetPack-specific CUDA wheels and bases, so override the build args (in `docker-compose.cuda.yml` or on the CLI):

- `llm_server` -> `docker/llm/Dockerfile.cuda` with `CUDA_ARCH=87`, and L4T CUDA bases via `CUDA_DEVEL_IMAGE` / `CUDA_RUNTIME_IMAGE` set to your JetPack's `l4t-cuda` devel/runtime images.
- `stt_server` -> `TORCH_INDEX_URL` pointing at NVIDIA's JetPack PyTorch index (cu126 for JetPack 6.x); keep `numpy<2` (already pinned in the image).


### Raspberry Pi / other ARM64 (CPU)

Use the base (CPU) compose, but build the STT image's torch from the default PyPI aarch64 wheels by passing an empty index:

```
docker compose build --build-arg TORCH_INDEX_URL= stt_server
docker compose up
```

Expect modest speeds, as with the native Pi setup.

### Audio on Windows / macOS

`main` does live mic/speaker I/O via `/dev/snd`, which **only exists on Linux**; Docker Desktop can't pass the host audio device through on Windows/macOS. So on those hosts, run the **servers** in Docker and the **pipeline** on the host:

```
docker compose up -d llm_server stt_server tts_server
pip install -e . && pip install -r requirements.txt
python pipeline.py                                      
```

The servers publish to `127.0.0.1:43001-43003` and `pipeline.py`'s defaults already point at `127.0.0.1`, so no extra config is needed.

## <a name="mdl_ref"></a>Model reference

   | Model example | File(s) example | Used for |
   |---|---|---|
   | Qwen2.5-3B-Instruct (Q4_K_M GGUF) | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | the LLM - replies and tool calls |
   | Whisper base | `whisper-base.pt` | speech-to-text |
   | Piper en_US-lessac-medium | `en_US-lessac-medium.onnx` (+ `.onnx.json`) | text-to-speech |
   | openwakeword "hey jarvis" | `hey_jarvis_v0.1.onnx` | wake word |

   Any GGUF chat model with tool-calling can replace Qwen, set its path in config. Everything else that's not a model is located in `assets/`. `stt_warmup.wav` is already provided, but feel free to switch up the file.

## Configuration

Everything lives in `.env`, edit the defaults there (copy the `.env.example` first). This section would be too large to explain every detail of config, so please consult the [config help](CONFIG_HELP.md) doc.

## Adding a tool

Tools are Python functions registered with a schema and exposed to the LLM in agent mode. The functionality is identical to [OpenAI agent tools](https://developers.openai.com/api/docs/guides/function-calling). Drop a module in `modules/tools/` and register a handler:

```python
# modules/tools/weather_tools.py
from modules.tools.registry import registry

@registry.register({
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city. Use when the user asks about the weather.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
})
def get_weather(city: str) -> str:
    ...
    return "Clear, 18C."   # the returned string is fed back to the model
```

Then add the module name to `ToolsConfig.enabled_tool_modules` (here, `"weather_tools"`). The handler's return value goes back to the model as the tool result; raised exceptions are caught and returned as an error string, so the turn does not crash.

> [!NOTE]
> Feel free to fork the repository and add some tools yourself. I'd be glad to see them.

## License
MIT

## Credits

Huge thanks to ÚFAL, for releasing the opensource [SimulStreaming](https://github.com/ufal/SimulStreaming) repository. This project wouldn't be half as efficient without them.
