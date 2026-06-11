# ai-voice-stack

[![unit](https://github.com/dh3b/ai-voice-stack/actions/workflows/unit.yml/badge.svg)](https://github.com/dh3b/ai-voice-stack/actions/workflows/unit.yml)
[![smoke](https://github.com/dh3b/ai-voice-stack/actions/workflows/smoke.yml/badge.svg)](https://github.com/dh3b/ai-voice-stack/actions/workflows/smoke.yml)

## What it is

A local voice assistant. It listens for a wake word, transcribes what you say, runs it through a language model that can call tools, and speaks the reply. Wake word, speech-to-text, LLM, and text-to-speech all run on your own machine - no cloud, nothing leaves the device - and it is built to run on edge hardware such as Jetson boards.

## How it works

The stack is four components, each a client talking to a local server, wired together by `pipeline.py`.

![ai-voice-stack architecture](demo/aistack_diagram1.svg)

A turn runs left to right: the wake word listener (openwakeword) triggers on loaded wakeword models, speech-to-text (Whisper, via SimulStreaming) produces a transcript, the LLM (llama.cpp) generates a reply and may call tools while doing so, and text-to-speech (Piper) streams the audio back. Saying the wake word again interrupts a reply in progress (mutable via config), and after each reply it keeps listening briefly so you can follow up without the wake word (also mutable via config). Tools, the backend servers, and the utilities (audio cues, latency timing) sit around that main flowchart.

## Compatibility

- Windows and Linux, on x86_64 and arm64. Targets edge devices such as NVIDIA Jetson; also runs on RPi 4/5 (CPU only, slow).
- Python 3.10+ (provided/managed by `uv` - you don't need it preinstalled).
- A microphone and a speaker.
- A CUDA-capable GPU is recommended; CPU works but the LLM and Whisper will be slow.

There is **no manual build step**. The installer detects your machine and builds
`llama-server`, fetches the models, installs the right `torch`, and clones the STT
backend for you - see below.

> [!TIP]
> If you encounter any installation problems, you can always build the dependencies yourself, and link the paths in `config.py`, also view [troubleshooting](#trouble).

## <a name="installation"></a>Installation

Two prerequisites - **`uv`** (Python/venv/deps) and **`task`** ([go-task](https://taskfile.dev), the command runner):

| | uv | task |
|---|---|---|
| **Windows** | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `winget install Task.Task` |
| **Linux** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `sudo apt install task` |

Then:

```
git clone https://github.com/dh3b/ai-voice-stack
cd ai-voice-stack
task setup     # detects the machine, installs deps + torch, builds llama.cpp, fetches models
task run       # launches wakeword + STT + LLM + TTS
```

If you want to use the example models, **you need to have git lfs installed**. If the models still don't work after cloning, make sure to run `git lfs pull` inside the repo.

`task setup` can fail when a step genuinely can't
self-install (typically a system C++ compiler, or a full CUDA toolkit on Windows),
it stops with the exact command to run - fix it and re-run `task setup` to continue.
Run `task doctor` any time to see the detected profile and what's still missing.

Everything lands **inside the repo**, never your global environment: the venv in
`.venv/`, the LLM server in `llama_cpp_bin/`, models in `models/`, the STT backend
in `simulstreaming_lib/`.

### Models

Set the model paths in `config.py`. Examples are provided within the repo, feel free to switch them out.

**To use a different model:** point its path in `config.py` at your file and drop the
file in `models/` yourself. The installer never overwrites a file that already exists,
so it won't clobber yours. Run `task doctor` once you're done, to confirm everything was done right.

| Model | File | Used for |
|---|---|---|
| Qwen2.5-3B-Instruct (Q4_K_M GGUF) | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | the LLM - replies and tool calls |
| Whisper base | `whisper-base.pt` | speech-to-text |
| Piper en_US-lessac-medium | `en_US-lessac-medium.onnx` (+ `.onnx.json`) | text-to-speech |
| openwakeword "hey jarvis" | `hey_jarvis_v0.1.onnx` | wake word |

### Granular & advanced

Each phase is a task you can run on its own, and re-running is cheap:

```
task doctor          # report machine profile + gaps
task toolchain       # cmake / ninja / compiler / CUDA
task torch           # torch+torchaudio for the detected accelerator
task stt             # clone SimulStreaming + install its requirements
task llama           # build llama.cpp's llama-server
task llama -- --force --jobs 4   # flags pass through after `--`
```

- **GPU offload:** after a CUDA `llama` build, set `gpu_layers=99` (and
  `flash_attn=True`) in `LLMServerConfig` in `config.py` to run the LLM on the GPU.
- **Pinned versions:** the llama.cpp ref is `LLAMA_REF` in `installer/build_llama.py`;
  the SimulStreaming ref is `SIMUL_REF` in `installer/setup_stt.py`.

### <a name="trouble"></a>Troubleshooting

- **No C++ compiler** - Windows: install "Build Tools for Visual Studio 2022" with the
  *Desktop development with C++* workload; Linux: `sudo apt-get install -y build-essential`.
  Then re-run `task setup`.
- **CUDA build can't find `nvcc`** - install a CUDA toolkit matching your driver, or let
  the installer try the experimental pip CUDA-wheel route; re-run `task setup`.
- **Jetson** - building on-device is expected; watch power/thermal with `tegrastats`,
  lower `task llama -- --jobs N` or `nvpmodel` if the board resets. The CUDA/Jetson torch
  stack needs `numpy<2`; the installer pins it, so prefer `task setup`/`task torch` over a
  bare `uv sync` (which would restore the locked `numpy 2`).

## Configuration

Everything lives in `config.py` as per-component dataclasses; edit the defaults there. The settings you are most likely to change:

```python
AppConfig.enable_earcons         # audio ack/nack sound cues
AppConfig.continuation_enabled   # keep listening for a follow-up after each reply
AppConfig.warmup_on_init         # warm the models on startup

LLMClientConfig.mode             # "agent" (tool-calling) or "chatbot"
LLMClientConfig.system_instructions
LLMClientConfig.temperature
LLMClientConfig.max_iterations   # max tool-call rounds per turn
LLMClientConfig.response_timeout # max seconds to wait for a response before aborting the turn
LLMClientConfig.history_enabled  # Whether to enable chat history (current session only)
LLMClientConfig.history_max_turns # How much messages (ai-user pairs) aback should the history store
LLMClientConfig.history_idle_timeout_s # After how much seconds should the earliest message be wiped from memory

ToolsConfig.enabled_tool_modules # which tool modules to load
ToolsConfig.memory_db_path       # SQLite file backing persistent memory

OWWClientConfig.model_paths      # wake word .onnx model(s)
OWWClientConfig.threshold        # detection sensitivity, 0-1

STTServerConfig.model_path       # Whisper checkpoint
STTServerConfig.language         # "auto" or an ISO code
STTServerConfig.response_timeout # default window, seconds
STTClientConfig.continuation_timeout   # follow-up window, seconds

TTSClientConfig.length_scale     # speech rate (higher is slower)
```

Model paths and server addresses (LLM `:43001`, STT `:43002`, TTS `:43003`) are defined here too.

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
