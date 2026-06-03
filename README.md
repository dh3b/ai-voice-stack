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

4. Models. Download into `models/`:

   | Model example | File(s) example | Used for |
   |---|---|---|
   | Qwen2.5-3B-Instruct (Q4_K_M GGUF) | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | the LLM - replies and tool calls |
   | Whisper base | `whisper-base.pt` | speech-to-text |
   | Piper en_US-lessac-medium | `en_US-lessac-medium.onnx` (+ `.onnx.json`) | text-to-speech |
   | openwakeword "hey jarvis" | `hey_jarvis_v0.1.onnx` | wake word |

   Any GGUF chat model with tool-calling can replace Qwen, set its path in config.

### Run

Start the three servers (each supervises its backend), then the assistant:

```
python -m modules.server.llm_server
python -m modules.server.stt_server
python -m modules.server.tts_server
python pipeline.py
```

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
