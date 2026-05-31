from dataclasses import dataclass, field
from numpy import int16
from typing import Literal


# Per-stage latency instrumentation (P0-1). When True, the pipeline prints the
# inter-stage durations of each turn (endpoint -> first LLM token -> first synth
# chunk -> first audio out). Set to False to silence the timing output.
DEBUG_LATENCY: bool = True

# Confirmation earcons (P1-5): short tones after the wakeword and after speech
# endpointing, masking the LLM time-to-first-token silence. Set False to disable.
ENABLE_EARCONS: bool = True

# Block client init until a warmup request primes each server, so the first real
# turn isn't a cold prefill. Measured cold-start cost: agent TTFT 2586ms cold vs
# ~1100ms warm. Resilient to a server being down (warmup is skipped with a note).
WARMUP_ON_INIT: bool = True


@dataclass
class LLMClientConfig:
    # llama-server is launched with a fixed model (modules/server/llm_server.py)
    # and IGNORES the `model` field in requests, so these are informational only
    # and must match whatever the server actually serves. Currently Qwen2.5-3B.
    agent_model_path: str = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    chatbot_model_path: str = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    system_instructions: str = (
        "Respond in plain spoken prose only - no markdown, bullet points, headers, bold, "
        "emojis, or special characters. Keep responses short: 10-30 seconds of speaking "
        "time, scaled to task complexity. Saying I don't know is rewarded, but rambling is not. "
    )
    temperature: float = 0.6
    max_iterations: int = 10
    mode: Literal["agent", "chatbot"] = "agent"
    # Overall budget for one LLM turn (including any agentic tool iterations).
    # A stalled or hung stream is cancelled past this so it can't wedge the turn.
    response_timeout: float = 60.0


@dataclass
class OWWClientConfig:
    model_paths: list[str] = field(
        default_factory=lambda: [
            "C:/Users/user/Documents/Projects/python/ai-voice-stack/models/hey_jarvis_v0.1.onnx",
        ]
    )
    framework: str = "onnx"
    chunk_size: int = 1280  # 80ms at 16kHz
    channels: int = 1
    dtype: type = int16
    sample_rate: int = 16000
    threshold: float = 0.5


@dataclass
class STTClientConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 43002
    sample_rate: int = 16000
    channels: int = 1
    dtype: type = int16
    block_size: int = 4000  # 250ms chunks
    response_timeout: float = 5.0  # max seconds to wait for speech before aborting


@dataclass
class TTSClientConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 43003
    length_scale: float = 1.5
    noise_scale: float = 1.0
    noise_w_scale: float = 0.5
    chunk_mode: Literal["hybrid", "sentence", "chars", "words"] = "hybrid"
    chunk_size: int = 3
    # P1-1 hybrid first-chunk: emit the very first audio as soon as the first
    # clause boundary OR this many words is reached (whichever comes first), then
    # revert to sentence granularity. Cuts time-to-first-audio without globally
    # degrading Piper prosody. Only affects chunk_mode="hybrid".
    first_chunk_max_words: int = 5
    # Granularity of blocking writes on the playback thread, in milliseconds. This
    # is the barge-in latency knob: smaller = faster interrupt, more write calls.
    playback_chunk_ms: int = 20
