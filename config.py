import os
from dataclasses import dataclass, field
from numpy import int16
from typing import Literal
from logging import INFO, DEBUG, WARNING, ERROR, CRITICAL

# llama-server binary name is OS-specific; the provisioner builds it into llama_cpp_bin/.
_LLAMA_SERVER_BIN = "llama-server.exe" if os.name == "nt" else "llama-server"

@dataclass
class AppConfig:
    enable_earcons: bool = True
    warmup_on_init: bool = True
    continuation_enabled: bool = True
    barge_in_enabled: bool = True
    logging_format: str = "%(asctime)s %(levelname)s [%(name)s]: %(message)s"
    logging_level: int = DEBUG
    disable_http_logging: bool = True # set to True to reduce noise from httpx and openai client logs


@dataclass
class LLMServerConfig:
    executable_path: str = f"llama_cpp_bin/{_LLAMA_SERVER_BIN}"
    model_path: str = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    server_host: str = "127.0.0.1"
    server_port: int = 43001
    context_window: int = 8192
    gpu_layers: int = 0       # -ngl passed to llama-server. 0 = CPU; set 99 to offload all layers (CUDA/Metal builds).
    flash_attn: bool = False  # adds "-fa on" when True. Recommended alongside GPU offload.


@dataclass
class LLMClientConfig:
    model_path: str = LLMServerConfig.model_path
    server_host: str = LLMServerConfig.server_host
    server_port: int = LLMServerConfig.server_port
    system_instructions: str = (
        "Respond in plain spoken prose only - no markdown, bullet points, headers, bold, "
        "emojis, or special characters. Keep responses up to 60 seconds of speaking "
        "time, scaled to task complexity. "
    )
    temperature: float = 0.6
    max_iterations: int = 10
    mode: Literal["agent", "chatbot"] = "agent"
    response_timeout: float = 60.0 # max seconds to wait for a response before aborting the turn
    history_enabled: bool = True
    history_max_turns: int = 3
    history_idle_timeout_s: float = 120.0


@dataclass
class ToolsConfig:
    enabled_tool_modules: list[str] = field(
        default_factory=lambda: ["math_tools", "datetime_tools", "random_tools", "memory_tools"]
    )
    tool_timeout: float = 15.0 # max seconds to wait for a tool response before aborting and returning an error to the model
    # Appended to LLMClientConfig.system_instructions only when "memory_tools" is enabled above.
    memory_system_instructions: str = (
        "You have persistent memory tools. Use store_memory to save facts the user shares "
        "about themselves (name, preferences, recurring topics, important dates), proactively "
        "and without being asked. Whenever the user asks about themselves or their preferences, "
        "or refers to anything they may have told you before, call recall_memory first and "
        "answer from what it returns. Never say you don't know a personal detail without "
        "searching memory first."
    )
    memory_db_path: str = "./assets/memory.db"


@dataclass
class OWWClientConfig:
    model_paths: list[str] = field(
        default_factory=lambda: [
            "./models/hey_jarvis_v0.1.onnx",
        ]
    )
    framework: str = "onnx"
    chunk_size: int = 1280  # 80ms at 16kHz
    channels: int = 1
    dtype: type = int16
    sample_rate: int = 16000
    threshold: float = 0.5


@dataclass
class STTServerConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 43002
    model_path: str = "./models/whisper-base.pt"
    language: str = "auto"
    min_chunk_size: int = 1 # process every ~1s of audio
    warmup_audio_path: str = "./assets/stt_warmup.wav" # if warmup is enabled


@dataclass
class STTClientConfig:
    server_host: str = STTServerConfig.server_host
    server_port: int = STTServerConfig.server_port
    sample_rate: int = 16000
    channels: int = 1
    dtype: type = int16
    block_size: int = 4000  # 250ms chunks
    response_timeout: float = 5.0
    continuation_timeout: float = 3.0


@dataclass
class TTSServerConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 43003
    model_path: str = "./models/en_US-lessac-medium.onnx"


@dataclass
class TTSClientConfig:
    server_host: str = TTSServerConfig.server_host
    server_port: int = TTSServerConfig.server_port
    length_scale: float = 1.5
    noise_scale: float = 1.0
    noise_w_scale: float = 0.5
    chunk_mode: Literal["hybrid", "sentence", "chars", "words"] = "hybrid"
    chunk_size: int = 3
    first_chunk_max_words: int = 5
    playback_chunk_ms: int = 20
