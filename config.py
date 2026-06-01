from dataclasses import dataclass, field
from numpy import int16
from typing import Literal
from logging import INFO, DEBUG, WARNING, ERROR, CRITICAL

@dataclass
class AppConfig:
    enable_earcons: bool = True
    warmup_on_init: bool = True
    logging_format: str = "%(asctime)s %(levelname)s [%(name)s]: %(message)s"
    logging_level: int = INFO
    disable_http_logging: bool = True # set to True to reduce noise from httpx and openai client logs


@dataclass
class LLMServerConfig:
    model_path: str = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    server_host: str = "127.0.0.1"
    server_port: int = 43001
    context_window: int = 8192


@dataclass
class LLMClientConfig:
    model_path: str = LLMServerConfig.model_path
    server_host: str = LLMServerConfig.server_host
    server_port: int = LLMServerConfig.server_port
    system_instructions: str = (
        "Respond in plain spoken prose only - no markdown, bullet points, headers, bold, "
        "emojis, or special characters. Keep responses short: 10-30 seconds of speaking "
        "time, scaled to task complexity. Saying I don't know is rewarded, but rambling is not. "
    )
    temperature: float = 0.6
    max_iterations: int = 10
    mode: Literal["agent", "chatbot"] = "agent"
    response_timeout: float = 60.0 # max seconds to wait for a response before aborting the turn
    history_enabled: bool = True
    history_max_turns: int = 3
    history_idle_timeout_s: float = 120.0


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
class STTServerConfig:
    server_host: str = "127.0.0.1"
    server_port: int = 43002
    model_path: str = "C:/Users/user/Documents/Projects/python/ai-voice-stack/models/whisper-base.pt"
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
    response_timeout: float = 5.0  # max seconds to wait for speech before aborting


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
