"""Application configuration.

Every setting is an environment variable. On import, loads (in order) the committed
`.env.example` baseline and then a local `.env` that overrides it.
"""

import os
from logging import getLevelName
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from dotenv import load_dotenv

    _ROOT = Path(__file__).resolve().parent
    load_dotenv(_ROOT / ".env.example") 
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass

# llama-server binary name is OS-specific; the provisioner builds it into llama_cpp_bin/.
_LLAMA_SERVER_BIN = "llama-server.exe" if os.name == "nt" else "llama-server"


def _to_level(value: object) -> int:
    """Accept a level name ("INFO") or number ("20")."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else getLevelName(text.upper())


def _to_list(value: object) -> list[str]:
    """Parse a comma-separated env value into a list (already-a-list passes through)."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value  # type: ignore[return-value]


LogLevel = Annotated[int, BeforeValidator(_to_level)]
CsvList = Annotated[list[str], BeforeValidator(_to_list)]


_ENV = SettingsConfigDict(
    extra="ignore",
    enable_decoding=False,
    protected_namespaces=(),
    populate_by_name=True,
)


class AppConfig(BaseSettings):
    model_config = _ENV
    enable_earcons: bool = Field(alias="APP_ENABLE_EARCONS")
    warmup_on_init: bool = Field(alias="APP_WARMUP_ON_INIT")
    continuation_enabled: bool = Field(alias="APP_CONTINUATION_ENABLED")
    barge_in_enabled: bool = Field(alias="APP_BARGE_IN_ENABLED")
    logging_format: str = Field(alias="APP_LOGGING_FORMAT")
    logging_level: LogLevel = Field(alias="APP_LOGGING_LEVEL")
    disable_http_logging: bool = Field(alias="APP_DISABLE_HTTP_LOGGING")


class LLMServerConfig(BaseSettings):
    model_config = _ENV
    # OS-specific binary name, so this one keeps an auto-detected default; the env var
    # (commented out in .env.example) overrides it if you point elsewhere.
    executable_path: str = Field(
        default_factory=lambda: f"llama_cpp_bin/{_LLAMA_SERVER_BIN}",
        alias="LLM_EXECUTABLE_PATH",
    )
    model_path: str = Field(alias="LLM_MODEL_PATH")
    server_host: str = Field(alias="LLM_BIND_HOST")
    server_port: int = Field(alias="LLM_SERVER_PORT")
    context_window: int = Field(alias="LLM_CONTEXT_WINDOW")
    gpu_layers: int = Field(alias="LLM_GPU_LAYERS")
    flash_attn: bool = Field(alias="LLM_FLASH_ATTN")


class LLMClientConfig(BaseSettings):
    model_config = _ENV
    model_path: str = Field(alias="LLM_MODEL_PATH")
    server_host: str = Field(alias="LLM_SERVER_HOST")
    server_port: int = Field(alias="LLM_SERVER_PORT")
    system_instructions: str = Field(alias="LLM_SYSTEM_INSTRUCTIONS")
    temperature: float = Field(alias="LLM_TEMPERATURE")
    max_iterations: int = Field(alias="LLM_MAX_ITERATIONS")
    mode: Literal["agent", "chatbot"] = Field(alias="LLM_MODE")
    response_timeout: float = Field(alias="LLM_RESPONSE_TIMEOUT")
    history_enabled: bool = Field(alias="LLM_HISTORY_ENABLED")
    history_max_turns: int = Field(alias="LLM_HISTORY_MAX_TURNS")
    history_idle_timeout_s: float = Field(alias="LLM_HISTORY_IDLE_TIMEOUT_S")


class ToolsConfig(BaseSettings):
    model_config = _ENV
    enabled_tool_modules: CsvList = Field(alias="TOOLS_ENABLED_MODULES")
    tool_timeout: float = Field(alias="TOOLS_TOOL_TIMEOUT")
    memory_system_instructions: str = Field(alias="TOOLS_MEMORY_SYSTEM_INSTRUCTIONS")
    memory_db_path: str = Field(alias="TOOLS_MEMORY_DB_PATH")


class OWWClientConfig(BaseSettings):
    model_config = _ENV
    model_paths: CsvList = Field(alias="OWW_MODEL_PATHS")
    framework: str = Field(alias="OWW_FRAMEWORK")
    chunk_size: int = Field(alias="OWW_CHUNK_SIZE")
    channels: int = Field(alias="OWW_CHANNELS")
    dtype: Literal["int16"] = Field(alias="OWW_DTYPE")  # sounddevice accepts the string form
    sample_rate: int = Field(alias="OWW_SAMPLE_RATE")
    threshold: float = Field(alias="OWW_THRESHOLD")


class STTServerConfig(BaseSettings):
    model_config = _ENV
    server_host: str = Field(alias="STT_BIND_HOST")
    server_port: int = Field(alias="STT_SERVER_PORT")
    model_path: str = Field(alias="STT_MODEL_PATH")
    language: str = Field(alias="STT_LANGUAGE")
    min_chunk_size: int = Field(alias="STT_MIN_CHUNK_SIZE")
    warmup_audio_path: str = Field(alias="STT_WARMUP_AUDIO_PATH")


class STTClientConfig(BaseSettings):
    model_config = _ENV
    server_host: str = Field(alias="STT_SERVER_HOST")
    server_port: int = Field(alias="STT_SERVER_PORT")
    sample_rate: int = Field(alias="STT_SAMPLE_RATE")
    channels: int = Field(alias="STT_CHANNELS")
    dtype: Literal["int16"] = Field(alias="STT_DTYPE")  # sounddevice accepts the string form
    block_size: int = Field(alias="STT_BLOCK_SIZE")
    response_timeout: float = Field(alias="STT_RESPONSE_TIMEOUT")
    continuation_timeout: float = Field(alias="STT_CONTINUATION_TIMEOUT")


class TTSServerConfig(BaseSettings):
    model_config = _ENV
    server_host: str = Field(alias="TTS_BIND_HOST")
    server_port: int = Field(alias="TTS_SERVER_PORT")
    model_path: str = Field(alias="TTS_MODEL_PATH")


class TTSClientConfig(BaseSettings):
    model_config = _ENV
    server_host: str = Field(alias="TTS_SERVER_HOST")
    server_port: int = Field(alias="TTS_SERVER_PORT")
    length_scale: float = Field(alias="TTS_LENGTH_SCALE")
    noise_scale: float = Field(alias="TTS_NOISE_SCALE")
    noise_w_scale: float = Field(alias="TTS_NOISE_W_SCALE")
    chunk_mode: Literal["hybrid", "sentence", "chars", "words"] = Field(alias="TTS_CHUNK_MODE")
    chunk_size: int = Field(alias="TTS_CHUNK_SIZE")
    first_chunk_max_words: int = Field(alias="TTS_FIRST_CHUNK_MAX_WORDS")
    playback_chunk_ms: int = Field(alias="TTS_PLAYBACK_CHUNK_MS")
