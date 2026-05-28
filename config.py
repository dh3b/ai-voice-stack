from dataclasses import dataclass, field
from numpy import int16
from typing import Literal


@dataclass
class LLMClientConfig:
    agent_model_path: str = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
    chatbot_model_path: str = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
    system_instructions: str = (
        "Respond in plain spoken prose only - no markdown, bullet points, headers, bold, "
        "emojis, or special characters. Keep responses short: 10-30 seconds of speaking "
        "time, scaled to task complexity. Saying I don't know is rewarded, but rambling is not. "
    )
    temperature: float = 0.6
    max_iterations: int = 10
    mode: Literal["agent", "chatbot"] = "agent"


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
