import os
from typing import List, Optional


class Config:
    # Service URLs
    WAKEWORD_SERVICE_URL: str = os.getenv("WAKEWORD_SERVICE_URL", "http://wakeword:8001")
    STT_SERVICE_URL: str = os.getenv("STT_SERVICE_URL", "http://stt:8002")
    LLM_SERVICE_URL: str = os.getenv("LLM_SERVICE_URL", "http://llm:8003")
    TTS_SERVICE_URL: str = os.getenv("TTS_SERVICE_URL", "http://tts:8004")

    # Audio
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_CHUNK_SIZE: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1280"))
    AUDIO_VOLUME: float = float(os.getenv("AUDIO_VOLUME", "1.0"))

    # Device indices: -1 means auto-detect (None for sounddevice)
    @staticmethod
    def _parse_device(env_key: str) -> Optional[int]:
        raw = os.getenv(env_key, "-1")
        try:
            val = int(raw)
            return None if val < 0 else val
        except (ValueError, TypeError):
            return None

    AUDIO_INPUT_DEVICE: Optional[int] = _parse_device.__func__("AUDIO_INPUT_DEVICE")
    AUDIO_OUTPUT_DEVICE: Optional[int] = _parse_device.__func__("AUDIO_OUTPUT_DEVICE")

    # Recording / silence detection
    SILENCE_DURATION: float = float(os.getenv("SILENCE_DURATION", "1.5"))
    MAX_RECORD_SECONDS: float = float(os.getenv("MAX_RECORD_SECONDS", "15.0"))
    SILENCE_THRESHOLD: float = float(os.getenv("SILENCE_THRESHOLD", "0.005"))

    # Agent
    AGENT_ENABLED: bool = os.getenv("AGENT_ENABLED", "false").lower() == "true"

    # Chat continuation
    CHAT_CONTINUATION_ENABLED: bool = os.getenv("CHAT_CONTINUATION_ENABLED", "true").lower() == "true"
    CHAT_CONTINUATION_TIMEOUT: float = float(os.getenv("CHAT_CONTINUATION_TIMEOUT", "5.0"))

    # Confirmations
    CONFIRMATIONS_PATH: str = "/app/assets/confirmations"

    @property
    def confirmation_phrases(self) -> List[str]:
        raw = os.getenv("CONFIRMATION_PHRASES", "mhm,listening,go ahead")
        return [p.strip() for p in raw.split(",") if p.strip()]
