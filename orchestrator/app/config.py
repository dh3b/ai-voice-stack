import os
from typing import Optional

class Config:
    # Service URLs
    WAKEWORD_SERVICE_URL = os.getenv("WAKEWORD_SERVICE_URL", "http://wakeword:8001")
    STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt:8002")
    LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm:8003")
    TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts:8004")
    
    # Audio Configuration
    AUDIO_INPUT_DEVICE: Optional[int] = None if os.getenv("AUDIO_INPUT_DEVICE", "auto") == "auto" else int(os.getenv("AUDIO_INPUT_DEVICE"))
    AUDIO_OUTPUT_DEVICE: Optional[int] = None if os.getenv("AUDIO_OUTPUT_DEVICE", "auto") == "auto" else int(os.getenv("AUDIO_OUTPUT_DEVICE"))
    AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
    
    # Pipeline Configuration
    CHAT_CONTINUATION_ENABLED = os.getenv("CHAT_CONTINUATION_ENABLED", "true").lower() == "true"
    CHAT_CONTINUATION_TIMEOUT = float(os.getenv("CHAT_CONTINUATION_TIMEOUT", "5.0"))
    SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
    SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.5"))
    MAX_RECORDING_DURATION = float(os.getenv("MAX_RECORDING_DURATION", "30.0"))
    
    # Paths
    CONFIRMATIONS_PATH = "/app/assets/confirmations"
    
    CONFIRMATION_PHRASES = ["huh", "mhm", "listening", "yes", "go ahead"]
