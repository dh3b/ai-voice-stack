"""Model files the app needs at runtime, read from config.py."""

from __future__ import annotations
from pathlib import Path
import config as _cfg
from .util import REPO_ROOT

_tts = _cfg.TTSServerConfig()
REQUIRED: list[tuple[str, str]] = [
    ("LLM (GGUF)", _cfg.LLMServerConfig().model_path),
    ("Whisper (STT)", _cfg.STTServerConfig().model_path),
    ("Piper voice (TTS)", _tts.model_path),
    ("Piper config (TTS)", _tts.model_path + ".json"),
    ("Wake word", _cfg.OWWClientConfig().model_paths[0]),
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def missing() -> list[tuple[str, str]]:
    """(label, path) for each required model file that is absent."""
    return [(label, path) for label, path in REQUIRED if not resolve(path).exists()]
