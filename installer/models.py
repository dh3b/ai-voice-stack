"""Fetch example models if missing"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config as _cfg

from . import util
from .util import REPO_ROOT

_HF = "https://huggingface.co"
_OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"


@dataclass(frozen=True)
class Model:
    label: str
    path: str  # destination, as configured in config.py (repo-relative or absolute)
    url: str
    sha256: str | None  # None: fetch without an integrity check (small config file)


_tts = _cfg.TTSServerConfig()

REQUIRED: list[Model] = [
    Model(
        "LLM (GGUF)",
        _cfg.LLMServerConfig().model_path,
        f"{_HF}/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94",
    ),
    Model(
        "Whisper (STT)",
        _cfg.STTServerConfig().model_path,
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
    ),
    Model(
        "Piper voice (TTS)",
        _tts.model_path,
        f"{_HF}/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    ),
    Model(
        "Piper config (TTS)",
        _tts.model_path + ".json",
        f"{_HF}/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        None,
    ),
    Model(
        "Wake word",
        _cfg.OWWClientConfig().model_paths[0],
        f"{_OWW}/hey_jarvis_v0.1.onnx",
        "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    ),
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def missing() -> list[Model]:
    """Each required model file that is absent on disk."""
    return [m for m in REQUIRED if not resolve(m.path).exists()]


def fetch(*, force: bool = False) -> None:
    """Download the example models to the paths config.py expects.

    Idempotent: an existing file is skipped, and a user-provided file with a
    different checksum is left untouched. ``force`` deletes first to re-fetch the
    bundled default.
    """
    util.banner("Models")
    for m in REQUIRED:
        dest = resolve(m.path)
        if force and dest.exists():
            dest.unlink()
        util.download(m.url, dest, sha256=m.sha256, label=m.label)
