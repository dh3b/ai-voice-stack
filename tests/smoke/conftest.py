"""
Servers are started through the project's own ``start_server(config)`` entry points with
test ``*ServerConfig`` dataclasses (tiny model where given, a free port), and readiness is
checked with the same probes ``installer/serve.py`` uses in production.

Model selection: each backend resolves an env override first (CI points these at the tiny
substitutes it downloads), then the repo's full model, else the test skips.
"""

from __future__ import annotations

import math
import os
import re
import socket
import subprocess
import sys
import time
import types
import wave
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import modules.server.llm_server as llm_server
import modules.server.stt_server as stt_server
import modules.server.tts_server as tts_server
from config import LLMServerConfig, OWWClientConfig, STTServerConfig, TTSServerConfig
from installer.serve import _http_ok, _port_open

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
LOGS_DIR = REPO_ROOT / "logs"


def _looks_real(path: Path, min_bytes: int = 50_000) -> bool:
    """True if `path` is a real model file present on disk (not missing or a tiny stub)."""
    try:
        return path.stat().st_size >= min_bytes
    except OSError:
        return False


def _resolve_model(env_var: str, default: str) -> str | None:
    override = os.environ.get(env_var)
    if override and _looks_real(Path(override)):
        return override
    default_path = Path(default)
    if not default_path.is_absolute():
        default_path = REPO_ROOT / default_path
    return str(default_path) if _looks_real(default_path) else None


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _spawn(start_fn, cfg, label):
    """Call a server module's start_server(cfg), routing its stdout/stderr (which it
    inherits from sys.stdout/sys.stderr) to a log file so it survives pytest capture."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"test_{label}_server.log"
    log = open(log_path, "w")
    saved = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = log, log
    try:
        proc = start_fn(cfg)
    finally:
        sys.stdout, sys.stderr = saved
    return proc, log, log_path


def _tail(log_path: Path, n: int = 25) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _wait_ready(ready, proc, *, timeout, label, log_path):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"{label} server exited early (code {proc.returncode}).\n{_tail(log_path)}")
        if ready():
            return
        time.sleep(0.5)
    pytest.fail(f"{label} server not ready within {timeout}s.\n{_tail(log_path)}")


def _stop(proc, log):
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        log.close()


@pytest.fixture(scope="module")
def llm_server_config():
    gguf = _resolve_model("VOICE_STACK_TEST_LLM", LLMServerConfig().model_path)
    if gguf is None:
        pytest.skip("no LLM GGUF available (set VOICE_STACK_TEST_LLM or place the repo model)")
    exe = LLMServerConfig().executable_path
    exe_path = Path(exe) if Path(exe).is_absolute() else REPO_ROOT / exe
    if not exe_path.exists():
        pytest.skip(f"llama-server not built at {exe_path} (run: uv run python -m installer llama)")

    cfg = LLMServerConfig(model_path=gguf, server_port=_free_port(), context_window=2048)
    proc, log, log_path = _spawn(llm_server.start_server, cfg, "llm")
    try:
        _wait_ready(
            lambda: _http_ok(f"http://{cfg.server_host}:{cfg.server_port}/health"),
            proc, timeout=180, label="llm", log_path=log_path,
        )
        yield cfg
    finally:
        _stop(proc, log)


@pytest.fixture(scope="module")
def tts_server_config():
    voice = _resolve_model("VOICE_STACK_TEST_TTS", TTSServerConfig().model_path)
    if voice is None:
        pytest.skip("no Piper voice available (set VOICE_STACK_TEST_TTS or place the repo model)")

    cfg = TTSServerConfig(model_path=voice, server_port=_free_port())
    proc, log, log_path = _spawn(tts_server.start_server, cfg, "tts")
    try:
        _wait_ready(
            lambda: _port_open(cfg.server_host, cfg.server_port),
            proc, timeout=120, label="tts", log_path=log_path,
        )
        yield cfg
    finally:
        _stop(proc, log)


@pytest.fixture(scope="module")
def stt_server_config():
    model = _resolve_model("VOICE_STACK_TEST_STT", STTServerConfig().model_path)
    if model is None:
        pytest.skip("no Whisper checkpoint available (set VOICE_STACK_TEST_STT or place the repo model)")
    if not (REPO_ROOT / "simulstreaming_lib" / "simulstreaming_whisper_server.py").exists():
        pytest.skip("SimulStreaming not present (run: uv run python -m installer stt)")

    cfg = STTServerConfig(model_path=model, server_port=_free_port())
    proc, log, log_path = _spawn(stt_server.start_server, cfg, "stt")
    try:
        _wait_ready(
            lambda: _port_open(cfg.server_host, cfg.server_port),
            proc, timeout=240, label="stt", log_path=log_path,
        )
        yield cfg
    finally:
        _stop(proc, log)


@pytest.fixture
def wakeword_model_path():
    model = _resolve_model("VOICE_STACK_TEST_WAKEWORD", OWWClientConfig().model_paths[0])
    if model is None:
        pytest.skip("no wakeword model available (set VOICE_STACK_TEST_WAKEWORD or place the repo model)")
    return model


def _read_wav_int16(path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1, "fixtures are mono"
        assert wav.getsampwidth() == 2, "fixtures are int16"
        frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype=np.int16)


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein over word tokens.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def _fuzzy_match(expected: str, actual: str, max_wer: float = 0.4) -> bool:
    norm_expected = " ".join(_normalize(expected))
    norm_actual = " ".join(_normalize(actual))
    if norm_expected and norm_expected in norm_actual:
        return True
    return _wer(expected, actual) <= max_wer


def _rms(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))


@pytest.fixture
def smoke():
    """Pure helpers for the smoke tests (no server interaction)."""
    return types.SimpleNamespace(
        fixtures_dir=FIXTURES_DIR,
        read_wav_int16=_read_wav_int16,
        fuzzy_match=_fuzzy_match,
        wer=_wer,
        rms=_rms,
    )
