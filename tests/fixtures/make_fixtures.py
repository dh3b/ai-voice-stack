"""Regenerate the test fixtures in tests/fixtures/.

    python tests/fixtures/make_fixtures.py

Requires the repo's Piper voice at models/en_US-lessac-medium.onnx. Synthesis is made
deterministic by disabling Piper's noise sampling (noise_scale = noise_w_scale = 0), and
the noise fixture is seeded, so re-running produces byte-identical WAVs.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FIXTURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIXTURES_DIR.parents[1]
VOICE_PATH = REPO_ROOT / "models" / "en_US-lessac-medium.onnx"
TARGET_SR = 16000
STT_PHRASE = "the quick brown fox"


def _write_wav(path: Path, samples: np.ndarray, sr: int = TARGET_SR) -> None:
    assert samples.dtype == np.int16
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(samples.tobytes())


def _synth_16k(text: str) -> np.ndarray:
    """Synthesize `text` with Piper and return 16 kHz mono int16 samples."""
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(VOICE_PATH))
    syn = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, normalize_audio=True)
    chunks = list(voice.synthesize(text, syn))
    src_sr = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float64)
    audio = resample_poly(audio, TARGET_SR, src_sr)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * TARGET_SR), dtype=np.int16)


def make_hey_jarvis() -> Path:
    """'hey jarvis' with a little lead-in/tail so openwakeword has context to settle."""
    speech = _synth_16k("hey jarvis")
    samples = np.concatenate([_silence(0.25), speech, _silence(0.4)])
    path = FIXTURES_DIR / "hey_jarvis.wav"
    _write_wav(path, samples)
    return path


def make_stt_phrase() -> Path:
    path = FIXTURES_DIR / "stt_phrase.wav"
    _write_wav(path, _synth_16k(STT_PHRASE))
    (FIXTURES_DIR / "stt_phrase.txt").write_text(STT_PHRASE + "\n", encoding="utf-8")
    return path


def make_white_noise() -> Path:
    rng = np.random.default_rng(1234)
    noise = (rng.uniform(-0.3, 0.3, int(1.5 * TARGET_SR)) * 32767.0).astype(np.int16)
    path = FIXTURES_DIR / "white_noise.wav"
    _write_wav(path, noise)
    return path


def make_silence() -> Path:
    path = FIXTURES_DIR / "silence.wav"
    _write_wav(path, _silence(1.5))
    return path


def main() -> None:
    if not VOICE_PATH.exists():
        raise SystemExit(f"Piper voice not found at {VOICE_PATH} (needed to synthesize speech fixtures).")
    builders = [make_hey_jarvis, make_stt_phrase, make_white_noise, make_silence]
    total = 0
    for build in builders:
        path = build()
        size = path.stat().st_size
        total += size
        print(f"  wrote {path.relative_to(REPO_ROOT)}  ({size / 1024:.1f} KB)")
    print(f"  total: {total / 1024:.1f} KB")


if __name__ == "__main__":
    main()
