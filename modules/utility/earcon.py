"""Short confirmation earcons (P1-5).

Audible acknowledgements that mask perceived latency: a rising chirp once the
wakeword fires, and a single blip once the user's speech is endpointed -- so the
unavoidable LLM time-to-first-token (~1s on CPU) isn't dead silence ("did it
hear me?"). Tones are synthesized in code (no audio assets) and played
fire-and-forget so they never block the event loop.

NOTE: like always-on barge-in, earcons that play while a mic is open can bleed
into capture without acoustic echo cancellation (P2-4). Disable via
config.ENABLE_EARCONS if that proves disruptive.
"""
import numpy as np
import sounddevice as sd

import config as cfg

_SR = 16000
_FADE_S = 0.005  # 5ms raised-cosine fade to avoid clicks


def _tone(segments: list[tuple[float, float]], volume: float = 0.18) -> np.ndarray:
    """Build a click-free int16 waveform from (freq_hz, duration_ms) segments."""
    parts = []
    fade = max(1, int(_SR * _FADE_S))
    for freq, dur_ms in segments:
        n = max(2 * fade, int(_SR * dur_ms / 1000))
        wave = np.sin(2 * np.pi * freq * (np.arange(n) / _SR))
        env = np.ones(n)
        ramp = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
        env[:fade] = ramp
        env[-fade:] = ramp[::-1]
        parts.append(wave * env)
    audio = np.concatenate(parts) * volume
    return (audio * 32767).astype(np.int16)


# Ascending two-note chirp: "I'm listening."
WAKE_ACK = _tone([(660, 70), (990, 90)])
# Single soft blip: "got it, working on it."
ENDPOINT_ACK = _tone([(880, 90)])


def play(sound: np.ndarray) -> None:
    """Fire-and-forget playback. Never raises into the pipeline."""
    if not cfg.ENABLE_EARCONS:
        return
    try:
        sd.play(sound, _SR)
    except Exception as e:
        print(f"[earcon] playback failed: {e}")
