"""
NOTE: like always-on barge-in, earcons that play while a mic is open can bleed
into capture without acoustic echo cancellation. Disable via
AppConfig.enable_earcons if that proves disruptive.
"""
import logging

import numpy as np
import sounddevice as sd

from config import AppConfig

logger = logging.getLogger("voice_stack.earcon")

_SR = 16000
_FADE_S = 0.005

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


WAKE_ACK = _tone([(660, 70), (990, 90)])
ENDPOINT_ACK = _tone([(880, 90)])


def play(sound: np.ndarray) -> None:
    """Fire-and-forget playback. Never raises into the pipeline."""
    if not AppConfig().enable_earcons:
        return
    try:
        sd.play(sound, _SR)
    except Exception as e:
        logger.warning(f"Playback failed: {e}")
