"""Acoustic echo cancellation (P2-4) for always-on barge-in.

With an open mic during TTS playback, the wakeword listener hears the assistant's
own voice and can self-trigger (HANDOFF blind-spot #5). This module subtracts the
known playback (the *far-end* reference) from the captured mic signal (*near-end*)
using a block-NLMS adaptive filter, leaving mostly the user's voice for the
wakeword model.

Data flow (a module-level singleton, like the latency tracer):
  - TTS playback thread calls ``push_far(samples, rate)`` as it writes to the speaker.
  - The wakeword mic loop calls ``process(near)`` before running detection.
  - ``far_end_active()`` lets the barge listener raise its threshold while audio plays.

Design notes / honesty:
  - Pure numpy/scipy, no native deps -> portable from the Windows dev box to the
    Jetson target. The ``EchoCanceller`` interface is deliberately small so a
    production canceller (WebRTC APM / SpeexDSP) can be dropped in later.
  - Far/near are kept roughly time-aligned by consuming far in lockstep with near;
    the adaptive filter (``AEC_FILTER_LEN``) must be long enough to span the real
    speaker->mic delay + tail. That delay is hardware-specific -- tune on device.
  - When no far-end is present (normal wakeword listening) ``process`` is a clean
    passthrough and does not adapt, so initial detection is unaffected.
"""
import threading
import time
from collections import deque

import numpy as np
from scipy.signal import resample_poly

import config as cfg

RATE = 16000  # the wakeword/mic sample rate; far-end is resampled to this


class EchoCanceller:
    def __init__(self, rate: int, filter_len: int, mu: float, hangover_s: float, enabled: bool):
        self._rate = rate
        self._L = filter_len
        self._mu = mu
        self._hangover = hangover_s
        self.enabled = enabled
        self._w = np.zeros(filter_len, dtype=np.float64)        # adaptive filter (process thread only)
        self._x_hist = np.zeros(filter_len - 1, dtype=np.float64)  # far history for tap continuity
        self._far: deque[float] = deque()                       # far samples at self._rate (shared)
        self._far_max = filter_len * 8                          # cap to bound drift/memory
        self._far_active_until = 0.0
        self._lock = threading.Lock()

    def push_far(self, samples: np.ndarray, sample_rate: int) -> None:
        """Feed played audio as the far-end reference (called from the TTS thread)."""
        if not self.enabled:
            return
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x / 32768.0
        if sample_rate != self._rate:
            x = resample_poly(x, self._rate, sample_rate)
        with self._lock:
            self._far.extend(x.tolist())
            excess = len(self._far) - self._far_max
            for _ in range(max(0, excess)):
                self._far.popleft()
            self._far_active_until = time.monotonic() + self._hangover

    def process(self, near: np.ndarray) -> np.ndarray:
        """Return the near-end mic chunk with the far-end echo removed."""
        if not self.enabled:
            return near
        near_f = np.asarray(near, dtype=np.float64).flatten() / 32768.0
        B = len(near_f)

        with self._lock:
            avail = min(B, len(self._far))
            far_block = np.zeros(B, dtype=np.float64)
            for i in range(avail):
                far_block[i] = self._far.popleft()

        ext = np.concatenate((self._x_hist, far_block))  # len L-1+B
        far_energy = float(np.dot(far_block, far_block))

        # No (significant) far-end -> nothing to cancel. Keep history, don't adapt,
        # return the input untouched so plain wakeword listening is lossless.
        if far_energy < 1e-6:
            self._x_hist = ext[-(self._L - 1):]
            return near

        y = np.convolve(ext, self._w, mode="valid")     # echo estimate, len B
        e = near_f - y                                   # echo-cancelled near-end
        grad = np.correlate(ext, e, mode="valid")[::-1]  # block-NLMS gradient, len L
        self._w += self._mu * grad / (far_energy + 1e-6)
        self._x_hist = ext[-(self._L - 1):]

        return np.clip(e * 32768.0, -32768, 32767).astype(np.int16)

    def far_end_active(self) -> bool:
        """True while audio is playing (plus a short tail) -- gates the barge threshold."""
        return self.enabled and time.monotonic() < self._far_active_until

    def reset(self) -> None:
        with self._lock:
            self._far.clear()
            self._far_active_until = 0.0
        self._w[:] = 0.0
        self._x_hist[:] = 0.0


canceller = EchoCanceller(
    rate=RATE,
    filter_len=cfg.AEC_FILTER_LEN,
    mu=cfg.AEC_MU,
    hangover_s=cfg.AEC_FAR_HANGOVER_S,
    enabled=cfg.ENABLE_AEC,
)
