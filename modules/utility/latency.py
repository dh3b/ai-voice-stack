"""Per-stage latency instrumentation for a single voice turn (P0-1).

A turn flows STT endpoint -> first LLM token -> first synth chunk -> first audio
out. Each stage calls ``tracer.mark(...)`` the moment its milestone happens; the
orchestrator calls ``tracer.reset()`` at the start of a turn and ``tracer.report()``
at the end to print the inter-stage durations.

The tracer is a module-level singleton (mirroring ``tool_registry.registry``) so
stages can import it without threading an object through every call. ``mark`` only
records the *first* occurrence of each label per turn, so callers can call it
unconditionally inside their streaming loops.
"""

import time

import config as cfg

# Milestone labels, in pipeline order. The report walks this sequence and prints
# the delta between each consecutive pair that was actually recorded.
ENDPOINT_FINAL = "endpoint_final"        # (a) STT/VAD declared the utterance final
LLM_FIRST_TOKEN = "llm_first_token"      # (b) first content token streamed from the LLM
TTS_FIRST_CHUNK = "tts_first_chunk"      # (c) first audio chunk synthesized
AUDIO_FIRST_WRITE = "audio_first_write"  # (d) first audio sample written to the device

_STAGES: tuple[str, ...] = (
    ENDPOINT_FINAL,
    LLM_FIRST_TOKEN,
    TTS_FIRST_CHUNK,
    AUDIO_FIRST_WRITE,
)


class LatencyTracer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._marks: dict[str, float] = {}

    def reset(self) -> None:
        """Clear all marks. Call once at the start of every turn."""
        self._marks.clear()

    def mark(self, label: str) -> None:
        """Record a monotonic timestamp for ``label`` the first time it is seen.

        Cheap no-op when disabled or already marked this turn, so it is safe to
        call inside hot streaming loops. Each label is written by exactly one
        stage, so the check-then-set is single-writer even once a stage runs on
        its own thread (e.g. the playback thread after P0-3).
        """
        if not self.enabled or label in self._marks:
            return
        self._marks[label] = time.perf_counter()

    def report(self) -> None:
        """Print the inter-stage durations recorded this turn."""
        if not self.enabled:
            return

        recorded = [(s, self._marks[s]) for s in _STAGES if s in self._marks]
        if len(recorded) < 2:
            missing = [s for s in _STAGES if s not in self._marks]
            print(f"[latency] incomplete turn; missing marks: {', '.join(missing)}")
            return

        print("[latency] turn timings (ms):")
        for (a_label, a_ts), (b_label, b_ts) in zip(recorded, recorded[1:]):
            print(f"  {a_label:>18} -> {b_label:<18} {(b_ts - a_ts) * 1000:8.1f}")

        first_label, first_ts = recorded[0]
        last_label, last_ts = recorded[-1]
        print(f"  {'TOTAL':>18}    {first_label} -> {last_label}: {(last_ts - first_ts) * 1000:.1f}")

        missing = [s for s in _STAGES if s not in self._marks]
        if missing:
            print(f"[latency] (missing marks: {', '.join(missing)})")


tracer = LatencyTracer(enabled=cfg.DEBUG_LATENCY)
