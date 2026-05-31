"""Per-stage latency debug for a single voice turn."""

import logging
import time

logger = logging.getLogger("voice_stack")

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
    def __init__(self) -> None:
        self._marks: dict[str, float] = {}

    def reset(self) -> None:
        """Clear all marks. Call once at the start of every turn."""
        self._marks.clear()

    def mark(self, label: str) -> None:
        if label in self._marks:
            return
        self._marks[label] = time.perf_counter()

    def report(self) -> None:
        """Log the inter-stage durations recorded this turn."""
        recorded = [(s, self._marks[s]) for s in _STAGES if s in self._marks]
        if len(recorded) < 2:
            missing = [s for s in _STAGES if s not in self._marks]
            logger.debug(f"[latency] incomplete turn; missing marks: {', '.join(missing)}")
            return

        logger.debug("[latency] turn timings (ms):")
        for (a_label, a_ts), (b_label, b_ts) in zip(recorded, recorded[1:]):
            logger.debug(f"  {a_label:>18} -> {b_label:<18} {(b_ts - a_ts) * 1000:8.1f}")

        first_label, first_ts = recorded[0]
        last_label, last_ts = recorded[-1]
        logger.debug(f"  {'TOTAL':>18}    {first_label} -> {last_label}: {(last_ts - first_ts) * 1000:.1f}")

        missing = [s for s in _STAGES if s not in self._marks]
        if missing:
            logger.debug(f"[latency] (missing marks: {', '.join(missing)})")


tracer = LatencyTracer()
