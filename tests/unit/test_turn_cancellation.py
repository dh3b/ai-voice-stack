"""
When the LLM never finishes, run_turn should cancel it at response_timeout, push the
None sentinel so the TTS consumer drains and stops, and leave no tasks behind.
"""

import asyncio
import pipeline

class _StubLLM:
    async def run(self, transcript, queue):
        await asyncio.Event().wait()


class _RecordingTTS:
    def __init__(self):
        self.received = []
        self.terminated = False

    async def play(self, queue):
        try:
            while True:
                item = await queue.get()
                self.received.append(item)
                if item is None:
                    break
        finally:
            self.terminated = True


async def test_turn_cancellation():
    tts = _RecordingTTS()
    tasks_before = asyncio.all_tasks()

    await pipeline.run_turn("hello", llm=_StubLLM(), tts=tts, response_timeout=0.2)
    await asyncio.sleep(0)  # let the cancelled LLM task finalize

    # No tasks created by run_turn are still alive (all_tasks excludes finished tasks).
    leaked = asyncio.all_tasks() - tasks_before
    assert not leaked, f"leaked tasks: {leaked}"

    # The None sentinel reached the TTS queue and the TTS consumer terminated.
    assert tts.received[-1] is None
    assert tts.terminated
