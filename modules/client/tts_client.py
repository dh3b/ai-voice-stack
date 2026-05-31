import asyncio
import io
import logging
import queue
import re
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import httpx
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TTSClientConfig, AppConfig
from modules.utility.latency import tracer, TTS_FIRST_CHUNK, AUDIO_FIRST_WRITE

logger = logging.getLogger("voice_stack.tts")


class TTSClient:
    SENTENCE_END = re.compile(r"[.?!]+")
    CLAUSE_END = re.compile(r"[,;:.?!]+")  # first-chunk boundary for hybrid mode

    def __init__(self, config: TTSClientConfig):
        self._config = config
        self._url = f"http://{config.server_host}:{config.server_port}/"
        self._playback_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-playback"
        )

        self._stop_event = threading.Event()

        if AppConfig().warmup_on_init:
            self._warmup()

    def _warmup(self) -> None:
        """Prime the Piper HTTP server (blocking) so the first synth isn't cold."""
        try:
            with httpx.Client() as client:
                resp = client.post(
                    self._url,
                    json={
                        "text": "Warming up.",
                        "length_scale": self._config.length_scale,
                        "noise_scale": self._config.noise_scale,
                        "noise_w_scale": self._config.noise_w_scale,
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
            logger.debug("Piper warmed up.")
        except Exception as e:
            logger.warning(f"warmup skipped ({e!r}); first synth may be cold.")

    def interrupt(self) -> None:
        """Signal an in-progress play() to stop ASAP and flush buffered audio."""
        self._stop_event.set()

    async def play(self, token_queue: asyncio.Queue):
        self._stop_event.clear()
        loop = asyncio.get_event_loop()
        # thread consumes. Must be a queue.Queue, not asyncio.Queue.
        audio_queue: queue.Queue = queue.Queue()

        async def _synthesize_loop():
            try:
                async with httpx.AsyncClient() as client:
                    mode = self._config.chunk_mode
                    if mode == "chars":
                        iterator = self._iter_chars(token_queue)
                    elif mode == "words":
                        iterator = self._iter_words(token_queue)
                    elif mode == "sentence":
                        iterator = self._iter_sentences(token_queue)
                    else:
                        iterator = self._iter_hybrid(token_queue)

                    async for text in iterator:
                        if self._stop_event.is_set():
                            break
                        wav_bytes = await self._synthesize(client, text)
                        if wav_bytes is None:
                            continue
                        parsed = self._parse_wav(wav_bytes)
                        if parsed is None:
                            continue
                        tracer.mark(TTS_FIRST_CHUNK)
                        audio_queue.put_nowait(parsed)
            finally:
                audio_queue.put_nowait(None)

        # Playback runs on the dedicated thread; run_in_executor hands back an
        # awaitable so it can be gathered with the (async) synth loop.
        playback_future = loop.run_in_executor(
            self._playback_executor, self._playback_worker, audio_queue
        )
        await asyncio.gather(_synthesize_loop(), playback_future)

    def _playback_worker(self, audio_queue: "queue.Queue") -> None:
        """Consume synthesized audio and write it to the speaker in small chunks."""
        stream: sd.OutputStream | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    item = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                audio_data, sample_rate, channels = item
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=sample_rate, channels=channels, dtype=np.int16
                    )
                    stream.start()
                samples = np.frombuffer(audio_data, dtype=np.int16)
                step = max(1, int(sample_rate * self._config.playback_chunk_ms / 1000)) * channels
                for start in range(0, len(samples), step):
                    if self._stop_event.is_set():
                        break
                    tracer.mark(AUDIO_FIRST_WRITE)
                    stream.write(samples[start:start + step])
        finally:
            if stream is not None:
                if self._stop_event.is_set():
                    stream.abort()  # flush buffered samples immediately (barge-in)
                else:
                    stream.stop()
                stream.close()

    async def _synthesize(self, client: httpx.AsyncClient, text: str) -> bytes | None:
        try:
            resp = await client.post(
                self._url,
                json={
                    "text": text,
                    "length_scale": self._config.length_scale,
                    "noise_scale": self._config.noise_scale,
                    "noise_w_scale": self._config.noise_w_scale,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPStatusError:
            return None

    @staticmethod
    def _parse_wav(wav_bytes: bytes) -> tuple[bytes, int, int] | None:
        try:
            with io.BytesIO(wav_bytes) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    sr = wav_file.getframerate()
                    ch = wav_file.getnchannels()
                    data = wav_file.readframes(wav_file.getnframes())
            return data, sr, ch
        except wave.Error:
            return None

    async def _iter_hybrid(self, token_queue: asyncio.Queue):
        """Emit the first chunk early, then revert to sentence granularity.

        The opener is whichever lands first as tokens stream: a clause boundary
        (, ; : . ? !) or first_chunk_max_words complete words. That starts audio
        well before a full sentence exists, while every later chunk stays
        sentence-sized so Piper prosody isn't globally degraded.
        """
        cap = self._config.first_chunk_max_words
        buffer = ""
        first_done = False
        while True:
            chunk = await token_queue.get()
            if chunk is None:
                break
            buffer += chunk

            if not first_done:
                match = self.CLAUSE_END.search(buffer)
                if match:
                    head = buffer[: match.end()].strip()
                    if head:
                        yield head
                        buffer = buffer[match.end():]
                        first_done = True
                    continue
                words = buffer.split()
                if len(words) > cap:
                    yield " ".join(words[:cap])
                    buffer = " ".join(words[cap:])
                    first_done = True
                continue

            sentences = self.SENTENCE_END.split(buffer)
            if len(sentences) > 1:
                buffer = sentences[-1]
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence

        remaining = buffer.strip()
        if remaining:
            yield remaining

    async def _iter_sentences(self, token_queue: asyncio.Queue):
        buffer = ""
        while True:
            chunk = await token_queue.get()
            if chunk is None:
                break
            buffer += chunk
            sentences = self.SENTENCE_END.split(buffer)
            if len(sentences) > 1:
                buffer = sentences[-1]
                for sentence in sentences[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence
        remaining = buffer.strip()
        if remaining:
            yield remaining

    async def _iter_chars(self, token_queue: asyncio.Queue):
        buffer = ""
        size = self._config.chunk_size
        while True:
            chunk = await token_queue.get()
            if chunk is None:
                break
            buffer += chunk
            while len(buffer) >= size:
                yield buffer[:size]
                buffer = buffer[size:]
        if buffer:
            yield buffer

    async def _iter_words(self, token_queue: asyncio.Queue):
        buffer = ""
        size = self._config.chunk_size
        while True:
            chunk = await token_queue.get()
            if chunk is None:
                break
            buffer += chunk
            words = buffer.split()
            while len(words) >= size:
                yield " ".join(words[:size])
                words = words[size:]
            buffer = " ".join(words)
        if buffer:
            yield buffer

    def close(self):
        pass


async def main():
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    client = TTSClient(TTSClientConfig())
    token_queue = asyncio.Queue()
    for text in [
        "Hello, this is the first sentence.",
        "And here is another one.",
        "Finally, a third sentence to synthesize.",
    ]:
        await token_queue.put(text)
    await token_queue.put(None)

    await client.play(token_queue)
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
