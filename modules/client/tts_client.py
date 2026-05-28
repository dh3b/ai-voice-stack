import asyncio
import io
import re
import wave
from pathlib import Path
import sys
import httpx
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TTSClientConfig


class TTSClient:
    SENTENCE_END = re.compile(r"[.?!]+")

    def __init__(self, config: TTSClientConfig):
        self._config = config
        self._url = f"http://{config.server_host}:{config.server_port}/"
        self._stream: sd.OutputStream | None = None

    async def play(self, queue: asyncio.Queue):
        audio_queue: asyncio.Queue[tuple[bytes, int, int] | None] = asyncio.Queue()

        async def _synthesize_loop():
            try:
                async with httpx.AsyncClient() as client:
                    mode = self._config.chunk_mode
                    if mode == "chars":
                        iterator = self._iter_chars(queue)
                    elif mode == "words":
                        iterator = self._iter_words(queue)
                    else:
                        iterator = self._iter_sentences(queue)

                    async for text in iterator:
                        wav_bytes = await self._synthesize(client, text)
                        if wav_bytes is None:
                            continue
                        parsed = self._parse_wav(wav_bytes)
                        if parsed is None:
                            continue
                        await audio_queue.put(parsed)
            finally:
                await audio_queue.put(None)

        async def _playback_loop():
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                audio_data, sample_rate, channels = item
                if self._stream is None:
                    self._stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=channels,
                        dtype=np.int16,
                    )
                    self._stream.start()
                self._stream.write(np.frombuffer(audio_data, dtype=np.int16))

        try:
            await asyncio.gather(_synthesize_loop(), _playback_loop())
        finally:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

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

    async def _iter_sentences(self, queue: asyncio.Queue):
        buffer = ""
        while True:
            chunk = await queue.get()
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

    async def _iter_chars(self, queue: asyncio.Queue):
        buffer = ""
        size = self._config.chunk_size
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            buffer += chunk
            while len(buffer) >= size:
                yield buffer[:size]
                buffer = buffer[size:]
        if buffer:
            yield buffer

    async def _iter_words(self, queue: asyncio.Queue):
        buffer = ""
        size = self._config.chunk_size
        while True:
            chunk = await queue.get()
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
    client = TTSClient(TTSClientConfig())
    queue = asyncio.Queue()
    for text in [
        "Hello, this is the first sentence.",
        "And here is another one.",
        "Finally, a third sentence to synthesize.",
    ]:
        await queue.put(text)
    await queue.put(None)

    await client.play(queue)
    client.close()


if __name__ == "__main__":
    asyncio.run(main())
