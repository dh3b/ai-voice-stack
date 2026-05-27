import asyncio
import io
import wave
from pathlib import Path
import sys
import httpx
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TTSClientConfig


class TTSClient:
    def __init__(self, config: TTSClientConfig):
        self._config = config
        self._url = f"http://{config.server_host}:{config.server_port}/"

    async def play(self, queue: asyncio.Queue):
        stream = None
        try:
            async with httpx.AsyncClient() as client:
                while True:
                    text = await queue.get()
                    if text is None:
                        break

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
                    wav_bytes = resp.content

                    with io.BytesIO(wav_bytes) as wav_io:
                        with wave.open(wav_io, "rb") as wav_file:
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            audio_data = wav_file.readframes(wav_file.getnframes())

                    if stream is None:
                        stream = sd.OutputStream(
                            samplerate=sample_rate,
                            channels=channels,
                            dtype=np.int16,
                        )
                        stream.start()
                    stream.write(np.frombuffer(audio_data, dtype=np.int16))
        finally:
            if stream:
                stream.stop()
                stream.close()

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
