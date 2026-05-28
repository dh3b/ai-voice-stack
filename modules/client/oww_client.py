import asyncio
from pathlib import Path
import sys
import sounddevice as sd
from openwakeword.model import Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import OWWClientConfig


class OWWClient:
    def __init__(self, config: OWWClientConfig) -> None:
        self._config = config
        self._stop_event = asyncio.Event()
        self._settle_threshold = max(0.1, config.threshold * 0.4)
        self._settle_chunks = 5
        try:
            self._model = Model(
                wakeword_models=self._config.model_paths,
                inference_framework=self._config.framework,
            )
            self._n_models = len(self._model.models.keys())
        except Exception as e:
            print(f"[error] Failed to load wakeword model: {e}")
            raise

    async def run(self, detected_event: asyncio.Event | None = None) -> None:
        print("Listening for wakewords...")
        try:
            with sd.InputStream(
                channels=self._config.channels,
                samplerate=self._config.sample_rate,
                dtype=self._config.dtype,
                blocksize=self._config.chunk_size,
            ) as mic_stream:
                settle_count = 0
                while not self._stop_event.is_set():
                    audio, _ = mic_stream.read(self._config.chunk_size)
                    audio = audio.flatten()

                    self._model.predict(audio)
                    max_score = max(list(buf)[-1] for buf in self._model.prediction_buffer.values())

                    if settle_count < self._settle_chunks:
                        if max_score < self._settle_threshold:
                            settle_count += 1
                        else:
                            settle_count = 0
                        continue

                    if max_score > self._config.threshold:
                        print(f"Wakeword detected with score {max_score:.5f}/{self._config.threshold}")
                        if detected_event:
                            detected_event.set()
                        self._stop_event.set()
                        break
        except Exception as e:
            print(f"[error] Audio stream error: {e}")

    def reset(self) -> None:
        self._stop_event.clear()

    def stop(self) -> None:
        self._stop_event.set()


async def main() -> None:
    client = OWWClient(OWWClientConfig())
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        client.stop()


if __name__ == "__main__":
    asyncio.run(main())
