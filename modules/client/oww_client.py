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
        try:
            self._model = Model(wakeword_models=self._config.model_paths, inference_framework=self._config.framework)
            self._n_models = len(self._model.models.keys())
        except Exception as e:
            print(f"[error] Failed to load wakeword model: {e}")
            raise

    async def run(self) -> None:
        print("Listening for wakewords...")
        try:
            with sd.InputStream(
                channels=self._config.channels,
                samplerate=self._config.sample_rate,
                dtype=self._config.dtype,
                blocksize=self._config.chunk_size,
            ) as mic_stream:
                while not self._stop_event.is_set():
                    audio, _ = mic_stream.read(self._config.chunk_size)
                    audio = audio.flatten()

                    pred = self._model.predict(audio)

                    for mdl in self._model.prediction_buffer.keys():
                        scores = list(self._model.prediction_buffer[mdl])
                        curr_score = format(scores[-1], ".20f").replace("-", "")

                        if scores[-1] > self._config.threshold:
                            print(f"Wakeword detected by model '{mdl}' with score {curr_score}/{self._config.threshold}")
        except Exception as e:
            print(f"[error] Audio stream error: {e}")

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
