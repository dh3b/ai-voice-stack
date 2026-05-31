import asyncio
import json
from pathlib import Path
import sys
import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import STTClientConfig
from modules.utility.latency import tracer, ENDPOINT_FINAL


class STTClient:
    def __init__(self, config: STTClientConfig):
        self._config = config

    async def run(self, transcript_queue: asyncio.Queue):
        try:
            reader, writer = await asyncio.open_connection(
                self._config.server_host, self._config.server_port
            )
        except ConnectionRefusedError:
            print(f"[error] Could not connect to {self._config.server_host}:{self._config.server_port} - is the server running?")
            # Unblock the orchestrator's transcript_queue.get(); "" => return to listening.
            await transcript_queue.put("")
            return

        print(f"Connected to {self._config.server_host}:{self._config.server_port}")
        stop_event = asyncio.Event()

        try:
            await asyncio.gather(
                self._stream_mic_to_server(writer, stop_event),
                self._read_transcripts(reader, transcript_queue, stop_event),
            )
        except KeyboardInterrupt:
            print("\nStopping...")
            stop_event.set()

    async def _stream_mic_to_server(
        self, writer: asyncio.StreamWriter, stop_event: asyncio.Event
    ):
        loop = asyncio.get_event_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def sd_callback(indata: np.ndarray, frames, time, status):
            if status:
                print(f"[sounddevice] {status}", file=sys.stderr)
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.tobytes())

        with sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self._config.block_size,
            callback=sd_callback,
        ):
            print("[mic] Streaming audio to SimulStreaming server...")
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                    writer.write(chunk)
                    await writer.drain()
                except asyncio.TimeoutError:
                    pass

        writer.close()
        await writer.wait_closed()

    async def _read_transcripts(self, reader: asyncio.StreamReader, transcript_queue: asyncio.Queue, stop_event: asyncio.Event):
        confirmed_text = []
        deadline = asyncio.get_event_loop().time() + self._config.response_timeout
        got_first_partial = False

        while not stop_event.is_set():
            if not got_first_partial:
                now = asyncio.get_event_loop().time()
                if now >= deadline:
                    print(
                        f"\n[stt] No speech detected within {self._config.response_timeout}s"
                    )
                    await transcript_queue.put("")
                    stop_event.set()
                    break
                timeout = min(deadline - now, 0.5)
            else:
                timeout = None

            try:
                raw_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                continue

            if not raw_line:
                break

            line = raw_line.decode().strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = msg.get("text", "").strip()
            if text:
                if not got_first_partial:
                    got_first_partial = True
                confirmed_text.append(text)
                print(f"[stt partial] {''.join(confirmed_text)}", end="\r")

            if msg.get("is_final"):
                tracer.mark(ENDPOINT_FINAL)
                utterance = " ".join(confirmed_text).strip()
                print(f"\n[stt final]   '{utterance}'")
                await transcript_queue.put(utterance)
                confirmed_text = []
                stop_event.set()
                break


async def main():
    client = STTClient(STTClientConfig())
    transcript_queue = asyncio.Queue()

    async def llm_consumer(queue: asyncio.Queue):
        while True:
            utterance = await queue.get()
            print(f"[llm] Received utterance: '{utterance}'")
            queue.task_done()

    try:
        await asyncio.gather(
            client.run(transcript_queue),
            llm_consumer(transcript_queue),
        )
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    asyncio.run(main())
