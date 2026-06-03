import asyncio
import json
import logging
import sys
import numpy as np
import sounddevice as sd

from config import STTClientConfig, AppConfig
from modules.utility.latency import tracer, ENDPOINT_FINAL

logger = logging.getLogger("voice_stack.stt")


class STTClient:
    def __init__(self, config: STTClientConfig):
        self._config = config

        if AppConfig().warmup_on_init:
            logger.debug("warmed up.") # I hope so

    async def run(self, transcript_queue: asyncio.Queue, listen_timeout: float | None = None):
        if listen_timeout is None:
            listen_timeout = self._config.response_timeout

        try:
            reader, writer = await asyncio.open_connection(
                self._config.server_host, self._config.server_port
            )
        except ConnectionRefusedError:
            logger.error(f"Could not connect to {self._config.server_host}:{self._config.server_port} - is the server running?")
            # Unblock the orchestrator's transcript_queue.get(); "" => return to listening.
            await transcript_queue.put("")
            return

        logger.info(f"Connected to {self._config.server_host}:{self._config.server_port}")
        stop_event = asyncio.Event()

        try:
            await asyncio.gather(
                self._stream_mic_to_server(writer, stop_event),
                self._read_transcripts(reader, transcript_queue, stop_event, listen_timeout),
            )
        except KeyboardInterrupt:
            logger.info("Stopping...")
            stop_event.set()

    async def _stream_mic_to_server(
        self, writer: asyncio.StreamWriter, stop_event: asyncio.Event
    ):
        loop = asyncio.get_event_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        def sd_callback(indata: np.ndarray, frames, time, status):
            if status:
                logger.warning(f"sounddevice: {status}")
            loop.call_soon_threadsafe(audio_queue.put_nowait, indata.tobytes())

        with sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=self._config.channels,
            dtype=self._config.dtype,
            blocksize=self._config.block_size,
            callback=sd_callback,
        ):
            logger.info("Streaming audio to SimulStreaming server...")
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                    writer.write(chunk)
                    await writer.drain()
                except asyncio.TimeoutError:
                    pass

        writer.close()
        await writer.wait_closed()

    async def _read_transcripts(self, reader: asyncio.StreamReader, transcript_queue: asyncio.Queue, stop_event: asyncio.Event, listen_timeout: float):
        confirmed_text = []
        deadline = asyncio.get_event_loop().time() + listen_timeout
        got_first_partial = False

        while not stop_event.is_set():
            if not got_first_partial:
                now = asyncio.get_event_loop().time()
                if now >= deadline:
                    logger.info(f"No speech detected within {listen_timeout}s")
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
                sys.stdout.write(f"\r[stt partial] {''.join(confirmed_text)}")
                sys.stdout.flush()

            if msg.get("is_final"):
                tracer.mark(ENDPOINT_FINAL)
                utterance = " ".join(confirmed_text).strip()
                sys.stdout.write("\n")
                logger.info(f"transcript: '{utterance}'")
                await transcript_queue.put(utterance)
                confirmed_text = []
                stop_event.set()
                break


async def main():
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    client = STTClient(STTClientConfig())
    transcript_queue = asyncio.Queue()

    async def llm_consumer(queue: asyncio.Queue):
        while True:
            utterance = await queue.get()
            logger.info(f"received utterance: '{utterance}'")
            queue.task_done()

    try:
        await asyncio.gather(
            client.run(transcript_queue),
            llm_consumer(transcript_queue),
        )
    except KeyboardInterrupt:
        logger.info("Stopping...")


if __name__ == "__main__":
    asyncio.run(main())
