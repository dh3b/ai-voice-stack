import asyncio
import config as cfg
from modules.client.oww_client import OWWClient
from modules.client.stt_client import STTClient
from modules.client.tts_client import TTSClient
from modules.client.llm_client import LLMClient

oww_client = OWWClient(cfg.OWWClientConfig())
stt_client = STTClient(cfg.STTClientConfig())
tts_client = TTSClient(cfg.TTSClientConfig())
llm_client = LLMClient(cfg.LLMClientConfig())


async def main():
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    while True:
        wakeword_detected = asyncio.Event()

        print("\n[main] Listening for wakeword...")
        await oww_client.run(detected_event=wakeword_detected)
        oww_client.reset()

        print("[main] Wakeword detected. Starting STT...")
        await stt_client.run(transcript_queue)

        transcript = await transcript_queue.get()
        print(f"[main] Transcript: {transcript}")

        if not transcript:
            continue

        # TODO: send transcript to llm_client, then tts_client


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        oww_client.stop()
