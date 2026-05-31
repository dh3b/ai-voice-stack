import asyncio
import logging
import config as cfg
from modules.client.oww_client import OWWClient
from modules.client.stt_client import STTClient
from modules.client.tts_client import TTSClient
from modules.client.llm_client import LLMClient
from modules.utility.latency import tracer
from modules.utility import earcon

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger("voice_stack")

oww_cfg = cfg.OWWClientConfig()
stt_cfg = cfg.STTClientConfig()
tts_cfg = cfg.TTSClientConfig()
llm_cfg = cfg.LLMClientConfig()

oww_client = OWWClient(oww_cfg)
stt_client = STTClient(stt_cfg)
tts_client = TTSClient(tts_cfg)
llm_client = LLMClient(llm_cfg)


async def run_turn(transcript: str) -> None:
    """Run one LLM->TTS turn"""
    response_queue: asyncio.Queue = asyncio.Queue()
    llm_task = asyncio.create_task(llm_client.run(transcript, response_queue))
    tts_task = asyncio.create_task(tts_client.play(response_queue))

    try:
        # wait_for cancels llm_task if it overruns the budget.
        await asyncio.wait_for(llm_task, timeout=llm_cfg.response_timeout)
    except asyncio.TimeoutError:
        logger.warning(f"[main] LLM exceeded {llm_cfg.response_timeout}s; ending turn.")
    except Exception as e:
        logger.error(f"[main] LLM error: {e!r}; ending turn.")
    finally:
        if not llm_task.done():
            llm_task.cancel()
        response_queue.put_nowait(None)
        try:
            await tts_task
        except Exception as e:
            logger.error(f"[main] TTS error: {e!r}")


async def run_turn_with_bargein(transcript: str) -> bool:
    """Run a turn while an always-on wakeword listener watches for a barge-in.

    Returns True if the user interrupted (the wakeword fired mid-turn): the turn
    is aborted (synth stopped, playback flushed, LLM cancelled) and the caller
    should go straight back to STT. Returns False if the turn finished on its own.
    """
    barge_task = asyncio.create_task(oww_client.run())
    turn_task = asyncio.create_task(run_turn(transcript))

    done, _ = await asyncio.wait(
        {barge_task, turn_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if turn_task in done:
        oww_client.stop()
        await barge_task
        return False

    # Barge-in: the wakeword fired while the turn was still going.
    tts_client.interrupt()
    turn_task.cancel()
    try:
        await turn_task
    except asyncio.CancelledError:
        pass
    await barge_task # ensures the mic is closed
    return True


async def main():
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    skip_wakeword = False  # set after a barge-in

    while True:
        if not skip_wakeword:
            logger.info("[main] Listening for wakeword...")
            await oww_client.run()
        skip_wakeword = False

        earcon.play(earcon.WAKE_ACK)

        logger.info("[main] Wakeword detected. Starting STT...")
        tracer.reset()
        await stt_client.run(transcript_queue)

        transcript = await transcript_queue.get()
        logger.info(f"[main] Transcript: {transcript}")

        if not transcript:
            continue

        earcon.play(earcon.ENDPOINT_ACK)

        # Optional transcript processing might go here

        interrupted = await run_turn_with_bargein(transcript)
        tracer.report()

        if interrupted:
            logger.info("[main] Barge-in: re-listening immediately.")
            skip_wakeword = True


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[main] Shutting down...")
        oww_client.stop()
