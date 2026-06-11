import asyncio
import logging
import config as cfg
from modules.client.oww_client import OWWClient
from modules.client.stt_client import STTClient
from modules.client.tts_client import TTSClient
from modules.client.llm_client import LLMClient
from modules.utility.latency import tracer
from modules.utility import earcon

app_config = cfg.AppConfig()

logging.basicConfig(level=app_config.logging_level, format=app_config.logging_format)
logger = logging.getLogger("voice_stack.main")

if app_config.disable_http_logging:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)


oww_cfg = cfg.OWWClientConfig()
stt_cfg = cfg.STTClientConfig()
tts_cfg = cfg.TTSClientConfig()
llm_cfg = cfg.LLMClientConfig()
tools_cfg = cfg.ToolsConfig()

oww_client: OWWClient | None = None
stt_client: STTClient | None = None
tts_client: TTSClient | None = None
llm_client: LLMClient | None = None


def _init_clients() -> None:
    global oww_client, stt_client, tts_client, llm_client
    oww_client = OWWClient(oww_cfg)
    stt_client = STTClient(stt_cfg)
    tts_client = TTSClient(tts_cfg)
    llm_client = LLMClient(llm_cfg, tools_cfg)


async def run_turn(transcript: str, llm=None, tts=None, response_timeout: float | None = None) -> None:
    """Run one LLM->TTS turn"""
    llm = llm if llm is not None else llm_client
    tts = tts if tts is not None else tts_client
    if response_timeout is None:
        response_timeout = llm_cfg.response_timeout
    response_queue: asyncio.Queue = asyncio.Queue()
    llm_task = asyncio.create_task(llm.run(transcript, response_queue))
    tts_task = asyncio.create_task(tts.play(response_queue))

    try:
        # wait_for cancels llm_task if it overruns the budget.
        await asyncio.wait_for(llm_task, timeout=response_timeout)
    except asyncio.TimeoutError:
        logger.warning(f"LLM exceeded {response_timeout}s; ending turn.")
    except Exception as e:
        logger.error(f"LLM error: {e!r}; ending turn.")
    finally:
        if not llm_task.done():
            llm_task.cancel()
        response_queue.put_nowait(None)
        try:
            await tts_task
        except Exception as e:
            logger.error(f"TTS error: {e!r}")


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
    _init_clients()
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()
    skip_wakeword = False  # set after a barge-in or to enter continuation listening
    listen_timeout = None  # None => default response_timeout; set => continuation follow-up window

    while True:
        if not skip_wakeword:
            await oww_client.run()
        skip_wakeword = False

        earcon.play(earcon.WAKE_ACK)

        if listen_timeout is None:
            logger.info("Wakeword detected. Starting STT...")
        else:
            logger.info(f"Listening for a follow-up ({listen_timeout}s, no wakeword)...")
        tracer.reset()
        await stt_client.run(transcript_queue, listen_timeout=listen_timeout)
        in_continuation = listen_timeout is not None
        listen_timeout = None

        transcript = await transcript_queue.get()
        logger.info(f"Transcript: {transcript}")

        if not transcript:
            earcon.play(earcon.ENDPOINT_NACK)
            if in_continuation:
                logger.info("No follow-up within the window; returning to wakeword.")
            continue

        logger.info("Starting LLM/TTS turn...")
        earcon.play(earcon.ENDPOINT_ACK)

        # Optional transcript processing might go here

        interrupted = await run_turn_with_bargein(transcript)
        tracer.report()

        if interrupted:
            logger.info("Barge-in: re-listening immediately.")
            skip_wakeword = True
        elif app_config.continuation_enabled:
            logger.info(f"Continuation: listening for a follow-up ({stt_cfg.continuation_timeout}s)...")
            skip_wakeword = True
            listen_timeout = stt_cfg.continuation_timeout


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if oww_client is not None:
            oww_client.stop()
