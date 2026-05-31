import asyncio
import config as cfg
from modules.client.oww_client import OWWClient
from modules.client.stt_client import STTClient
from modules.client.tts_client import TTSClient
from modules.client.llm_client import LLMClient
from modules.utility.latency import tracer
from modules.utility import earcon

oww_cfg = cfg.OWWClientConfig()
stt_cfg = cfg.STTClientConfig()
tts_cfg = cfg.TTSClientConfig()
llm_cfg = cfg.LLMClientConfig()

oww_client = OWWClient(oww_cfg)
stt_client = STTClient(stt_cfg)
tts_client = TTSClient(tts_cfg)
llm_client = LLMClient(llm_cfg)


async def run_turn(transcript: str) -> None:
    """Run one LLM->TTS turn with a bounded, always-draining lifecycle.

    The TTS synth loop blocks on a None sentinel, so it must be emitted no matter
    how the LLM ends (success, error, timeout, or an outside cancel from barge-in)
    or playback would hang forever. The LLM is bounded by a timeout so a stalled
    or hung stream can't wedge the turn. The finally owns cleanup of both child
    tasks, so cancelling this coroutine (barge-in) never orphans the LLM or TTS.
    """
    response_queue: asyncio.Queue = asyncio.Queue()
    llm_task = asyncio.create_task(llm_client.run(transcript, response_queue))
    tts_task = asyncio.create_task(tts_client.play(response_queue))

    try:
        # wait_for cancels llm_task if it overruns the budget.
        await asyncio.wait_for(llm_task, timeout=llm_cfg.response_timeout)
    except asyncio.TimeoutError:
        print(f"\n[main] LLM exceeded {llm_cfg.response_timeout}s; ending turn.")
    except Exception as e:
        print(f"\n[main] LLM error: {e!r}; ending turn.")
    finally:
        # Stop the LLM if still streaming (barge-in cancel path), then always
        # release the synth loop and drain TTS so neither child is left running.
        if not llm_task.done():
            llm_task.cancel()
        response_queue.put_nowait(None)  # unbounded queue -> cancellation-safe
        try:
            await tts_task
        except Exception as e:
            print(f"[main] TTS error: {e!r}")


async def run_turn_with_bargein(transcript: str) -> bool:
    """Run a turn while an always-on wakeword listener watches for a barge-in.

    Returns True if the user interrupted (the wakeword fired mid-turn): the turn
    is aborted (synth stopped, playback flushed, LLM cancelled) and the caller
    should go straight back to STT. Returns False if the turn finished on its own.

    NOTE: this is the *control* path only. Always-on barge-in is not shippable
    until acoustic echo cancellation (P2-4) lands -- otherwise the open mic hears
    the assistant's own playback and self-triggers (HANDOFF blind-spot #5).
    """
    barge_task = asyncio.create_task(oww_client.run())
    turn_task = asyncio.create_task(run_turn(transcript))

    done, _ = await asyncio.wait(
        {barge_task, turn_task}, return_when=asyncio.FIRST_COMPLETED
    )

    if turn_task in done:
        # Turn finished on its own; shut the barge listener down and reclaim the mic.
        oww_client.stop()
        await barge_task
        return False

    # Barge-in: the wakeword fired while the turn was still going.
    tts_client.interrupt()  # stop synth + abort playback (flush buffered audio)
    turn_task.cancel()      # run_turn's finally cancels the LLM and drains TTS
    try:
        await turn_task
    except asyncio.CancelledError:
        pass
    await barge_task        # already complete (it detected); ensures the mic is closed
    return True


async def main():
    transcript_queue: asyncio.Queue[str] = asyncio.Queue()

    skip_wakeword = False  # set after a barge-in: the wakeword already fired

    while True:
        if not skip_wakeword:
            print("\n[main] Listening for wakeword...")
            await oww_client.run()
        skip_wakeword = False

        earcon.play(earcon.WAKE_ACK)  # acknowledge the wakeword

        print("[main] Wakeword detected. Starting STT...")
        tracer.reset()
        await stt_client.run(transcript_queue)

        transcript = await transcript_queue.get()
        print(f"[main] Transcript: {transcript}")

        if not transcript:
            continue

        # Acknowledge end-of-speech; masks the LLM time-to-first-token silence.
        earcon.play(earcon.ENDPOINT_ACK)

        # Optional transcript processing might go here

        interrupted = await run_turn_with_bargein(transcript)
        tracer.report()

        if interrupted:
            print("[main] Barge-in: re-listening immediately.")
            skip_wakeword = True


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        oww_client.stop()
