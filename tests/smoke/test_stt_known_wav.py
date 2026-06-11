"""
Bypasses the mic and streams a fixture WAV's PCM over TCP in block_size chunks exactly as
STTClient._stream_mic_to_server would, then reads via the real STTClient._read_transcripts
until an is_final line arrives and fuzzy-matches the known phrase (tiny model -> tolerant).
"""

import asyncio

import numpy as np
import pytest

from config import STTClientConfig
from modules.client.stt_client import STTClient


@pytest.mark.smoke
async def test_stt_known_wav(stt_server_config, smoke):
    cfg = STTClientConfig(
        server_host=stt_server_config.server_host,
        server_port=stt_server_config.server_port,
    )
    expected = (smoke.fixtures_dir / "stt_phrase.txt").read_text(encoding="utf-8").strip()
    speech = smoke.read_wav_int16(smoke.fixtures_dir / "stt_phrase.wav")
    # Trailing silence so the server's VAD endpoints the utterance and emits is_final.
    silence = np.zeros(int(1.5 * cfg.sample_rate), dtype=np.int16)
    pcm = np.concatenate([speech, silence]).astype(np.int16).tobytes()

    reader, writer = await asyncio.open_connection(cfg.server_host, cfg.server_port)
    transcript_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    client = STTClient(cfg)
    reader_task = asyncio.create_task(
        client._read_transcripts(reader, transcript_queue, stop_event, listen_timeout=30.0)
    )

    block_bytes = cfg.block_size * 2  # int16 -> 2 bytes/sample
    for i in range(0, len(pcm), block_bytes):
        writer.write(pcm[i:i + block_bytes])
        await writer.drain()
        await asyncio.sleep(0.02)  # pace the stream so the VAD behaves

    try:
        await asyncio.wait_for(reader_task, timeout=40.0)
    finally:
        if not reader_task.done():
            reader_task.cancel()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    transcript = transcript_queue.get_nowait() if not transcript_queue.empty() else ""
    assert transcript, "no is_final transcript arrived"
    assert smoke.fuzzy_match(expected, transcript), f"{transcript!r} did not match {expected!r}"
