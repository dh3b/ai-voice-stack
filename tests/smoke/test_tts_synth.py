"""
Goes through the real TTSClient._synthesize + _parse_wav against a live Piper server.
"""

import httpx
import numpy as np
import pytest

from config import TTSClientConfig


@pytest.mark.smoke
async def test_tts_synth_nonsilent(tts_server_config, make_tts_client, smoke):
    cfg = tts_server_config
    client = make_tts_client(
        TTSClientConfig(server_host=cfg.server_host, server_port=cfg.server_port)
    )
    short = "Hello there, this is a short test sentence."
    long = short + " " + short + " " + short  # ~3x the words

    async with httpx.AsyncClient() as http:
        wav_short = await client._synthesize(http, short)
        wav_long = await client._synthesize(http, long)

    assert wav_short is not None and wav_long is not None

    data, sr, channels = client._parse_wav(wav_short)
    assert channels == 1
    assert 8000 <= sr <= 48000, f"implausible sample rate {sr}"
    samples = np.frombuffer(data, dtype=np.int16)
    assert smoke.rms(samples) > 200.0, "synthesized audio is essentially silent"

    data_long, sr_long, _ = client._parse_wav(wav_long)
    dur_short = len(samples) / sr
    dur_long = len(np.frombuffer(data_long, dtype=np.int16)) / sr_long
    assert dur_long > 1.8 * dur_short, f"{dur_long:.2f}s not ~3x {dur_short:.2f}s"
