"""
Drives the real prediction path (Model.predict over 1280-sample int16 chunks), mic
bypassed, on Piper-synthesized speech plus seeded noise and silence fixtures.
"""

import pytest

from config import OWWClientConfig
from modules.client.oww_client import OWWClient


def _max_score(model, samples, chunk):
    best = 0.0
    for start in range(0, len(samples) - chunk + 1, chunk):
        model.predict(samples[start:start + chunk])
        best = max(best, max(list(buf)[-1] for buf in model.prediction_buffer.values()))
    return best


@pytest.mark.smoke
def test_wakeword_fires_and_stays_silent(wakeword_model_path, smoke):
    cfg = OWWClientConfig(model_paths=[wakeword_model_path])

    def score(name):
        # Fresh model per clip so prediction buffers don't bleed across files.
        client = OWWClient(cfg)
        samples = smoke.read_wav_int16(smoke.fixtures_dir / name)
        return _max_score(client._model, samples, cfg.chunk_size)

    assert score("hey_jarvis.wav") > cfg.threshold
    assert score("white_noise.wav") < cfg.threshold
    assert score("silence.wav") < cfg.threshold
