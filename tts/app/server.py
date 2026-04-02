"""
TTS service using PiperVoice Python API.

Endpoint:
  POST /synthesize  — JSON {"text": "..."} -> WAV audio bytes
  GET  /health
"""

import io
import logging
import os
import wave

from aiohttp import web
from piper import PiperVoice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tts")


class TTSService:
    def __init__(self):
        self.model_dir = os.getenv("TTS_MODEL_PATH", "/models/tts")
        self.voice_name = os.getenv("TTS_VOICE", "en_US-lessac-medium")
        self.speaker_id = int(os.getenv("TTS_SPEAKER_ID", "0"))
        self.noise_scale = float(os.getenv("TTS_NOISE_SCALE", "0.667"))
        self.length_scale = float(os.getenv("TTS_LENGTH_SCALE", "1.0"))

        model_file = os.path.join(self.model_dir, f"{self.voice_name}.onnx")
        config_file = os.path.join(self.model_dir, f"{self.voice_name}.onnx.json")

        logger.info("Loading TTS  model=%s", model_file)

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"TTS model not found: {model_file}")

        config_path = config_file if os.path.exists(config_file) else None
        self.voice = PiperVoice.load(model_file, config_path)
        logger.info("TTS ready")

    async def synthesize(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            text = data.get("text", "").strip()

            if not text:
                return web.json_response({"error": "No text provided"}, status=400)

            logger.info("TTS <- '%s'", text)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                wav_file.setframerate(self.voice.config.sample_rate)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setnchannels(1)  # mono
                self.voice.synthesize(
                    text,
                    wav_file,
                    speaker_id=self.speaker_id,
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale,
                )

            wav_bytes = buf.getvalue()
            logger.info("TTS -> %d bytes", len(wav_bytes))

            return web.Response(
                body=wav_bytes,
                content_type="audio/wav",
            )

        except Exception as e:
            logger.error("Synthesis error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})


async def create_app() -> web.Application:
    app = web.Application()
    service = TTSService()

    app.router.add_post("/synthesize", service.synthesize)
    app.router.add_get("/health", service.health)

    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8004)
