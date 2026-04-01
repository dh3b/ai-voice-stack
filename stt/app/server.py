import os
import logging

import numpy as np
from aiohttp import web
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stt")


class STTService:
    def __init__(self):
        self.model_path = os.getenv("WHISPER_MODEL_PATH", "/models/whisper/")
        self.language = os.getenv("WHISPER_LANGUAGE", "en")
        self.device = os.getenv("WHISPER_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        logger.info(
            "Loading Whisper  path=%s  device=%s  compute_type=%s",
            self.model_path, self.device, self.compute_type,
        )
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper ready")

    async def transcribe(self, request: web.Request) -> web.Response:
        try:
            audio_bytes = await request.read()
            # Orchestrator sends float32 audio bytes
            audio = np.frombuffer(audio_bytes, dtype=np.float32)

            logger.info("Transcribing %.2f s of audio", len(audio) / 16000)

            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()

            logger.info(
                "Transcription [lang=%s  prob=%.2f]: '%s'",
                info.language, info.language_probability, text or "<empty>",
            )

            return web.json_response({
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
            })

        except Exception as e:
            logger.error("Transcription error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})


async def create_app() -> web.Application:
    app = web.Application(client_max_size=50 * 1024 * 1024)  # 50 MB
    service = STTService()

    app.router.add_post("/transcribe", service.transcribe)
    app.router.add_get("/health", service.health)

    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8002)
