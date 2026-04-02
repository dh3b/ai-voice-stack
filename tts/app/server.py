"""
TTS service using piper CLI (subprocess).

Endpoint:
  POST /synthesize  — JSON {"text": "..."} -> WAV audio bytes
  GET  /health
"""

import asyncio
import logging
import os
import tempfile

from aiohttp import web

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
        self.speaker_id = os.getenv("TTS_SPEAKER_ID", "0")
        self.noise_scale = os.getenv("TTS_NOISE_SCALE", "0.667")
        self.length_scale = os.getenv("TTS_LENGTH_SCALE", "1.0")

        self.model_file = os.path.join(self.model_dir, f"{self.voice_name}.onnx")
        config_file = os.path.join(self.model_dir, f"{self.voice_name}.onnx.json")
        self.config_file = config_file if os.path.exists(config_file) else None

        logger.info("Loading TTS  model=%s", self.model_file)

        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"TTS model not found: {self.model_file}")

        logger.info("TTS ready")

    def _build_cmd(self, output_path: str) -> list[str]:
        cmd = [
            "piper",
            "--model", self.model_file,
            "--output_file", output_path,
            "--speaker", self.speaker_id,
            "--length_scale", self.length_scale,
            "--noise_scale", self.noise_scale,
        ]
        if self.config_file:
            cmd += ["--config", self.config_file]
        return cmd

    async def synthesize(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            text = data.get("text", "").strip()

            if not text:
                return web.json_response({"error": "No text provided"}, status=400)

            logger.info("TTS <- '%s'", text)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                cmd = self._build_cmd(tmp_path)
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate(input=text.encode("utf-8"))

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Piper exited {proc.returncode}: {stderr.decode(errors='replace')}"
                    )

                with open(tmp_path, "rb") as f:
                    wav_bytes = f.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

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
