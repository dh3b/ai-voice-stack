import os
import logging

import numpy as np
from aiohttp import web
from openwakeword.model import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wakeword")

# Suppress aiohttp access log spam from /detect polling
logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


class WakewordService:
    def __init__(self):
        self.model_path = os.getenv("WAKEWORD_MODEL_PATH", "")
        self.threshold = float(os.getenv("WAKEWORD_THRESHOLD", "0.3"))
        self.framework = os.getenv("WAKEWORD_FRAMEWORK", "onnx")

        logger.info(
            "Loading wake-word model  path=%s  framework=%s  threshold=%.2f",
            self.model_path, self.framework, self.threshold,
        )

        # If model_path points to a file, load it directly.
        # Otherwise treat it as a built-in model name (e.g. "hey_jarvis").
        wakeword_models = [self.model_path] if self.model_path else []
        self.model = Model(
            wakeword_models=wakeword_models,
            inference_framework=self.framework,
        )
        logger.info("Wake-word model ready")

    async def detect(self, request: web.Request) -> web.Response:
        try:
            audio_bytes = await request.read()
            # Orchestrator sends int16 bytes
            pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            predictions: dict[str, float] = self.model.predict(pcm_int16)

            detected = False
            for model_name, score in predictions.items():
                # Log scores at DEBUG for diagnostics
                logger.debug("predict  model=%s  score=%.4f", model_name, score)

                # Log at INFO when score approaches threshold (>50% of threshold)
                if score >= self.threshold * 0.5 and score < self.threshold:
                    logger.info(
                        "Near detection  model=%s  score=%.3f  threshold=%.2f",
                        model_name, score, self.threshold,
                    )

                if score >= self.threshold:
                    logger.info(
                        "Wake word detected  model=%s  score=%.3f",
                        model_name, score,
                    )
                    detected = True
                    self.model.reset()
                    break

            return web.json_response({"detected": detected})

        except Exception as e:
            logger.error("Detection error: %s", e)
            return web.json_response({"error": str(e)}, status=500)

    async def reset(self, request: web.Request) -> web.Response:
        self.model.reset()
        return web.json_response({"status": "reset"})

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})


async def create_app() -> web.Application:
    app = web.Application()
    service = WakewordService()

    app.router.add_post("/detect", service.detect)
    app.router.add_post("/reset", service.reset)
    app.router.add_get("/health", service.health)

    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8001)
