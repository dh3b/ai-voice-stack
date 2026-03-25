import os
import logging
import numpy as np
from aiohttp import web
import openwakeword
from openwakeword.model import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakewordService:
    def __init__(self):
        self.model_path = os.getenv("WAKEWORD_MODEL_PATH", "/models/wakeword")
        self.threshold = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
        self.phrase = os.getenv("WAKEWORD_PHRASE", "hey_jarvis")
        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        
        # Initialize model
        logger.info(f"Loading wakeword model: {self.phrase}")
        self.model = Model(wakeword_models=[self.phrase], inference_framework='onnx')
        
    async def detect(self, request):
        """Detect wakeword in audio chunk"""
        try:
            audio_bytes = await request.read()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Predict
            prediction = self.model.predict(audio_array)
            
            # Check if wakeword detected
            detected = False
            for key, score in prediction.items():
                if score >= self.threshold:
                    detected = True
                    logger.info(f"Wakeword detected: {key} (score: {score})")
                    break
            
            return web.json_response({"detected": detected})
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health(self, request):
        return web.json_response({"status": "healthy"})

async def create_app():
    app = web.Application()
    service = WakewordService()
    
    app.router.add_post('/detect', service.detect)
    app.router.add_get('/health', service.health)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8001)
