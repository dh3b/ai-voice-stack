import os
import logging
import numpy as np
import tempfile
from aiohttp import web
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        self.model_path = os.getenv("STT_MODEL_PATH", "/models/stt")
        self.model_size = os.getenv("STT_MODEL_SIZE", "base")
        self.language = os.getenv("STT_LANGUAGE", "en")
        self.device = os.getenv("STT_DEVICE", "cpu")
        self.compute_type = os.getenv("STT_COMPUTE_TYPE", "int8")
        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        
        logger.info(f"Loading Whisper model: {self.model_size}")
        
        # Try to load from custom path, fallback to model size
        try:
            if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
                self.model = WhisperModel(
                    self.model_path,
                    device=self.device,
                    compute_type=self.compute_type
                )
            else:
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
        except Exception as e:
            logger.warning(f"Failed to load from path, using model size: {e}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
        
        logger.info("STT model loaded successfully")
    
    async def transcribe(self, request):
        """Transcribe audio to text"""
        try:
            audio_bytes = await request.read()
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_array,
                language=self.language,
                beam_size=5,
                vad_filter=True
            )
            
            # Collect text
            text = " ".join([segment.text for segment in segments]).strip()
            
            logger.info(f"Transcribed: {text}")
            
            return web.json_response({
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability
            })
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health(self, request):
        return web.json_response({"status": "healthy"})

async def create_app():
    app = web.Application(client_max_size=50*1024*1024)  # 50MB max
    service = STTService()
    
    app.router.add_post('/transcribe', service.transcribe)
    app.router.add_get('/health', service.health)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8002)
