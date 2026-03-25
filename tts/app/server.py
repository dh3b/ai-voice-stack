import os
import logging
import io
import wave
from aiohttp import web
from piper import PiperVoice
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self):
        self.model_path = os.getenv("TTS_MODEL_PATH", "/models/tts")
        self.voice = os.getenv("TTS_VOICE", "en_US-lessac-medium")
        self.speaker_id = int(os.getenv("TTS_SPEAKER_ID", "0"))
        self.sample_rate = int(os.getenv("TTS_SAMPLE_RATE", "22050"))
        self.noise_scale = float(os.getenv("TTS_NOISE_SCALE", "0.667"))
        self.length_scale = float(os.getenv("TTS_LENGTH_SCALE", "1.0"))
        
        logger.info(f"Loading TTS model: {self.voice}")
        
        # Load Piper voice
        model_file = os.path.join(self.model_path, f"{self.voice}.onnx")
        config_file = os.path.join(self.model_path, f"{self.voice}.onnx.json")
        
        if os.path.exists(model_file):
            self.voice_model = PiperVoice.load(model_file, config_file)
            logger.info("TTS model loaded successfully")
        else:
            logger.warning(f"Model file not found: {model_file}")
            self.voice_model = None
    
    async def synthesize(self, request):
        """Synthesize speech from text"""
        try:
            data = await request.json()
            text = data.get("text", "")
            
            if not text:
                return web.json_response({"error": "No text provided"}, status=400)
            
            if not self.voice_model:
                return web.json_response({"error": "TTS model not loaded"}, status=500)
            
            # Synthesize
            audio_stream = io.BytesIO()
            
            with wave.open(audio_stream, 'wb') as wav_file:
                self.voice_model.synthesize(
                    text,
                    wav_file,
                    speaker_id=self.speaker_id,
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale
                )
            
            audio_stream.seek(0)
            audio_data = audio_stream.read()
            
            return web.Response(
                body=audio_data,
                content_type='audio/wav',
                headers={'Content-Disposition': 'attachment; filename="speech.wav"'}
            )
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def synthesize_stream(self, request):
        """Stream synthesis for real-time TTS"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'audio/wav'
        await response.prepare(request)
        
        try:
            # Read streaming text input
            text_buffer = ""
            sentence_endings = ['.', '!', '?', '\n']
            
            async for chunk in request.content.iter_any():
                if not chunk:
                    break
                
                text_buffer += chunk.decode('utf-8')
                
                # Process complete sentences
                while any(ending in text_buffer for ending in sentence_endings):
                    # Find first sentence ending
                    min_idx = len(text_buffer)
                    for ending in sentence_endings:
                        idx = text_buffer.find(ending)
                        if idx != -1 and idx < min_idx:
                            min_idx = idx
                    
                    if min_idx < len(text_buffer):
                        sentence = text_buffer[:min_idx + 1].strip()
                        text_buffer = text_buffer[min_idx + 1:]
                        
                        if sentence and self.voice_model:
                            # Synthesize sentence
                            audio_stream = io.BytesIO()
                            with wave.open(audio_stream, 'wb') as wav_file:
                                self.voice_model.synthesize(
                                    sentence,
                                    wav_file,
                                    speaker_id=self.speaker_id,
                                    length_scale=self.length_scale,
                                    noise_scale=self.noise_scale
                                )
                            
                            audio_stream.seek(0)
                            await response.write(audio_stream.read())
            
            # Process remaining text
            if text_buffer.strip() and self.voice_model:
                audio_stream = io.BytesIO()
                with wave.open(audio_stream, 'wb') as wav_file:
                    self.voice_model.synthesize(
                        text_buffer.strip(),
                        wav_file,
                        speaker_id=self.speaker_id,
                        length_scale=self.length_scale,
                        noise_scale=self.noise_scale
                    )
                
                audio_stream.seek(0)
                await response.write(audio_stream.read())
            
        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
        finally:
            await response.write_eof()
        
        return response
    
    async def health(self, request):
        return web.json_response({"status": "healthy"})

async def create_app():
    app = web.Application()
    service = TTSService()
    
    app.router.add_post('/synthesize', service.synthesize)
    app.router.add_post('/synthesize_stream', service.synthesize_stream)
    app.router.add_get('/health', service.health)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8004)
