import asyncio
import aiohttp
import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
import random
import os
from typing import Optional
from app.config import Config
from app.tool_registry import registry

logger = logging.getLogger(__name__)

async def generate_confirmations():
    """Generate confirmation audio files using TTS"""
    os.makedirs(Config.CONFIRMATIONS_PATH, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        for phrase in Config.CONFIRMATION_PHRASES:
            output_path = os.path.join(Config.CONFIRMATIONS_PATH, f"{phrase}.wav")
            if os.path.exists(output_path):
                continue
            
            try:
                async with session.post(
                    f"{Config.TTS_SERVICE_URL}/synthesize",
                    json={"text": phrase}
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        with open(output_path, 'wb') as f:
                            f.write(audio_data)
                        logger.info(f"Generated confirmation: {phrase}")
            except Exception as e:
                logger.error(f"Failed to generate confirmation '{phrase}': {e}")

class VoiceAssistantPipeline:
    def __init__(self):
        self.config = Config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.pipeline_active = False
        self.wakeword_task: Optional[asyncio.Task] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        self.cancel_event = asyncio.Event()
        
    async def start(self):
        """Start the voice assistant pipeline"""
        self.session = aiohttp.ClientSession()
        self.running = True
        
        logger.info("Voice Assistant Pipeline started")
        logger.info(f"Audio Input Device: {self.config.AUDIO_INPUT_DEVICE}")
        logger.info(f"Audio Output Device: {self.config.AUDIO_OUTPUT_DEVICE}")
        
        # Start wakeword detection
        self.wakeword_task = asyncio.create_task(self._wakeword_loop())
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop the pipeline"""
        self.running = False
        
        if self.wakeword_task:
            self.wakeword_task.cancel()
        if self.pipeline_task:
            self.pipeline_task.cancel()
        
        if self.session:
            await self.session.close()
        
        logger.info("Voice Assistant Pipeline stopped")
    
    async def _wakeword_loop(self):
        """Continuously listen for wakeword"""
        while self.running:
            try:
                audio_chunk = await self._capture_audio_chunk()
                
                # Send to wakeword service
                detected = await self._detect_wakeword(audio_chunk)
                
                if detected:
                    logger.info("Wakeword detected!")
                    
                    # Cancel any ongoing pipeline
                    if self.pipeline_active:
                        self.cancel_event.set()
                        if self.pipeline_task:
                            self.pipeline_task.cancel()
                            try:
                                await self.pipeline_task
                            except asyncio.CancelledError:
                                pass
                    
                    # Start new pipeline
                    self.cancel_event.clear()
                    self.pipeline_task = asyncio.create_task(self._run_pipeline())
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in wakeword loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _run_pipeline(self):
        """Run the full voice assistant pipeline"""
        self.pipeline_active = True
        
        try:
            # Play confirmation
            await self._play_confirmation()
            
            # Record user speech
            audio_data = await self._record_user_speech()
            if audio_data is None or self.cancel_event.is_set():
                return
            
            # Transcribe
            text = await self._transcribe(audio_data)
            if not text or self.cancel_event.is_set():
                return
            
            logger.info(f"User said: {text}")
            
            # Generate LLM response (streaming)
            response_task = asyncio.create_task(self._generate_and_speak(text))
            await response_task
            
            # Chat continuation
            if self.config.CHAT_CONTINUATION_ENABLED and not self.cancel_event.is_set():
                await self._handle_chat_continuation()
                
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
        finally:
            self.pipeline_active = False
    
    async def _capture_audio_chunk(self, duration: float = 0.08) -> np.ndarray:
        """Capture a small audio chunk for wakeword detection"""
        frames = int(duration * self.config.AUDIO_SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=self.config.AUDIO_SAMPLE_RATE,
            channels=self.config.AUDIO_CHANNELS,
            device=self.config.AUDIO_INPUT_DEVICE,
            dtype='float32'
        )
        sd.wait()
        return audio
    
    async def _detect_wakeword(self, audio_chunk: np.ndarray) -> bool:
        """Send audio to wakeword service"""
        try:
            # Convert to bytes
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            
            async with self.session.post(
                f"{self.config.WAKEWORD_SERVICE_URL}/detect",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("detected", False)
        except Exception as e:
            logger.error(f"Wakeword detection error: {e}")
        return False
    
    async def _play_confirmation(self):
        """Play random confirmation sound"""
        try:
            phrase = random.choice(self.config.CONFIRMATION_PHRASES)
            audio_path = os.path.join(self.config.CONFIRMATIONS_PATH, f"{phrase}.wav")
            
            if os.path.exists(audio_path):
                data, samplerate = sf.read(audio_path)
                sd.play(data, samplerate, device=self.config.AUDIO_OUTPUT_DEVICE)
                sd.wait()
        except Exception as e:
            logger.error(f"Error playing confirmation: {e}")
    
    async def _record_user_speech(self) -> Optional[np.ndarray]:
        """Record user speech until silence"""
        logger.info("Recording user speech...")
        
        frames = []
        silence_frames = 0
        max_silence_frames = int(self.config.SILENCE_DURATION * self.config.AUDIO_SAMPLE_RATE / 1024)
        max_frames = int(self.config.MAX_RECORDING_DURATION * self.config.AUDIO_SAMPLE_RATE / 1024)
        
        for _ in range(max_frames):
            if self.cancel_event.is_set():
                return None
            
            chunk = sd.rec(
                1024,
                samplerate=self.config.AUDIO_SAMPLE_RATE,
                channels=self.config.AUDIO_CHANNELS,
                device=self.config.AUDIO_INPUT_DEVICE,
                dtype='float32'
            )
            sd.wait()
            
            frames.append(chunk)
            
            # Check for silence
            if np.abs(chunk).mean() < self.config.SILENCE_THRESHOLD:
                silence_frames += 1
                if silence_frames >= max_silence_frames and len(frames) > 10:
                    break
            else:
                silence_frames = 0
        
        if not frames:
            return None
        
        return np.concatenate(frames, axis=0)
    
    async def _transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using STT service"""
        try:
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            async with self.session.post(
                f"{self.config.STT_SERVICE_URL}/transcribe",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
        return None
    
    async def _generate_and_speak(self, text: str):
        """Generate LLM response and stream to TTS"""
        try:
            # Get tool definitions
            tools = registry.get_tool_definitions()
            
            async with self.session.post(
                f"{self.config.LLM_SERVICE_URL}/generate",
                json={"text": text, "tools": tools, "stream": True}
            ) as llm_resp:
                if llm_resp.status != 200:
                    return
                
                # Collect response for TTS
                response_text = ""
                
                async for line in llm_resp.content:
                    if self.cancel_event.is_set():
                        break
                    
                    if line:
                        decoded = line.decode('utf-8').strip()
                        if decoded.startswith('data: '):
                            token = decoded[6:]
                            
                            # Check for done signal
                            if token == '[DONE]':
                                break
                            
                            # Check for tool calls
                            if token.startswith('TOOL_CALL:'):
                                tool_result = await self._handle_tool_call(token[10:])
                                if tool_result:
                                    response_text += f" {tool_result}"
                                continue
                            
                            # Check for errors
                            if token.startswith('ERROR:'):
                                logger.error(f"LLM error: {token[6:]}")
                                continue
                            
                            # Accumulate response text
                            response_text += token
                
                # Synthesize complete response
                if response_text.strip() and not self.cancel_event.is_set():
                    await self._synthesize_and_play(response_text.strip())
                    
        except asyncio.CancelledError:
            logger.info("Generation cancelled")
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
    
    async def _synthesize_and_play(self, text: str):
        """Synthesize text and play audio"""
        try:
            async with self.session.post(
                f"{self.config.TTS_SERVICE_URL}/synthesize",
                json={"text": text}
            ) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    
                    # Save to temp file and play
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        tmp.write(audio_data)
                        tmp_path = tmp.name
                    
                    try:
                        data, samplerate = sf.read(tmp_path)
                        sd.play(data, samplerate, device=self.config.AUDIO_OUTPUT_DEVICE)
                        sd.wait()
                    finally:
                        import os
                        os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    async def _handle_tool_call(self, tool_data: str) -> str:
        """Handle tool execution and return result"""
        import json
        try:
            data = json.loads(tool_data)
            
            # Check if it's a tool call response
            if data.get("tool_call"):
                tool_name = data.get("name")
                arguments = data.get("arguments", {})
                
                result = await registry.execute_tool(tool_name, arguments)
                logger.info(f"Tool '{tool_name}' executed: {result}")
                return str(result)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
        return ""
    
    async def _handle_chat_continuation(self):
        """Handle chat continuation after response"""
        logger.info("Waiting for chat continuation...")
        
        # Wait for timeout or speech detection
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < self.config.CHAT_CONTINUATION_TIMEOUT:
            if self.cancel_event.is_set():
                return
            
            chunk = await self._capture_audio_chunk(0.1)
            
            if np.abs(chunk).mean() > self.config.SILENCE_THRESHOLD:
                logger.info("Speech detected, continuing conversation...")
                
                # Record and process
                audio_data = await self._record_user_speech()
                if audio_data is None:
                    return
                
                text = await self._transcribe(audio_data)
                if text:
                    logger.info(f"User continued: {text}")
                    await self._generate_and_speak(text)
                    
                    # Recursive continuation
                    await self._handle_chat_continuation()
                return
            
            await asyncio.sleep(0.1)
