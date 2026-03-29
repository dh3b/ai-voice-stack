import asyncio
import aiohttp
import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
import random
import os
import re
import json
import io
from typing import Optional, List, Tuple, Any
from app.config import Config
from app.tool_registry import registry

logger = logging.getLogger(__name__)

class VoiceAssistantPipeline:
    def __init__(self):
        self.config = Config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.pipeline_active = False
        self.wakeword_task: Optional[asyncio.Task] = None
        self.pipeline_task: Optional[asyncio.Task] = None
        self.playback_task: Optional[asyncio.Task] = None
        self.cancel_event = asyncio.Event()
        
        # Audio playback queue
        self.audio_queue: asyncio.Queue[Optional[Tuple[np.ndarray, int]]] = asyncio.Queue()
        
    async def start(self):
        """Start the voice assistant pipeline"""
        self.session = aiohttp.ClientSession()
        
        # Pre-check services
        if not await self._wait_for_services():
            logger.error("Required services not reachable. Exiting.")
            return

        self.running = True
        logger.info("Voice Assistant Pipeline started")
        
        # Generate confirmations if needed
        await self._ensure_confirmations()
        
        # Start playback worker
        self.playback_task = asyncio.create_task(self._playback_worker())
        
        # Start wakeword detection
        self.wakeword_task = asyncio.create_task(self._wakeword_loop())
        
        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop the pipeline"""
        self.running = False
        
        if self.wakeword_task:
            self.wakeword_task.cancel()
        if self.pipeline_task:
            self.pipeline_task.cancel()
        if self.playback_task:
            self.playback_task.cancel()
            
        if self.session:
            await self.session.close()
        
        logger.info("Voice Assistant Pipeline stopped")

    async def _wait_for_services(self, timeout: int = 30):
        """Wait for required services to be healthy"""
        if not self.session:
            return False

        services = {
            "Wakeword": self.config.WAKEWORD_SERVICE_URL,
            "STT": self.config.STT_SERVICE_URL,
            "LLM": self.config.LLM_SERVICE_URL,
            "TTS": self.config.TTS_SERVICE_URL
        }
        
        logger.info("Waiting for core services to be healthy...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            all_healthy = True
            for name, url in services.items():
                try:
                    async with self.session.get(f"{url}/health", timeout=2) as resp:
                        if resp.status != 200:
                            all_healthy = False
                            break
                except Exception:
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("All services are healthy!")
                return True
            
            await asyncio.sleep(2)
            
        return False

    async def _ensure_confirmations(self):
        """Ensure confirmation audio files exist"""
        if not self.session:
            return

        os.makedirs(self.config.CONFIRMATIONS_PATH, exist_ok=True)
        for phrase in self.config.CONFIRMATION_PHRASES:
            path = os.path.join(self.config.CONFIRMATIONS_PATH, f"{phrase}.wav")
            if not os.path.exists(path):
                try:
                    async with self.session.post(
                        f"{self.config.TTS_SERVICE_URL}/synthesize",
                        json={"text": phrase}
                    ) as resp:
                        if resp.status == 200:
                            with open(path, 'wb') as f:
                                f.write(await resp.read())
                            logger.info(f"Generated confirmation: {phrase}")
                except Exception as e:
                    logger.warning(f"Failed to generate confirmation '{phrase}': {e}")

    async def _playback_worker(self):
        """Consume audio chunks from queue and play them sequentially"""
        while self.running:
            try:
                item = await self.audio_queue.get()
                if item is None:
                    self.audio_queue.task_done()
                    continue
                
                audio_data, sample_rate = item
                
                # If we're cancelled, clear the rest of the queue
                if self.cancel_event.is_set():
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    self.audio_queue.task_done()
                    continue
                
                # Play audio chunk
                sd.play(audio_data, samplerate=sample_rate, device=self.config.AUDIO_OUTPUT_DEVICE)
                sd.wait() # Wait for this chunk to finish playing
                
                self.audio_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Playback worker error: {e}")

    async def _wakeword_loop(self):
        """Continuously listen for wakeword"""
        while self.running:
            try:
                audio_chunk = await self._capture_audio_chunk()
                detected = await self._detect_wakeword(audio_chunk)
                
                if detected:
                    logger.info("Wakeword detected!")
                    self.cancel_event.set()
                    sd.stop() # Stop any current playback
                    
                    # Clear playback queue
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.task_done()
                        except asyncio.QueueEmpty:
                            break

                    self.cancel_event.clear()
                    if self.pipeline_task:
                        self.pipeline_task.cancel()
                    
                    self.pipeline_task = asyncio.create_task(self._run_pipeline())
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in wakeword loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _run_pipeline(self):
        """Run the full voice assistant pipeline with true streaming"""
        self.pipeline_active = True
        try:
            # Play confirmation
            phrase = random.choice(self.config.CONFIRMATION_PHRASES)
            path = os.path.join(self.config.CONFIRMATIONS_PATH, f"{phrase}.wav")
            if os.path.exists(path):
                data, sr = sf.read(path)
                await self.audio_queue.put((data, sr))

            # Record
            audio_data = await self._record_user_speech()
            if audio_data is None or self.cancel_event.is_set():
                return
            
            # Transcribe
            text = await self._transcribe(audio_data)
            if not text or self.cancel_event.is_set():
                return
            
            logger.info(f"UTTERANCE: {text}")
            
            # Start streaming synthesis and playback
            await self._stream_respond(text)
            
            # Chat continuation
            if self.config.CHAT_CONTINUATION_ENABLED and not self.cancel_event.is_set():
                await self._handle_chat_continuation()
                
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        except Exception as e:
            logger.error(f"Error in pipeline: {e}", exc_info=True)
        finally:
            self.pipeline_active = False

    async def _stream_respond(self, user_text: str):
        """Coordination of LLM -> Sentences -> TTS -> Audio"""
        sentence_queue = asyncio.Queue()
        
        # Parallel tasks
        llm_task = asyncio.create_task(self._llm_to_sentences(user_text, sentence_queue))
        tts_task = asyncio.create_task(self._sentences_to_tts(sentence_queue))
        
        await asyncio.gather(llm_task, tts_task)

    async def _llm_to_sentences(self, user_text: str, sentence_queue: asyncio.Queue):
        """Stream tokens from LLM and push sentences to queue"""
        if not self.session:
            return

        tools = registry.get_tool_definitions()
        buffer = ""
        sentence_end_patterns = re.compile(r'([.!?])\s')

        try:
            async with self.session.post(
                f"{self.config.LLM_SERVICE_URL}/generate",
                json={"text": user_text, "tools": tools, "stream": True}
            ) as resp:
                if resp.status != 200:
                    logger.error(f"LLM error: {resp.status}")
                    return

                async for line in resp.content:
                    if self.cancel_event.is_set():
                        break
                    
                    line_text = line.decode('utf-8').strip()
                    if not line_text.startswith('data: '):
                        continue
                        
                    token = line_text[6:]
                    if token == '[DONE]':
                        break
                    
                    if token.startswith('TOOL_CALL:'):
                        tool_result = await self._handle_tool_call(token[10:])
                        if tool_result:
                            logger.info(f"Tool Result: {tool_result}")
                        continue
                    
                    if token.startswith('ERROR:'):
                        logger.error(f"LLM stream error: {token[6:]}")
                        break
                    
                    buffer += token
                    
                    # Look for sentence boundaries
                    while True:
                        match = sentence_end_patterns.search(buffer)
                        if not match:
                            break
                        
                        sentence = buffer[:match.end()].strip()
                        buffer = buffer[match.end():]
                        if sentence:
                            await sentence_queue.put(sentence)
                
                # Final flush
                if buffer.strip():
                    await sentence_queue.put(buffer.strip())
                    
        finally:
            await sentence_queue.put(None) # Sentinel

    async def _sentences_to_tts(self, sentence_queue: asyncio.Queue):
        """Consume sentences and send to TTS service"""
        if not self.session:
            return

        while True:
            sentence = await sentence_queue.get()
            if sentence is None or self.cancel_event.is_set():
                break
            
            try:
                logger.info(f"TTS Synthesis: {sentence}")
                async with self.session.post(
                    f"{self.config.TTS_SERVICE_URL}/synthesize",
                    json={"text": sentence}
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        data, sr = sf.read(io.BytesIO(content))
                        await self.audio_queue.put((data.astype(np.float32), sr))
            except Exception as e:
                logger.error(f"TTS error for '{sentence}': {e}")
            finally:
                sentence_queue.task_done()

    async def _handle_tool_call(self, tool_data: str) -> str:
        """Handle tool execution and return result"""
        try:
            data = json.loads(tool_data)
            tool_name = data.get("name")
            arguments = data.get("arguments", {})
            
            if tool_name:
                logger.info(f"Executing tool '{tool_name}' with {arguments}")
                result = await registry.execute_tool(tool_name, arguments)
                return str(result)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
        return ""

    async def _capture_audio_chunk(self, duration: float = 0.08) -> np.ndarray:
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
        if not self.session:
            return False

        try:
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            async with self.session.post(
                f"{self.config.WAKEWORD_SERVICE_URL}/detect",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("detected", False)
        except Exception:
            pass
        return False

    async def _record_user_speech(self) -> Optional[np.ndarray]:
        logger.info("Listening...")
        frames = []
        silence_frames = 0
        max_silence_frames = int(self.config.SILENCE_DURATION * self.config.AUDIO_SAMPLE_RATE / 1024)
        
        while self.running and not self.cancel_event.is_set():
            chunk = sd.rec(
                1024,
                samplerate=self.config.AUDIO_SAMPLE_RATE,
                channels=self.config.AUDIO_CHANNELS,
                device=self.config.AUDIO_INPUT_DEVICE,
                dtype='float32'
            )
            sd.wait()
            frames.append(chunk)
            
            if np.abs(chunk).mean() < self.config.SILENCE_THRESHOLD:
                silence_frames += 1
                if silence_frames >= max_silence_frames and len(frames) > 10:
                    break
            else:
                silence_frames = 0
                
            if len(frames) * 1024 / self.config.AUDIO_SAMPLE_RATE > self.config.MAX_RECORDING_DURATION:
                break
        
        if not frames:
            return None
        return np.concatenate(frames, axis=0)
    
    async def _transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        if not self.session:
            return None

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

    async def _handle_chat_continuation(self):
        """Wait for follow-up speech within timeout"""
        logger.info("Waiting for continuation...")
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < self.config.CHAT_CONTINUATION_TIMEOUT:
            if self.cancel_event.is_set():
                break
            
            chunk = await self._capture_audio_chunk(0.1)
            if np.abs(chunk).mean() > self.config.SILENCE_THRESHOLD * 1.5: # Slightly higher sensitivity
                logger.info("Follow-up detected!")
                audio_data = await self._record_user_speech()
                if audio_data is not None:
                    text = await self._transcribe(audio_data)
                    if text:
                        logger.info(f"UTTERANCE (cont): {text}")
                        await self._stream_respond(text)
                        # Refresh timeout
                        start_time = asyncio.get_event_loop().time()
                break
            await asyncio.sleep(0.05)
