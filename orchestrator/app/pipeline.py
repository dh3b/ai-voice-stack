"""
Async voice-assistant pipeline.

Orchestrates: wakeword -> record -> STT -> LLM -> TTS -> playback
with streaming, cancellation, and chat continuation.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
from functools import partial
from typing import Optional, Tuple

import aiohttp
import numpy as np
import sounddevice as sd
import soundfile as sf

from app.config import Config

logger = logging.getLogger("pipeline")

SENTENCE_BOUNDARY = re.compile(r"([.!?])\s")


class VoiceAssistantPipeline:
    def __init__(self):
        self.config = Config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False

        # Cancellation signal — set when wakeword fires during active pipeline
        self.cancel_event = asyncio.Event()

        # Audio playback queue: (ndarray, sample_rate) or None sentinel
        self.audio_queue: asyncio.Queue[Optional[Tuple[np.ndarray, int]]] = asyncio.Queue()

        # Task handles
        self._wakeword_task: Optional[asyncio.Task] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None

        # Persistent input stream (like rpi4ai)
        self._input_stream: Optional[sd.InputStream] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
        )

        if not await self._wait_for_services():
            logger.error("Services not reachable — exiting")
            return

        self.running = True

        # Open a persistent mic input stream
        self._input_stream = sd.InputStream(
            samplerate=self.config.AUDIO_SAMPLE_RATE,
            channels=self.config.AUDIO_CHANNELS,
            dtype="float32",
            blocksize=self.config.AUDIO_CHUNK_SIZE,
            device=self.config.AUDIO_INPUT_DEVICE,
        )
        self._input_stream.start()
        logger.info("Mic stream open  sr=%d  chunk=%d", self.config.AUDIO_SAMPLE_RATE, self.config.AUDIO_CHUNK_SIZE)

        # Generate confirmation audio if needed
        await self._ensure_confirmations()

        # Start background workers
        self._playback_task = asyncio.create_task(self._playback_worker())
        self._wakeword_task = asyncio.create_task(self._wakeword_loop())

        logger.info("Pipeline started — listening for wake word")

        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self.running = False

        for task in (self._wakeword_task, self._pipeline_task, self._playback_task):
            if task:
                task.cancel()

        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()

        if self.session:
            await self.session.close()

        logger.info("Pipeline stopped")

    # ── service readiness ─────────────────────────────────────────────────────

    async def _wait_for_services(self, timeout: int = 60) -> bool:
        services = {
            "wakeword": self.config.WAKEWORD_SERVICE_URL,
            "stt": self.config.STT_SERVICE_URL,
            "llm": self.config.LLM_SERVICE_URL,
            "tts": self.config.TTS_SERVICE_URL,
        }

        logger.info("Waiting for services...")
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            all_ok = True
            for name, url in services.items():
                try:
                    async with self.session.get(
                        f"{url}/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status != 200:
                            all_ok = False
                            break
                except Exception:
                    all_ok = False
                    break

            if all_ok:
                logger.info("All services healthy")
                return True

            await asyncio.sleep(2)

        return False

    # ── confirmation audio ────────────────────────────────────────────────────

    async def _ensure_confirmations(self):
        os.makedirs(self.config.CONFIRMATIONS_PATH, exist_ok=True)

        for phrase in self.config.confirmation_phrases:
            path = os.path.join(self.config.CONFIRMATIONS_PATH, f"{phrase}.wav")
            if os.path.exists(path):
                continue
            try:
                async with self.session.post(
                    f"{self.config.TTS_SERVICE_URL}/synthesize",
                    json={"text": phrase},
                ) as resp:
                    if resp.status == 200:
                        with open(path, "wb") as f:
                            f.write(await resp.read())
                        logger.info("Generated confirmation: %s", phrase)
            except Exception as e:
                logger.warning("Failed to generate confirmation '%s': %s", phrase, e)

    async def _play_confirmation(self):
        phrases = self.config.confirmation_phrases
        if not phrases:
            return

        phrase = random.choice(phrases)
        path = os.path.join(self.config.CONFIRMATIONS_PATH, f"{phrase}.wav")

        if not os.path.exists(path):
            return

        loop = asyncio.get_event_loop()
        data, sr = await loop.run_in_executor(None, partial(sf.read, path, dtype="float32"))
        await self.audio_queue.put((data.astype(np.float32) * self.config.AUDIO_VOLUME, sr))

    # ── audio helpers (all run in executor to avoid blocking) ─────────────────

    def _read_chunk_sync(self) -> np.ndarray:
        """Read one chunk from the persistent input stream (blocking)."""
        chunk, _ = self._input_stream.read(self.config.AUDIO_CHUNK_SIZE)
        return chunk.flatten().astype(np.float32)

    async def _read_chunk(self) -> np.ndarray:
        return await asyncio.get_event_loop().run_in_executor(None, self._read_chunk_sync)

    def _play_audio_sync(self, audio: np.ndarray, sr: int):
        sd.play(audio * self.config.AUDIO_VOLUME, samplerate=sr, device=self.config.AUDIO_OUTPUT_DEVICE)
        sd.wait()

    # ── playback worker ──────────────────────────────────────────────────────

    async def _playback_worker(self):
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                item = await self.audio_queue.get()
                if item is None:
                    self.audio_queue.task_done()
                    continue

                if self.cancel_event.is_set():
                    self._drain_queue()
                    self.audio_queue.task_done()
                    continue

                audio_data, sample_rate = item
                await loop.run_in_executor(None, self._play_audio_sync, audio_data, sample_rate)
                self.audio_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Playback error: %s", e)

    def _drain_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except asyncio.QueueEmpty:
                break

    # ── wakeword loop ────────────────────────────────────────────────────────

    async def _wakeword_loop(self):
        while self.running:
            try:
                chunk = await self._read_chunk()
                detected = await self._detect_wakeword(chunk)

                if detected:
                    logger.info("Wake word detected!")

                    # Cancel any running pipeline
                    self.cancel_event.set()
                    await asyncio.get_event_loop().run_in_executor(None, sd.stop)
                    self._drain_queue()
                    self.cancel_event.clear()

                    if self._pipeline_task and not self._pipeline_task.done():
                        self._pipeline_task.cancel()
                        try:
                            await self._pipeline_task
                        except asyncio.CancelledError:
                            pass

                    self._pipeline_task = asyncio.create_task(self._run_pipeline())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Wakeword loop error: %s", e)
                await asyncio.sleep(0.1)

    async def _detect_wakeword(self, audio_chunk: np.ndarray) -> bool:
        pcm_int16 = np.clip(audio_chunk, -1.0, 1.0)
        audio_bytes = (pcm_int16 * 32767).astype(np.int16).tobytes()

        try:
            async with self.session.post(
                f"{self.config.WAKEWORD_SERVICE_URL}/detect",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("detected", False)
        except Exception:
            pass
        return False

    # ── main pipeline ────────────────────────────────────────────────────────

    async def _run_pipeline(self):
        try:
            # 1. Play confirmation
            await self._play_confirmation()

            # 2. Record utterance
            audio_data = await self._record_until_silence()
            if audio_data is None or self.cancel_event.is_set():
                return

            # 3. Transcribe
            text = await self._transcribe(audio_data)
            if not text or self.cancel_event.is_set():
                logger.info("No speech detected — resuming wake-word loop")
                return

            logger.info("User: %s", text)

            # 4. LLM -> TTS -> playback (streaming)
            await self._stream_respond(text)

            # 5. Chat continuation
            if self.config.CHAT_CONTINUATION_ENABLED and not self.cancel_event.is_set():
                await self._chat_continuation()

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        except Exception as e:
            logger.error("Pipeline error: %s", e, exc_info=True)
        finally:
            logger.info("Listening for wake word")

    # ── record until silence ─────────────────────────────────────────────────

    async def _record_until_silence(self) -> Optional[np.ndarray]:
        logger.info("Recording...")

        sr = self.config.AUDIO_SAMPLE_RATE
        chunk_size = self.config.AUDIO_CHUNK_SIZE
        max_chunks = int(self.config.MAX_RECORD_SECONDS * sr / chunk_size)
        silence_chunks = int(self.config.SILENCE_DURATION * sr / chunk_size)

        recorded: list[np.ndarray] = []
        silent_count = 0
        speech_detected = False

        for _ in range(max_chunks):
            if self.cancel_event.is_set():
                return None

            chunk = await self._read_chunk()
            recorded.append(chunk)

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= self.config.SILENCE_THRESHOLD:
                speech_detected = True
                silent_count = 0
            elif speech_detected:
                silent_count += 1
                if silent_count >= silence_chunks:
                    break

        if not recorded:
            return None

        audio = np.concatenate(recorded)
        logger.info("Captured %.2f s of audio", len(audio) / sr)
        return audio

    # ── transcribe via STT service ───────────────────────────────────────────

    async def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        try:
            # Send as float32 bytes (STT service expects this)
            audio_bytes = audio.astype(np.float32).tobytes()
            async with self.session.post(
                f"{self.config.STT_SERVICE_URL}/transcribe",
                data=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("text", "").strip()
        except Exception as e:
            logger.error("Transcription error: %s", e)
        return None

    # ── LLM -> sentences -> TTS -> audio queue ───────────────────────────────

    async def _stream_respond(self, user_text: str):
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        if self.config.AGENT_ENABLED:
            # Agent mode: get full response, then split into sentences
            llm_task = asyncio.create_task(self._agent_to_sentences(user_text, sentence_queue))
        else:
            # Streaming mode: buffer tokens into sentences
            llm_task = asyncio.create_task(self._stream_to_sentences(user_text, sentence_queue))

        tts_task = asyncio.create_task(self._sentences_to_audio(sentence_queue))

        await asyncio.gather(llm_task, tts_task)

    async def _agent_to_sentences(self, user_text: str, sentence_queue: asyncio.Queue):
        """Agent mode: POST /chat, split full response into sentences."""
        try:
            async with self.session.post(
                f"{self.config.LLM_SERVICE_URL}/chat",
                json={"text": user_text},
            ) as resp:
                if resp.status != 200:
                    logger.error("LLM /chat error: %d", resp.status)
                    return

                result = await resp.json()
                full_text = result.get("text", "").strip()

            if not full_text or self.cancel_event.is_set():
                return

            logger.info("Assistant: %s", full_text)

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", full_text)
            for sentence in sentences:
                s = sentence.strip()
                if s and not self.cancel_event.is_set():
                    await sentence_queue.put(s)
        finally:
            await sentence_queue.put(None)

    async def _stream_to_sentences(self, user_text: str, sentence_queue: asyncio.Queue):
        """Streaming mode: POST /stream (SSE), buffer tokens into sentences."""
        buffer = ""

        try:
            async with self.session.post(
                f"{self.config.LLM_SERVICE_URL}/stream",
                json={"text": user_text},
            ) as resp:
                if resp.status != 200:
                    logger.error("LLM /stream error: %d", resp.status)
                    return

                full_reply_parts: list[str] = []

                async for line in resp.content:
                    if self.cancel_event.is_set():
                        break

                    line_text = line.decode("utf-8").strip()
                    if not line_text.startswith("data: "):
                        continue

                    token = line_text[6:]
                    if token == "[DONE]":
                        break
                    if token.startswith("ERROR:"):
                        logger.error("LLM stream error: %s", token[6:])
                        break

                    buffer += token
                    full_reply_parts.append(token)

                    # Extract complete sentences
                    while True:
                        match = SENTENCE_BOUNDARY.search(buffer)
                        if not match:
                            break
                        sentence = buffer[: match.end()].strip()
                        buffer = buffer[match.end() :]
                        if sentence:
                            await sentence_queue.put(sentence)

                # Flush remaining buffer
                if buffer.strip() and not self.cancel_event.is_set():
                    await sentence_queue.put(buffer.strip())

                logger.info("Assistant: %s", "".join(full_reply_parts).strip())

        except Exception as e:
            logger.error("LLM stream error: %s", e)
        finally:
            await sentence_queue.put(None)

    async def _sentences_to_audio(self, sentence_queue: asyncio.Queue):
        """Consume sentences, synthesize via TTS, push audio to playback queue."""
        while True:
            sentence = await sentence_queue.get()
            if sentence is None or self.cancel_event.is_set():
                sentence_queue.task_done()
                break

            try:
                logger.info("TTS <- '%s'", sentence)
                async with self.session.post(
                    f"{self.config.TTS_SERVICE_URL}/synthesize",
                    json={"text": sentence},
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        data, sr = sf.read(io.BytesIO(content), dtype="float32")
                        await self.audio_queue.put((data.astype(np.float32), sr))
            except Exception as e:
                logger.error("TTS error for '%s': %s", sentence, e)
            finally:
                sentence_queue.task_done()

    # ── chat continuation ────────────────────────────────────────────────────

    async def _chat_continuation(self):
        """After response, listen for follow-up speech within timeout."""
        logger.info("Waiting for follow-up...")

        # Wait for playback queue to drain first
        await self.audio_queue.join()

        deadline = asyncio.get_event_loop().time() + self.config.CHAT_CONTINUATION_TIMEOUT

        while asyncio.get_event_loop().time() < deadline:
            if self.cancel_event.is_set():
                return

            chunk = await self._read_chunk()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > self.config.SILENCE_THRESHOLD * 1.5:
                logger.info("Follow-up detected!")

                audio_data = await self._record_until_silence()
                if audio_data is None:
                    return

                text = await self._transcribe(audio_data)
                if not text:
                    return

                logger.info("User (cont): %s", text)
                await self._stream_respond(text)

                # Recursive: wait for another follow-up
                if not self.cancel_event.is_set():
                    await self._chat_continuation()
                return

        logger.info("No follow-up — returning to wake-word loop")
