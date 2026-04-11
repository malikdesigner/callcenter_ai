"""
Call pipeline orchestrator.
Connects VAD → STT → LLM → TTS in real-time.
Each CallHandler instance manages one active call session.
"""

import asyncio
import time
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings
from src.llm.agent import HospitalAgent
from src.stt.transcriber import Transcriber
from src.tts.synthesizer import VoiceSynthesizer
from src.vad.detector import VADDetector


class CallHandler:
    """
    One instance per active call.
    Feed raw audio chunks via process_audio_chunk().
    """

    # Shared model instances (loaded once, reused across calls)
    _vad: Optional[VADDetector] = None
    _stt: Optional[Transcriber] = None
    _tts: Optional[VoiceSynthesizer] = None

    @classmethod
    def load_models(cls):
        """Pre-load all models at startup so the first call isn't slow."""
        if cls._vad is None:
            cls._vad = VADDetector()
        if cls._stt is None:
            cls._stt = Transcriber()
        if cls._tts is None:
            cls._tts = VoiceSynthesizer()
            cls._tts.load()  # Ensure weights are in GPU memory
        logger.info("All models ready.")

    # ── Instance ──────────────────────────────────────────────────────────────

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.agent = HospitalAgent()

        # Per-call audio state
        self._speech_buffer: list = []
        self._last_speech_ts: Optional[float] = None
        self._is_active = False
        self._processing = False   # prevent overlapping STT/LLM calls

        # Silence gate: how long of silence triggers processing
        self._silence_gate = settings.silence_duration

    @property
    def is_active(self) -> bool:
        return self._is_active

    # ── Call lifecycle ────────────────────────────────────────────────────────

    async def start_call(self):
        """
        Initialise the agent and yield ("audio"|"json", payload) tuples.
        Greeting audio is cached after first synthesis for zero-latency on repeat calls.
        """
        import hashlib
        import os
        import re

        self.agent.reset()
        self.__class__.load_models()
        self._is_active = True
        self._speech_buffer = []
        self._last_speech_ts = None

        greeting = await self.agent.get_greeting()
        logger.info(f"[{self.session_id}] Greeting: {greeting}")

        # Signal browser: Sara is about to speak
        yield ("json", {"type": "agent_speech", "text": greeting})

        # Check for cached greeting audio
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        greeting_hash = hashlib.md5(greeting.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"greeting_{greeting_hash}.bin")

        if os.path.exists(cache_path):
            logger.info(f"[{self.session_id}] Using cached greeting audio")
            with open(cache_path, "rb") as f:
                yield ("audio", f.read())
            return

        # Synthesize sentence-by-sentence and stream immediately
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', greeting) if s.strip()]
        full_audio = b""
        for sentence in sentences:
            audio = await self._tts.synthesize(sentence)
            if audio:
                full_audio += audio
                yield ("audio", audio)

        # Cache for next call
        if full_audio:
            with open(cache_path, "wb") as f:
                f.write(full_audio)
            logger.info(f"[{self.session_id}] Greeting cached")

    async def end_call(self):
        self._is_active = False
        self._speech_buffer = []
        self._processing = False
        self._vad.reset()
        logger.info(f"[{self.session_id}] Call ended.")

    # ── Audio ingestion ───────────────────────────────────────────────────────

    async def process_audio_chunk(self, chunk: bytes):
        """
        Accept raw PCM bytes (float32, 16 kHz, mono).
        Yields response WAV bytes sentences when a full user turn is detected.
        """
        if not self._is_active or self._processing:
            return

        audio = np.frombuffer(chunk, dtype=np.float32)
        if len(audio) == 0:
            return

        is_speech = self._vad.is_speech(audio)

        if is_speech:
            self._speech_buffer.extend(audio.tolist())
            self._last_speech_ts = time.time()
            return

        # No speech in this chunk — check for end-of-turn
        if self._last_speech_ts and self._speech_buffer:
            # Only trigger if we have a significant amount of speech (at least 0.3s)
            # to avoid triggering on clicks or background coughs
            if len(self._speech_buffer) < (settings.sample_rate * 0.3):
                 self._speech_buffer = []
                 self._last_speech_ts = None
                 return

            silence = time.time() - self._last_speech_ts
            if silence >= self._silence_gate:
                 async for response_block in self._process_turn():
                     yield response_block

    # ── Turn processing ───────────────────────────────────────────────────────

    async def _process_turn(self):
        """
        Transcribe buffered speech, get LLM response (streaming), synthesize.
        Yields tuples:
          ("audio", bytes)       — audio chunk to stream to browser
          ("json",  dict)        — metadata/signal to send as JSON text frame
        """
        if self._processing:
            return

        self._processing = True
        audio_array = np.array(self._speech_buffer, dtype=np.float32)
        self._speech_buffer = []
        self._last_speech_ts = None

        try:
            # 1. Speech → Text  (GPU, runs in thread pool to not block event loop)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, self._stt.transcribe, audio_array
            )
            logger.info(f"[{self.session_id}] User: {transcript}")

            if not transcript.strip():
                return

            # Tell browser what the user said
            yield ("json", {"type": "transcript", "text": transcript})
            # Tell browser we're thinking
            yield ("json", {"type": "processing"})

            # 2. LLM response (streaming sentences) → TTS per sentence
            agent_speech_parts = []
            async for item in self.agent.process_turn_stream(transcript):
                if isinstance(item, str):
                    sentence = item.strip()
                    if not sentence:
                        continue
                    logger.debug(f"[{self.session_id}] TTS sentence: {sentence}")
                    agent_speech_parts.append(sentence)
                    audio_bytes = await self._tts.synthesize(sentence)
                    if audio_bytes:
                        yield ("audio", audio_bytes)
                else:
                    # Final result dict — handle action
                    result = item
                    action = result.get("action", "none")
                    logger.info(f"[{self.session_id}] Action: {action}")
                    if action == "end_call":
                        self._is_active = False

            # Send full agent speech to transcript
            if agent_speech_parts:
                yield ("json", {"type": "agent_speech", "text": " ".join(agent_speech_parts)})

        except Exception as e:
            logger.exception(f"[{self.session_id}] Turn error: {e}")
            fallback = await self._tts.synthesize(
                "I'm sorry, I'm having a technical issue. Could you please repeat?"
            )
            if fallback:
                yield ("audio", fallback)
        finally:
            self._processing = False
