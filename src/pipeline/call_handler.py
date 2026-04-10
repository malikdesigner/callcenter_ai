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

    async def start_call(self) -> bytes:
        """
        Initialise the agent and return greeting audio bytes.
        Call this once when the WebSocket connects.
        """
        self.agent.reset()
        self.__class__.load_models()
        self._is_active = True
        self._speech_buffer = []
        self._last_speech_ts = None

        greeting = self.agent.get_greeting()
        logger.info(f"[{self.session_id}] Greeting: {greeting}")

        audio = await asyncio.get_event_loop().run_in_executor(
            None, self._tts.synthesize, greeting
        )
        return audio

    async def end_call(self):
        self._is_active = False
        self._speech_buffer = []
        self._processing = False
        self._vad.reset()
        logger.info(f"[{self.session_id}] Call ended.")

    # ── Audio ingestion ───────────────────────────────────────────────────────

    async def process_audio_chunk(self, chunk: bytes) -> Optional[bytes]:
        """
        Accept raw PCM bytes (float32, 16 kHz, mono).
        Returns response WAV bytes when a full user turn is detected, else None.
        """
        if not self._is_active or self._processing:
            return None

        audio = np.frombuffer(chunk, dtype=np.float32)
        if len(audio) == 0:
            return None

        is_speech = self._vad.is_speech(audio)

        if is_speech:
            self._speech_buffer.extend(audio.tolist())
            self._last_speech_ts = time.time()
            return None

        # No speech in this chunk — check for end-of-turn
        if self._last_speech_ts and self._speech_buffer:
            silence = time.time() - self._last_speech_ts
            if silence >= self._silence_gate:
                return await self._process_turn()

        return None

    # ── Turn processing ───────────────────────────────────────────────────────

    async def _process_turn(self) -> Optional[bytes]:
        """Transcribe buffered speech, get LLM response, synthesize."""
        if self._processing:
            return None

        self._processing = True
        audio_array = np.array(self._speech_buffer, dtype=np.float32)
        self._speech_buffer = []
        self._last_speech_ts = None

        try:
            # 1. Speech → Text  (GPU, offloaded to thread pool)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, self._stt.transcribe, audio_array
            )
            logger.info(f"[{self.session_id}] User: {transcript}")

            if not transcript.strip():
                return None

            # 2. Text → LLM response
            response = await loop.run_in_executor(
                None, self.agent.process_turn, transcript
            )
            speech_text = response.get("speech", "")
            action = response.get("action", "none")
            logger.info(f"[{self.session_id}] Agent [{action}]: {speech_text[:80]}")

            if not speech_text:
                return None

            # 3. Text → Audio
            audio_bytes = await loop.run_in_executor(
                None, self._tts.synthesize, speech_text
            )

            # End call after goodbye
            if action == "end_call":
                self._is_active = False

            return audio_bytes

        except Exception as e:
            logger.exception(f"[{self.session_id}] Turn error: {e}")
            # Return a fallback audio
            fallback = await asyncio.get_event_loop().run_in_executor(
                None,
                self._tts.synthesize,
                "I'm sorry, I didn't catch that. Could you please repeat?",
            )
            return fallback
        finally:
            self._processing = False
