"""
Call pipeline orchestrator.
Connects VAD → STT → LLM → TTS in real-time.
Each CallHandler instance manages one active call session.
"""

import asyncio
import re
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from loguru import logger

_load_lock = threading.Lock()


def _sanitize_for_tts(text: str) -> str:
    """
    Strip content that should never be spoken aloud:
    URLs, markdown formatting, HTML tags, JSON artifacts.
    """
    # Remove URLs (http/https/www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove markdown links [text](url) → keep text only
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove markdown bold/italic/code: **, *, __, _, ``, `
    text = re.sub(r'(\*\*|__|\*|_|`{1,3})', '', text)
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove JSON field names that leaked into speech (e.g. "action": "ask")
    text = re.sub(r'"?\w+"?\s*:\s*"[^"]*"', '', text)
    # Remove leftover curly/square braces
    text = re.sub(r'[{}\[\]]', '', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

    # Shared model instances (loaded once, reused across calls in the same process)
    _vad: Optional[VADDetector] = None
    _stt: Optional[Transcriber] = None
    _tts: Optional[VoiceSynthesizer] = None
    _models_loaded: bool = False

    @classmethod
    def load_models(cls):
        """Pre-load all models. Thread-safe — both server startups call this; only the first one loads."""
        with _load_lock:
            if cls._models_loaded:
                return
            if cls._vad is None:
                cls._vad = VADDetector()
            if cls._stt is None:
                cls._stt = Transcriber()
            if cls._tts is None:
                cls._tts = VoiceSynthesizer()
                cls._tts.load()
            cls._models_loaded = True
        logger.info("All models ready.")

    # ── Instance ──────────────────────────────────────────────────────────────

    def __init__(self, session_id: str, language: str = "en"):
        self.session_id = session_id
        self.language = language
        self.agent = HospitalAgent(language=language)

        # Per-call audio state
        self._speech_buffer: list = []
        self._last_speech_ts: Optional[float] = None
        self._is_active = False
        self._processing = False

        # Rolling pre-speech buffer: captures ~300ms before VAD fires
        # so the first syllable of a word is never clipped
        _pre_buf_frames = int(settings.sample_rate * 0.3)
        self._pre_buffer: deque = deque(maxlen=_pre_buf_frames)

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

        self.agent.reset()
        self.__class__.load_models()
        self._tts.set_language(self.agent.language)  # use agent.language — 'bi' mode starts as 'en'
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
        greeting = _sanitize_for_tts(greeting)
        # Includes Urdu full stop ۔ (U+06D4) and Urdu ? ؟ (U+061F)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?۔؟])\s+', greeting) if s.strip()]
        full_audio = b""
        for sentence in sentences:
            audio = await self._tts.synthesize(sentence)
            if audio:
                full_audio += audio
                yield ("audio", audio)

        # Cache for next call — skip if this is a fallback error message
        _ERROR_SIGNALS = ["تکنیکی مسئلہ", "technical issue", "having trouble", "brief technical"]
        is_error_greeting = any(sig in greeting for sig in _ERROR_SIGNALS)
        if full_audio and not is_error_greeting:
            with open(cache_path, "wb") as f:
                f.write(full_audio)
            logger.info(f"[{self.session_id}] Greeting cached")
        elif is_error_greeting:
            logger.warning(f"[{self.session_id}] Skipping cache — greeting is a fallback error message")

    async def end_call(self):
        self._is_active = False
        self._save_call_log()
        self._speech_buffer = []
        self._processing = False
        self._vad.reset()
        logger.info(f"[{self.session_id}] Call ended.")

    def _save_call_log(self):
        import json
        import os
        from datetime import datetime

        try:
            log_dir = "data/call_logs"
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_path = os.path.join(log_dir, f"{date_str}_{self.session_id}.json")
            log_data = {
                "session_id": self.session_id,
                "language": self.language,
                "timestamp": datetime.now().isoformat(),
                "transcript": self.agent.get_transcript(),
                "collected": self.agent._collected,
                "flow_state": self.agent._flow_state,
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[{self.session_id}] Call log saved → {log_path}")
        except Exception as e:
            logger.warning(f"[{self.session_id}] Could not save call log: {e}")

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
            if not self._speech_buffer:
                # Prepend pre-buffer so the start of the word isn't clipped
                self._speech_buffer.extend(list(self._pre_buffer))
            self._speech_buffer.extend(audio.tolist())
            self._last_speech_ts = time.time()
            return

        # No speech — keep pre-buffer rolling so next speech turn has context
        self._pre_buffer.extend(audio.tolist())

        # Check for end-of-turn
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
            # Sync TTS to agent's current language (handles mid-call language switches)
            self._tts.set_language(self.agent.language)

            # 1. Speech → Text  (GPU, runs in thread pool to not block event loop)
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, lambda: self._stt.transcribe(audio_array, language=self.agent.language)
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
                    sentence = _sanitize_for_tts(item.strip())
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
                    if action == "end":
                        self._is_active = False

            # Send full agent speech to transcript
            if agent_speech_parts:
                yield ("json", {"type": "agent_speech", "text": " ".join(agent_speech_parts)})

        except Exception as e:
            logger.exception(f"[{self.session_id}] Turn error: {e}")
            fallback_text = (
                "معذرت، ایک تکنیکی مسئلہ ہے۔ کیا آپ دوبارہ بتا سکتے ہیں؟"
                if self.language == "ur"
                else "I'm sorry, I'm having a technical issue. Could you please repeat?"
            )
            fallback = await self._tts.synthesize(fallback_text)
            if fallback:
                yield ("audio", fallback)
        finally:
            self._processing = False
