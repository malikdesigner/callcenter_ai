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

# Pre-synthesized micro-acknowledgment audio, cached by language.
_MICRO_ACK_TEXT = {
    "ur": ["جی...", "اچھا...", "ٹھیک ہے..."],
    "ro": ["Ji...", "Acha...", "Theek hai..."],
    "en": ["Mhm...", "Okay...", "Right..."],
    "bi": ["Mhm...", "Okay..."],
}


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
from src.llm.agent_v2 import HospitalAgentV2
from src.llm.data_extractor import extract_patient_data
from src.stt.transcriber import Transcriber
from src.tts.synthesizer import VoiceSynthesizer
from src.vad.detector import VADDetector
from src.pipeline.stt_filter import is_valid_input
from src.dialogue.semantic_validator import is_semantically_valid
from src.dialogue.policy import DialoguePolicy
from src.llm.intent import detect_intent


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
    _ack_cache: dict = {}  # {lang: bytes} — micro-ack audio, synthesized once per lang

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

    def __init__(self, session_id: str, language: str = "en", phone_mode: bool = False):
        self.session_id = session_id
        self.language = language
        self.phone_mode = phone_mode          # True for Twilio calls (8 kHz upsampled audio)
        self.agent = HospitalAgentV2(language=language)
        self.policy = DialoguePolicy(language=language)
        self._db_session = None               # opened in start_call(), closed in end_call()

        # Per-call audio state
        self._speech_buffer: list = []
        self._last_speech_ts: Optional[float] = None
        self._is_active = False
        self._processing = False
        self._response_queue = asyncio.Queue()
        self._turn_task = None

        # Rolling pre-speech buffer: captures ~300ms before VAD fires
        # so the first syllable of a word is never clipped
        _pre_buf_frames = int(settings.sample_rate * 0.3)
        self._pre_buffer: deque = deque(maxlen=_pre_buf_frames)

        # Silence gate: how long of silence triggers processing
        # 2.5s for Urdu/Roman to allow thinking pauses, 1.8s for English
        self._silence_gate = 2.5 if language in ("ur", "ro") else 1.8

    @property
    def is_active(self) -> bool:
        return self._is_active

    # ── Call lifecycle ────────────────────────────────────────────────────────


    async def consume_responses(self):
        while self._is_active:
            try:
                item = await self._response_queue.get()
                if item is None:
                    break
                yield item
            except asyncio.CancelledError:
                break

    async def start_call(self):
        """
        Initialise the agent and yield ("audio"|"json", payload) tuples.
        Greeting audio is cached after first synthesis for zero-latency on repeat calls.
        """
        import hashlib
        import os

        # Open a per-call DB session and inject into agent
        from src.db.database import AsyncSessionLocal
        self._db_session = AsyncSessionLocal()
        self.agent.set_db(self._db_session)

        self.agent.reset()
        self.policy.reset()
        self.__class__.load_models()
        self._tts.set_language(self.agent.language)  # use agent.language — 'bi' mode starts as 'en'
        self._is_active = True
        self._speech_buffer = []
        self._last_speech_ts = None

        greeting = await self.agent.get_greeting()
        logger.info(f"[{self.session_id}] Greeting: {greeting}")

        # Signal browser: Sara is about to speak
        await self._response_queue.put(("json", {"type": "agent_speech", "text": greeting}))

        # Check for cached greeting audio
        cache_dir = "data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        greeting_hash = hashlib.md5(greeting.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, f"greeting_{greeting_hash}.bin")

        # Bilingual greetings use mixed TTS voices — don't serve from single-voice cache
        if os.path.exists(cache_path) and not self.agent._bilingual_mode:
            logger.info(f"[{self.session_id}] Using cached greeting audio")
            with open(cache_path, "rb") as f:
                await self._response_queue.put(("audio", f.read()))
            return

        greeting = _sanitize_for_tts(greeting)
        full_audio = b""
        if self.agent.language in ("ur", "ro"):
            # Single TTS call for Urdu/Roman Urdu — avoids MP3 playback reset gaps between sentences
            self._tts.set_language(self.agent.language)
            audio = await self._tts.synthesize(greeting)
            if audio:
                full_audio = audio
                await self._response_queue.put(("audio", audio))
        else:
            # English/bilingual: sentence-by-sentence for faster first-audio delivery
            # Includes Urdu full stop ۔ (U+06D4) and Urdu ? ؟ (U+061F)
            sentences = [s.strip() for s in re.split(r'(?<=[.!?۔؟])\s+', greeting) if s.strip()]
            for sentence in sentences:
                # For bilingual greetings, switch TTS voice per sentence based on script.
                ur_chars = sum(1 for c in sentence if '؀' <= c <= 'ۿ')
                if ur_chars > 2:
                    self._tts.set_language("ur")
                else:
                    self._tts.set_language("en" if self.agent.language in ("en", "bi") else self.agent.language)
                audio = await self._tts.synthesize(sentence)
                if audio:
                    full_audio += audio
                    await self._response_queue.put(("audio", audio))
            # Restore correct TTS language after greeting
            self._tts.set_language(self.agent.language if self.agent.language != "bi" else "en")

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
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        await self._response_queue.put(None)
        self._save_call_log()
        # Close the per-call DB session
        if self._db_session:
            await self._db_session.close()
            self._db_session = None
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
                "policy_state": self.policy.state,
                "policy_collected": self.policy.collected_data,
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
        """
        if not self._is_active:
            return

        audio = np.frombuffer(chunk, dtype=np.float32)
        if len(audio) == 0:
            return

        # NEW: PCM Continuity & Normalization
        # Simple peak normalization if very quiet
        peak = np.max(np.abs(audio))
        if 0 < peak < 0.1:  # Boost if extremely quiet
            audio = audio * (0.1 / peak)

        is_speech = self._vad.is_speech(audio)

        if is_speech:
            if self._turn_task and not self._turn_task.done():
                logger.info(f"[{self.session_id}] Interruption detected! Canceling turn.")
                self._turn_task.cancel()
                self._turn_task = None
                self._processing = False
                await self._response_queue.put(("json", {"type": "interrupt"}))

            if not self._speech_buffer:
                # Prepend pre-buffer so the start of the word isn't clipped
                self._speech_buffer.extend(list(self._pre_buffer))
            self._speech_buffer.extend(audio.tolist())
            self._last_speech_ts = time.time()
            return

        # No speech
        # If we are already in an active turn, we MUST keep appending silence 
        # to preserve natural intra-word pauses for Whisper!
        if self._last_speech_ts:
            self._speech_buffer.extend(audio.tolist())

        # keep pre-buffer rolling so next speech turn has context
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
                 self._turn_task = asyncio.create_task(self._process_turn())

    # ── Conversation logging ──────────────────────────────────────────────────

    async def _log_turn(
        self,
        transcript: str,
        confidence: dict,
        intent: str,
        policy_state: str,
        response: str,
        used_fallback: bool = False,
        latency_ms: int = None,
    ):
        """Persist one conversation turn to the database for later analysis."""
        try:
            from src.db.models import ConversationLog
            log = ConversationLog(
                session_id=self.session_id,
                language=self.language,
                turn_number=getattr(self.agent, "_turn_count", 0),
                user_transcript=transcript,
                asr_confidence=confidence.get("avg_logprob"),
                detected_intent=intent,
                policy_state=policy_state,
                agent_response=response,
                model_used=settings.ollama_model if not used_fallback else settings.claude_fallback_model,
                used_fallback=used_fallback,
                latency_ms=latency_ms,
            )
            self._db_session.add(log)
            await self._db_session.commit()
        except Exception as e:
            logger.warning(f"[{self.session_id}] Conv log failed: {e}")

    # ── Micro-acknowledgment ──────────────────────────────────────────────────

    async def _get_ack_audio(self) -> bytes:
        """
        Return pre-synthesized micro-ack audio for the current language.
        Randomly picks one to sound natural.
        """
        import random
        lang = self.agent.language
        if lang not in self.__class__._ack_cache:
            self.__class__._ack_cache[lang] = []
            texts = _MICRO_ACK_TEXT.get(lang, ["Okay..."])
            tts_lang = "en" if lang == "bi" else lang
            self._tts.set_language(tts_lang)
            for text in texts:
                audio = await self._tts.synthesize(text)
                if audio:
                    self.__class__._ack_cache[lang].append(audio)
            self._tts.set_language(self.agent.language)  # restore
        
        cache = self.__class__._ack_cache.get(lang, [])
        return random.choice(cache) if cache else b""

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
        turn_start = time.time()
        audio_array = np.array(self._speech_buffer, dtype=np.float32)
        self._speech_buffer = []
        self._last_speech_ts = None

        try:
            # Sync TTS to agent's current language (handles mid-call language switches)
            self._tts.set_language(self.agent.language)



            # 1. Speech → Text  (GPU, runs in thread pool to not block event loop)
            # During bilingual language selection: auto-detect so Whisper's language
            # detection (not just the transcript text) can identify Urdu vs English.
            in_bilingual_selection = (
                self.agent._bilingual_mode and not self.agent._language_chosen
            )
            loop = asyncio.get_event_loop()
            if in_bilingual_selection:
                transcript, audio_language, confidence = await loop.run_in_executor(
                    None, lambda: self._stt.transcribe(
                        audio_array, language=None,
                        return_language=True, return_confidence=True,
                        phone_mode=self.phone_mode
                    )
                )
            else:
                transcript, confidence = await loop.run_in_executor(
                    None, lambda: self._stt.transcribe(
                        audio_array, language=self.agent.language,
                        return_confidence=True, phone_mode=self.phone_mode
                    )
                )
                audio_language = None
            logger.info(f"[{self.session_id}] User: {transcript}")

            if not transcript.strip():
                return

            # ── ASR confidence gate (STT Filter) ────────────────────────────────────────────
            is_valid_stt, filter_reason = is_valid_input(transcript, confidence, self.agent.language)

            if not is_valid_stt:
                logger.warning(
                    f"[{self.session_id}] Garbage transcript filtered "
                    f"({filter_reason}): '{transcript}'"
                )
                lang = self.agent.language
                if lang == "ur":
                    repeat_text = "آواز واضح نئیں آئی ... دوبارہ بتا دیجیے"
                elif lang == "ro":
                    repeat_text = "Awaaz wazeh nahi aayi ... dobara bataein"
                else:
                    repeat_text = "Sorry, I didn't catch that clearly — could you repeat?"
                self._tts.set_language(lang)
                repeat_audio = await self._tts.synthesize(repeat_text)
                if repeat_audio:
                    await self._response_queue.put(("audio", repeat_audio))
                return

            # Tell browser what the user said
            await self._response_queue.put(("json", {"type": "transcript", "text": transcript}))
            # Tell browser we're thinking
            await self._response_queue.put(("json", {"type": "processing"}))

            # ── Pre-LLM transcript log (for debugging STT vs LLM errors) ─────
            logger.info(
                f"[{self.session_id}] PRE-LLM | lang={self.agent.language} "
                f"conf={confidence.get('avg_logprob', 0):.2f} | '{transcript}'"
            )

            # ── Dialogue Policy & Semantic Validation ─────────────────────────
            # 1. Extract patient data from transcript to keep policy state in sync.
            #    Without this, _collected is always empty and policy loops on ASK_NAME.
            extracted = extract_patient_data(transcript, self.agent.language)
            if extracted:
                self.agent._collected.update(extracted)
                logger.debug(f"[{self.session_id}] Extractor → {extracted}")

            # 2. Detect Intent
            current_intent = detect_intent(transcript, self.policy.state, self.agent.language)

            # 3. Semantic Validation
            is_valid_sem, sem_reason = is_semantically_valid(transcript, self.policy.state, self.agent.language)
            if not is_valid_sem:
                current_intent = "GARBAGE"
                logger.warning(f"[{self.session_id}] Semantic validation failed: {sem_reason}")

            # 4. Policy Evaluation — pass extracted data so state advances correctly
            policy_result = self.policy.evaluate(current_intent, self.agent._collected)
            
            if policy_result["is_hard_fallback"]:
                self._tts.set_language(self.agent.language)
                fallback_audio = await self._tts.synthesize(policy_result["fallback_text"], policy_tone="empathetic")
                if fallback_audio:
                    await self._response_queue.put(("audio", fallback_audio))
                return

            # 0. Micro-acknowledgment — instant filler word
            ack = await self._get_ack_audio()
            if ack:
                await self._response_queue.put(("audio", ack))

            # 2. LLM response → TTS (Streaming chunk-by-chunk for ultra-low latency)
            agent_speech_parts = []
            policy_tone = "empathetic" if "CONFUSED" in str(current_intent) or policy_result["current_state"] == "ASK_SYMPTOM" else None
            
            async for item in self.agent.process_turn_stream(transcript, audio_language=audio_language, policy_state=policy_result["current_state"], nlg_constraint=policy_result["nlg_constraint"]):
                if isinstance(item, str):
                    sentence = _sanitize_for_tts(item.strip())
                    if not sentence:
                        continue
                    logger.debug(f"[{self.session_id}] TTS sentence: {sentence}")
                    agent_speech_parts.append(sentence)
                    
                    # Stream immediately regardless of language
                    self._tts.set_language(self.agent.language)
                    audio_bytes = await self._tts.synthesize(sentence, policy_tone=policy_tone)
                    if audio_bytes:
                        await self._response_queue.put(("audio", audio_bytes))
                else:
                    # Final result dict — handle action
                    result = item
                    action = result.get("action", "none")
                    logger.info(f"[{self.session_id}] Action: {action}")
                    if action == "end":
                        self._is_active = False
                    elif action == "redirect":
                        port = result.get("data", {}).get("port", 8000)
                        await self._response_queue.put(("json", {"type": "redirect", "port": port}))
                        self._is_active = False

            # Send full agent speech to transcript
            full_speech = " ".join(agent_speech_parts)
            if full_speech:
                await self._response_queue.put(("json", {"type": "agent_speech", "text": full_speech}))

            # ── Persist conversation turn to DB ──────────────────────────────
            if self._db_session and full_speech:
                await self._log_turn(
                    transcript=transcript,
                    confidence=confidence,
                    intent=current_intent,
                    policy_state=policy_result.get("current_state", ""),
                    response=full_speech,
                    used_fallback=result.get("used_fallback", False) if isinstance(result, dict) else False,
                    latency_ms=int((time.time() - turn_start) * 1000) if 'turn_start' in dir() else None,
                )

        except Exception as e:
            logger.exception(f"[{self.session_id}] Turn error: {e}")
            if self.language == "ur":
                fallback_text = "سوری ... تکنیکی مسئلہ ہے ... دوبارہ بتائیں"
            elif self.language == "ro":
                fallback_text = "Sori ... technical masla hai ... dobara bataein"
            else:
                fallback_text = "I'm sorry, I'm having a technical issue. Could you please repeat?"
            fallback = await self._tts.synthesize(fallback_text)
            if fallback:
                await self._response_queue.put(("audio", fallback))
        finally:
            self._processing = False
