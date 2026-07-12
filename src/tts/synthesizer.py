"""
Text-to-Speech — hybrid pipeline:

  English  → F5-TTS  (zero-shot voice cloning from voices/receptionist.wav)
  Urdu     → Edge-TTS ur-PK-UzmaNeural  (native Urdu phoneme support)
  Fallback → gTTS (Urdu) / pyttsx3 (English) if both primary engines fail

F5-TTS clones the receptionist's exact voice for English responses.
UzmaNeural is kept for Urdu because F5-TTS is English/Chinese trained
and cannot correctly pronounce Arabic-script phonemes.
"""

import io
import os
import re
import asyncio
import edge_tts
import pyttsx3
from loguru import logger
from config.settings import settings
from src.tts.speech_formatter import formatter as _formatter, TONE_PROSODY


# ── Urdu phonetic normalisation map ──────────────────────────────────────────
# Applied before TTS. Order matters: longer phrases first.
_URDU_PHONETIC_MAP = [
    # Contracted postpositions (no space → flows as one phoneme)
    ("آپ کا",           "آپکا"),
    ("آپ کو",           "آپکو"),
    ("آپ کے",           "آپکے"),
    ("آپ کی",           "آپکی"),
    ("آپ نے",           "آپنے"),
    # Common spoken contractions
    ("کے لیے",          "کیلیے"),
    ("ہو گئی",          "ہوگئی"),
    ("ہو گیا",          "ہوگیا"),
    ("کر سکتی ہوں",    "کرسکتی ہوں"),
    ("کر سکتا ہوں",    "کرسکتا ہوں"),
    ("کر دیتی ہوں",    "کردیتی ہوں"),
    ("بتا دیجیے",       "بتادیجیے"),
    ("لے لیتے ہیں",    "لے لیتے ہیں"),
    # Phonetic spellings UzmaNeural pronounces more naturally
    ("نہیں",            "نئیں"),
    ("موبائل",          "موبایل"),
    ("ڈاکٹر",           "ڈاکڑ"),
    ("تکلیف",          "تکلیف"),
    ("کیوں کہ",        "کیوں کہ"),
]

# ── Urdu hour words for time conversion ──────────────────────────────────────
_UR_HOURS = {
    1: "ایک", 2: "دو", 3: "تین", 4: "چار", 5: "پانچ",
    6: "چھ", 7: "سات", 8: "آٹھ", 9: "نو", 10: "دس",
    11: "گیارہ", 12: "بارہ",
}


# ── Voice configuration ────────────────────────────────────────────────────────
VOICE_CONFIG = {
    "en": {
        "voice":  "en-US-AriaNeural",
        "rate":   "-5%",
        "pitch":  "+0Hz",
        "volume": "+0%",
    },
    "ur": {
        "voice":  "ur-PK-UzmaNeural",
        "rate":   "-3%",    # measured pace — natural Pakistani Urdu cadence
        "pitch":  "-5Hz",   # warmer, more human-sounding
        "volume": "+0%",
    },
    # Roman Urdu: same Pakistani voice — handles phonetic Latin Urdu
    "ro": {
        "voice":  "ur-PK-UzmaNeural",
        "rate":   "-3%",
        "pitch":  "-5Hz",
        "volume": "+0%",
    },
}

# Minimum clause length (chars) to synthesise as a separate segment.
# Short fragments like "جی" or "ہاں" are kept with the next clause.
_MIN_CLAUSE_LEN = 6


class VoiceSynthesizer:
    def __init__(self):
        self._language = "en"
        self._f5_tts = None              # F5-TTS model instance (English voice cloning)
        self._f5_ref_text = ""           # transcript of voices/receptionist.wav
        self._f5_ref_audio_bytes = None  # ref audio cached in RAM — set once in _load_f5_tts
        logger.info(
            f"VoiceSynthesizer ready — "
            f"EN: F5-TTS (voice clone) | "
            f"UR: {VOICE_CONFIG['ur']['voice']}"
        )

    def load(self):
        """Load F5-TTS for English voice cloning. Falls back to Edge-TTS if unavailable."""
        if not settings.f5_tts_enabled:
            logger.info("[TTS] F5-TTS disabled in settings — using Edge-TTS for English")
            return
        self._load_f5_tts()

    def _load_f5_tts(self):
        try:
            import torch
            from f5_tts.api import F5TTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[TTS] Loading F5-TTS on {device}...")
            self._f5_tts = F5TTS(device=device)

            ref_path = settings.voice_sample_path

            # ── Step 1: Resolve reference text ───────────────────────────────
            if settings.f5_tts_ref_text.strip():
                self._f5_ref_text = settings.f5_tts_ref_text.strip()
                logger.info(f"[TTS] F5-TTS ref text from settings: '{self._f5_ref_text[:60]}'")
            elif os.path.exists(ref_path):
                self._f5_ref_text = self._auto_transcribe_ref(ref_path)
                logger.info(f"[TTS] F5-TTS ref text auto-detected: '{self._f5_ref_text[:60]}'")
            else:
                self._f5_ref_text = "Hello, thank you for calling. How can I help you today?"
                logger.warning(f"[TTS] F5-TTS: ref audio not found at '{ref_path}' — using generic ref text")

            # ── Step 2: Cache reference audio in RAM (no disk reads per call) ─
            # torchaudio.load() accepts BytesIO directly, so we read the file
            # once here and reuse the in-memory bytes on every synthesis call.
            if os.path.exists(ref_path):
                import soundfile as sf
                audio_data, sample_rate = sf.read(ref_path, dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
                self._f5_ref_audio_bytes = buf.getvalue()
                logger.info(
                    f"[TTS] F5-TTS ref audio cached in RAM "
                    f"({len(self._f5_ref_audio_bytes) // 1024} KB) — no disk reads per call"
                )
            else:
                self._f5_ref_audio_bytes = None

            logger.info(f"[TTS] F5-TTS ready on {device}")

        except ImportError:
            logger.warning("[TTS] f5-tts not installed — run: pip install f5-tts | Falling back to Edge-TTS")
            self._f5_tts = None
        except Exception as e:
            logger.error(f"[TTS] F5-TTS load failed: {e} — falling back to Edge-TTS")
            self._f5_tts = None

    @staticmethod
    def _auto_transcribe_ref(ref_path: str) -> str:
        """Transcribe the reference audio using a tiny Whisper model (runs once on startup)."""
        try:
            from faster_whisper import WhisperModel
            logger.info("[TTS] Auto-transcribing reference audio with Whisper base...")
            m = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = m.transcribe(ref_path, beam_size=3)
            text = " ".join(s.text.strip() for s in segments).strip()
            del m
            return text or "Hello, thank you for calling. How can I help you today?"
        except Exception as e:
            logger.warning(f"[TTS] Ref transcription failed: {e}")
            return "Hello, thank you for calling. How can I help you today?"

    def set_language(self, lang: str):
        self._language = lang

    # ── Public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_urdu_text(text: str) -> bool:
        """True when ≥25 % of non-space characters are Urdu/Arabic script."""
        non_space = [c for c in text if not c.isspace()]
        if not non_space:
            return False
        ur_count = sum(1 for c in non_space if '؀' <= c <= 'ۿ')
        return ur_count / len(non_space) >= 0.25

    async def synthesize(self, text: str, policy_tone: str = None) -> bytes:
        """
        Convert text → audio bytes.

        Routing:
          English  → F5-TTS voice cloning (if loaded) else Edge-TTS AriaNeural
          Urdu     → Edge-TTS UzmaNeural  (native Urdu phoneme support)
          Roman UR → Edge-TTS UzmaNeural
        """
        text = text.strip()
        if not text:
            return b""

        is_urdu = self._language == "ur" or self._is_urdu_text(text)
        is_roman_urdu = self._language == "ro" and not is_urdu

        # ── Speech formatting: formal→spoken rewrite + tone classification ────
        lang_for_fmt = "ur" if is_urdu else ("ro" if is_roman_urdu else "en")
        text, tone = _formatter.format(text, lang_for_fmt, policy_tone=policy_tone)
        logger.debug(f"[TTS] Tone={tone} | lang={lang_for_fmt} | {text[:60]}…")

        # ── Urdu / Roman Urdu → Edge-TTS (unchanged — native phoneme support) ─
        if is_urdu:
            text = self._prepare_urdu_for_tts(text)
            return await self._synthesize_raw(text, tone=tone)

        if is_roman_urdu:
            text = self._prepare_roman_urdu_for_tts(text)
            return await self._synthesize_raw(text, tone=tone)

        # ── English → F5-TTS voice cloning ────────────────────────────────────
        if self._f5_tts is not None:
            logger.debug(f"[TTS] F5-TTS EN | voice clone | tone={tone}")
            result = await self._synthesize_f5tts(text)
            if result:
                return result
            logger.warning("[TTS] F5-TTS returned empty — falling back to Edge-TTS")

        # ── English fallback → Edge-TTS AriaNeural ────────────────────────────
        clauses = self._split_clauses(text)
        logger.debug(f"[TTS] Edge-TTS EN | {VOICE_CONFIG['en']['voice']} | {len(clauses)} clause(s)")
        if len(clauses) == 1:
            return await self._synthesize_raw(clauses[0], tone=tone)

        results = await asyncio.gather(
            *[self._synthesize_raw(c, tone=tone) for c in clauses],
            return_exceptions=True,
        )
        combined = b""
        for r in results:
            if isinstance(r, bytes):
                combined += r
        return combined

    async def synthesize_to_file(self, text: str, output_path: str):
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"[TTS] saved to {output_path}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare_urdu_for_tts(text: str) -> str:
        """
        Prepare Urdu text for UzmaNeural Edge-TTS.
        Pipeline: time conversion → punctuation softening → phonetic normalisation.
        """
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Drop trailing ellipsis
        text = re.sub(r'\s*\.\.\.\s*$', '', text).strip()

        # Convert HH:MM AM/PM → natural Urdu (e.g. "10:00 AM بجے" → "صبح دس بجے")
        text = VoiceSynthesizer._convert_time_to_urdu(text)

        # ۔ (Urdu full stop) → ، (comma) — ۔ creates a robotic dead halt in UzmaNeural;
        # ، produces a short natural breath instead.
        text = text.replace('۔', '،')
        # ! → ، for the same reason (exclamation causes unnatural sharp rise)
        text = text.replace('!', '،')

        # Phonetic normalisation — longer phrases before shorter ones
        for original, phonetic in sorted(_URDU_PHONETIC_MAP, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(original, phonetic)

        return text

    @staticmethod
    def _convert_time_to_urdu(text: str) -> str:
        """
        Convert English time strings to natural Urdu speech.
        '10:00 AM' → 'صبح دس بجے'   '02:30 PM' → 'شام ساڑھے دو بجے'
        Also consumes a trailing 'بجے' so booking messages don't double up.
        """
        def _replace(m: re.Match) -> str:
            h = int(m.group(1))
            mins = int(m.group(2))
            meridiem = m.group(3).upper()
            if h > 12:
                h -= 12
            if h == 0:
                h = 12
            hour_word = _UR_HOURS.get(h, str(h))
            period = "صبح" if meridiem == "AM" else ("دوپہر" if h <= 4 else "شام")
            if mins == 0:
                return f"{period} {hour_word} بجے"
            elif mins == 30:
                return f"{period} ساڑھے {hour_word} بجے"
            return m.group(0)

        # Match "HH:MM AM/PM" and optionally consume a following "بجے" to avoid duplication
        return re.sub(
            r'\b(\d{1,2}):(\d{2})\s*(AM|PM)\b(\s*بجے)?',
            _replace,
            text,
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _prepare_roman_urdu_for_tts(text: str) -> str:
        """
        Prepare Roman Urdu (Latin-script) text for UzmaNeural Edge-TTS.
        Hard punctuation → ... so the Pakistani voice produces natural conversational pauses.
        """
        text = re.sub(r'[\.!?\,;:]', ' ... ', text)
        text = re.sub(r'(\s*\.\.\.\s*){2,}', ' ... ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s*\.\.\.\s*$', '', text).strip()
        return text

    def _split_clauses(self, text: str) -> list[str]:
        """
        Split text on commas so each clause gets its own synthesis pass.
        Keeps the comma attached to the preceding clause (sounds natural).
        Short fragments are merged with the next one to avoid choppy audio.
        """
        # Split AFTER ، or , when followed by whitespace
        parts = re.split(r'(?<=[،,])\s+', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return [text]

        # Merge fragments that are too short into the next clause
        merged: list[str] = []
        carry = ""
        for part in parts:
            combined = (carry + " " + part).strip() if carry else part
            if len(part) < _MIN_CLAUSE_LEN and part != parts[-1]:
                carry = combined
            else:
                merged.append(combined)
                carry = ""
        if carry:
            if merged:
                merged[-1] = (merged[-1] + " " + carry).strip()
            else:
                merged.append(carry)

        return merged if len(merged) > 1 else [text]

    async def _synthesize_f5tts(self, text: str) -> bytes:
        """Synthesise English text via F5-TTS voice cloning (non-blocking)."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._f5tts_sync, text)
        except Exception as e:
            logger.error(f"[TTS] F5-TTS synthesis error: {e}")
            return b""

    def _f5tts_sync(self, text: str) -> bytes:
        """Blocking F5-TTS inference — runs in thread pool to not block the event loop.
        Reference audio is passed as an in-memory BytesIO (cached at startup, zero disk I/O).
        """
        import soundfile as sf
        import numpy as np

        # Use in-memory ref audio (cached at startup) — torchaudio.load() accepts BytesIO
        ref_source = (
            io.BytesIO(self._f5_ref_audio_bytes)
            if self._f5_ref_audio_bytes
            else settings.voice_sample_path
        )

        wav, sr, _ = self._f5_tts.infer(
            ref_file=ref_source,
            ref_text=self._f5_ref_text,
            gen_text=text,
            speed=1.0,
            remove_silence=True,
        )

        wav = np.array(wav, dtype=np.float32)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.92

        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    async def _synthesize_xtts(self, text: str) -> bytes:
        """Synthesise via XTTS v2 (CPU/GPU, voice cloning from voices/receptionist.wav)."""
        lang = "en" if self._language == "en" and not self._is_urdu_text(text) else "hi"
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._xtts_sync, text, lang)
        except Exception as e:
            logger.error(f"[TTS] XTTS v2 synthesis failed: {e}")
            return b""

    def _xtts_sync(self, text: str, lang: str) -> bytes:
        """Blocking XTTS v2 generation — called in executor thread."""
        import io
        import numpy as np
        wav = self._xtts.tts(
            text=text,
            speaker_wav="voices/receptionist.wav",
            language=lang,
        )
        buf = io.BytesIO()
        import scipy.io.wavfile as wavfile
        wavfile.write(buf, 24000, np.array(wav, dtype=np.float32))
        buf.seek(0)
        return buf.read()

    async def _synthesize_raw(self, text: str, _attempt: int = 0, tone: str = "neutral") -> bytes:
        """Synthesise a single text segment via Edge-TTS. Falls back to pyttsx3 on failure."""
        # Use Urdu voice whenever the text itself is Urdu script, regardless of
        # _language setting — guards against language-detection misfires.
        lang = "ur" if self._is_urdu_text(text) else self._language
        cfg = VOICE_CONFIG.get(lang, VOICE_CONFIG["en"])

        # Tone-based prosody: override rate/pitch per sentence type.
        tone_cfg = TONE_PROSODY.get(lang, {}).get(tone, {})
        rate  = tone_cfg.get("rate",  cfg["rate"])
        pitch = tone_cfg.get("pitch", cfg["pitch"])

        try:
            communicate = edge_tts.Communicate(
                text,
                cfg["voice"],
                rate=rate,
                pitch=pitch,
                volume=cfg["volume"],
            )
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            data = buf.read()
            if not data:
                raise RuntimeError("No audio was received. Please verify that your parameters are correct.")
            return data
        except Exception as e:
            if _attempt == 0:
                logger.warning(f"[TTS] Edge-TTS attempt 1 failed, retrying in 1s: {e}")
                await asyncio.sleep(1.0)
                return await self._synthesize_raw(text, _attempt=1, tone=tone)
            if _attempt == 1:
                logger.warning(f"[TTS] Edge-TTS attempt 2 failed, retrying in 2s: {e}")
                await asyncio.sleep(2.0)
                return await self._synthesize_raw(text, _attempt=2, tone=tone)
            # All 3 Edge-TTS attempts failed — use fallback (gTTS for Urdu, pyttsx3 for English)
            logger.warning(f"[TTS] Edge-TTS failed 3 times, using fallback TTS")
            loop = asyncio.get_event_loop()
            if lang in ("ur", "ro"):
                return await loop.run_in_executor(None, self._synthesize_gtts, text, lang)
            return await loop.run_in_executor(None, self._synthesize_pyttsx3, text, lang)

    def _synthesize_gtts(self, text: str, lang: str) -> bytes:
        """Fallback TTS using gTTS (Google, free). Supports Urdu unlike pyttsx3."""
        try:
            from gtts import gTTS
            lang_map = {"ur": "ur", "ro": "ur", "en": "en"}
            gtts_lang = lang_map.get(lang, "en")
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            data = buf.read()
            logger.debug(f"[TTS] gTTS fallback OK | {len(data)} bytes")
            return data
        except Exception as e:
            logger.error(f"[TTS] gTTS fallback failed: {e}")
            return b""

    def _synthesize_pyttsx3(self, text: str, lang: str) -> bytes:
        """Fallback TTS using pyttsx3 (free, local, always available)."""
        try:
            import tempfile
            import os
            # Windows SAPI5 (pyttsx3) uses COM which must be initialized per-thread.
            # run_in_executor uses a thread pool, so we must CoInitialize here.
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

            engine = pyttsx3.init()
            # Set voice language
            if lang == "ur":
                # Use English voice for Urdu (better quality than Arabic voice)
                voices = engine.getProperty('voices')
                female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
                if female_voices:
                    engine.setProperty('voice', female_voices[0].id)
            # Adjust speech rate (default is usually 200)
            engine.setProperty('rate', 150)
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            with open(temp_path, 'rb') as f:
                data = f.read()
            os.unlink(temp_path)

            logger.debug(f"[TTS] pyttsx3 fallback OK | {len(data)} bytes")
            return data
        except Exception as e:
            logger.error(f"[TTS] pyttsx3 fallback also failed: {e}")
            return b""
