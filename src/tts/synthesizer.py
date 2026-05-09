"""
Text-to-Speech using Edge-TTS (Microsoft Neural Voices).

Urdu voice strategy:
  ur-PK-UzmaNeural — authentic Pakistani female, professional receptionist tone.
  Key insight: UzmaNeural is phonetically sensitive. Feeding it well-prepared
  conversational Urdu script (not formal, not Roman) sounds dramatically more
  natural than formal Urdu or transliterated text.

Optimisation applied before every Urdu TTS call (_prepare_urdu_for_tts):
  1. Hard punctuation (. ۔ ! ? ، ,) → ... (three-dot soft pause)
     Edge-TTS creates robotic quarter-second halts on ۔/./?  The ellipsis
     produces a much shorter, more conversational breath.
  2. Phonetic normalisation — contracted/phonetic spellings that UzmaNeural
     renders closer to actual Pakistani speech:
       آپ کا → آپکا   |  نہیں → نئیں   |  ڈاکٹر → ڈاکڑ   |  موبائل → موبایل

Rate / pitch:
  +5% rate — slightly faster. Slowing UzmaNeural down *exaggerates* the robotic
  quality between words; a bit faster masks it.
  -2Hz pitch — slightly warmer, closer to a real human female receptionist.
"""

import io
import re
import asyncio
import httpx
import edge_tts
from loguru import logger
from config.settings import settings


# ── Urdu phonetic normalisation map ──────────────────────────────────────────
# Applied before TTS. Order matters: longer phrases first.
_URDU_PHONETIC_MAP = [
    # Contracted postpositions (no space → flows as one phoneme)
    ("آپ کا",   "آپکا"),
    ("آپ کو",   "آپکو"),
    ("آپ کے",   "آپکے"),
    ("آپ کی",   "آپکی"),
    ("آپ نے",   "آپنے"),
    # Phonetic spellings UzmaNeural pronounces more naturally
    ("نہیں",    "نئیں"),
    ("موبائل",  "موبایل"),
    ("ڈاکٹر",   "ڈاکڑ"),
]


# ── Voice configuration ────────────────────────────────────────────────────────
VOICE_CONFIG = {
    "en": {
        "voice":  "en-US-JennyNeural",
        "rate":   "-5%",
        "pitch":  "+0Hz",
        "volume": "+0%",
    },
    "ur": {
        "voice":  "ur-PK-UzmaNeural",
        "rate":   "+5%",    # slightly faster — masks inter-word pauses
        "pitch":  "-2Hz",   # slightly warmer / lower
        "volume": "+0%",
    },
    # Roman Urdu: same Pakistani voice — handles phonetic Latin Urdu
    "ro": {
        "voice":  "ur-PK-UzmaNeural",
        "rate":   "+5%",
        "pitch":  "-2Hz",
        "volume": "+0%",
    },
}

# Minimum clause length (chars) to synthesise as a separate segment.
# Short fragments like "جی" or "ہاں" are kept with the next clause.
_MIN_CLAUSE_LEN = 6


class VoiceSynthesizer:
    def __init__(self):
        self._language = "en"
        self._xtts = None  # reserved for future GPU use via _synthesize_xtts()
        logger.info(
            f"VoiceSynthesizer ready — "
            f"EN: {VOICE_CONFIG['en']['voice']} | "
            f"UR: {VOICE_CONFIG['ur']['voice']} (+5% rate, -2Hz pitch)"
        )

    def load(self):
        pass  # Edge-TTS is cloud-based; nothing to pre-load.
        # XTTS v2 voice cloning is available via _synthesize_xtts() but requires GPU
        # for real-time use (CPU RTF ~2.6× means 37s wait for a 3-sentence greeting).

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

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text → MP3 bytes.
        Priority: ElevenLabs (if configured) → Edge-TTS.
        Urdu script detection is script-based, not just _language-based, so that
        misfires in language detection cannot route Urdu text to the wrong voice.
        """
        text = text.strip()
        if not text:
            return b""

        is_urdu = self._language == "ur" or self._is_urdu_text(text)
        is_roman_urdu = self._language == "ro" and not is_urdu
        if is_urdu:
            text = self._prepare_urdu_for_tts(text)
        elif is_roman_urdu:
            text = self._prepare_roman_urdu_for_tts(text)

        # ── ElevenLabs (primary when configured) ──────────────────────────────
        el_key = settings.elevenlabs_api_key
        el_voice = (
            settings.elevenlabs_voice_id_ur
            if (is_urdu or is_roman_urdu)
            else settings.elevenlabs_voice_id_en
        )
        if el_key and el_voice:
            audio = await self._synthesize_elevenlabs(text, el_key, el_voice)
            if audio:
                return audio
            logger.warning("[TTS] ElevenLabs failed — falling back to Edge-TTS")

        # ── Edge-TTS ──────────────────────────────────────────────────────────
        if is_urdu or is_roman_urdu:
            # Urdu/Roman Urdu: single call — no clause splitting to avoid MP3 gaps.
            lang_key = "ur" if is_urdu else "ro"
            logger.debug(
                f"[TTS] Edge-TTS {lang_key.upper()} | {VOICE_CONFIG[lang_key]['voice']} | {text[:60]}…"
            )
            return await self._synthesize_raw(text)

        # English: clause-splitting for natural comma pauses
        clauses = self._split_clauses(text)
        logger.debug(
            f"[TTS] Edge-TTS EN | {VOICE_CONFIG['en']['voice']} | "
            f"{len(clauses)} clause(s) | {text[:60]}…"
        )
        if len(clauses) == 1:
            return await self._synthesize_raw(clauses[0])

        results = await asyncio.gather(
            *[self._synthesize_raw(c) for c in clauses],
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
        Prepare Urdu text for UzmaNeural Edge-TTS:
        1. Replace all hard punctuation with ... (three-dot soft pause).
           Edge-TTS creates robotic 250ms halts on ۔/./!/? — ellipsis is much shorter.
        2. Apply phonetic normalisation for natural Pakistani pronunciation.
        """
        # All hard punctuation → soft pause (ellipsis)
        text = re.sub(r'[۔\.!؟\?،,،]', ' ... ', text)
        # Collapse consecutive ellipses into one
        text = re.sub(r'(\s*\.\.\.\s*){2,}', ' ... ', text)
        # Normalise whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Drop trailing ellipsis (sounds unnatural at the very end)
        text = re.sub(r'\s*\.\.\.\s*$', '', text).strip()

        # Phonetic normalisation — longer phrases before shorter ones
        for original, phonetic in sorted(_URDU_PHONETIC_MAP, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(original, phonetic)

        return text

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

    async def _synthesize_elevenlabs(self, text: str, api_key: str, voice_id: str) -> bytes:
        """
        Synthesise via ElevenLabs eleven_multilingual_v2.
        Free tier: 10,000 chars/month. Dramatically more natural than Edge-TTS for Urdu.
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.4,          # slight variation = more human
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True,
            },
        }
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                logger.debug(f"[TTS] ElevenLabs OK | {len(resp.content)} bytes | {text[:50]}…")
                return resp.content
            if resp.status_code == 401:
                logger.error("[TTS] ElevenLabs: invalid API key")
            elif resp.status_code == 429:
                logger.warning("[TTS] ElevenLabs: monthly quota exhausted")
            else:
                logger.error(f"[TTS] ElevenLabs: HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"[TTS] ElevenLabs request failed: {e}")
        return b""

    async def _synthesize_xtts(self, text: str) -> bytes:
        """Synthesise via XTTS v2 (CPU, voice cloning from voices/receptionist.wav)."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._xtts_sync, text)
        except Exception as e:
            logger.error(f"[TTS] XTTS v2 synthesis failed: {e}")
            return b""

    def _xtts_sync(self, text: str) -> bytes:
        """Blocking XTTS v2 generation — called in executor thread."""
        import io
        import numpy as np
        wav = self._xtts.tts(
            text=text,
            speaker_wav="voices/receptionist.wav",
            language="hi",  # "hi" handles Roman Urdu with South Asian accent
        )
        buf = io.BytesIO()
        import scipy.io.wavfile as wavfile
        wavfile.write(buf, 24000, np.array(wav, dtype=np.float32))
        buf.seek(0)
        return buf.read()

    async def _synthesize_raw(self, text: str, _attempt: int = 0) -> bytes:
        """Synthesise a single text segment via Edge-TTS. Retries once on failure."""
        # Use Urdu voice whenever the text itself is Urdu script, regardless of
        # _language setting — guards against language-detection misfires.
        lang = "ur" if self._is_urdu_text(text) else self._language
        cfg = VOICE_CONFIG.get(lang, VOICE_CONFIG["en"])
        try:
            communicate = edge_tts.Communicate(
                text,
                cfg["voice"],
                rate=cfg["rate"],
                pitch=cfg["pitch"],
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
                logger.warning(f"[TTS] Edge-TTS attempt 1 failed, retrying: {e}")
                await asyncio.sleep(0.4)
                return await self._synthesize_raw(text, _attempt=1)
            logger.error(f"[TTS] Edge-TTS failed: {e}")
            return b""
