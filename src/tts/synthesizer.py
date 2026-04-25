"""
Text-to-Speech using Edge-TTS (Microsoft Neural Voices).

What is Edge-TTS?
  Microsoft's cloud neural TTS (free, no API key, uses Azure under the hood).
  It is the best FREE Urdu voice available. Paid alternatives (Google Cloud TTS,
  Azure TTS directly) offer SSML fine-tuning but require a billing account.

Voice options for Urdu:
  ur-PK-UzmaNeural  — Female (current). Clear, professional, receptionist-like.
  ur-PK-AsadNeural  — Male. Slightly more natural prosody for some speakers.
  Change URDU_VOICE below to switch.

Punctuation / pause fix:
  Edge-TTS HTML-escapes all text, so SSML <break> tags cannot be injected via the
  standard API.  The workaround: split on commas and synthesise each clause
  separately, then concatenate the MP3 bytes.  The MP3 frame boundary between
  clauses creates a natural pause — longer than a comma, shorter than a full stop.
  Full stops are already handled at the sentence-split level in call_handler.py.
"""

import io
import re
import asyncio
import edge_tts
from loguru import logger

# Urdu punctuation equivalents for English marks
_EN_TO_UR_PUNCT = [
    # Question mark — always safe to replace
    (re.compile(r'\?'), '؟'),
    # Comma — skip if between digits (e.g. 1,000)
    (re.compile(r',(?!\s*\d)'), '،'),
    # Period — skip decimals (0.5) and common English abbreviations (Dr. Mr.)
    (re.compile(r'(?<![A-Za-z\d])\.(?!\d)(?!\s*[A-Z][a-z])'), '۔'),
]


# ── Voice configuration ────────────────────────────────────────────────────────
# To switch Urdu voice change URDU_VOICE:
#   "ur-PK-UzmaNeural"  — female (default, clear and professional)
#   "ur-PK-AsadNeural"  — male  (try this if Uzma sounds unnatural for your use)

URDU_VOICE = "ur-PK-UzmaNeural"

VOICE_CONFIG = {
    "en": {
        "voice":  "en-US-JennyNeural",
        "rate":   "-5%",
        "pitch":  "+0Hz",
        "volume": "+0%",
    },
    "ur": {
        "voice":  URDU_VOICE,
        "rate":   "-12%",   # Slower = more natural pause between words
        "pitch":  "+0Hz",   # Neutral pitch — UzmaNeural already sounds warm
        "volume": "+0%",
    },
}

# Minimum clause length (chars) to synthesise as a separate segment.
# Short fragments like "جی" or "ہاں" are kept with the next clause.
_MIN_CLAUSE_LEN = 6


class VoiceSynthesizer:
    def __init__(self):
        self._language = "en"
        logger.info(
            f"VoiceSynthesizer ready — "
            f"EN: {VOICE_CONFIG['en']['voice']} | "
            f"UR: {VOICE_CONFIG['ur']['voice']}"
        )

    def load(self):
        pass  # Edge-TTS is cloud-based; nothing to pre-load

    def set_language(self, lang: str):
        self._language = lang

    # ── Public API ─────────────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text → MP3 bytes with natural pauses at commas.

        Normalises Urdu punctuation, then splits on ، / , and synthesises each
        clause separately.  Concatenated MP3 frames produce a natural pause at
        each boundary without any audio-processing library.
        """
        text = text.strip()
        if not text:
            return b""

        if self._language == "ur":
            text = self._normalize_urdu_punctuation(text)

        clauses = self._split_clauses(text)
        logger.debug(
            f"[TTS] {self._language.upper()} | {VOICE_CONFIG[self._language]['voice']} | "
            f"{len(clauses)} clause(s) | {text[:60]}…"
        )

        if len(clauses) == 1:
            return await self._synthesize_raw(clauses[0])

        # Synthesise all clauses concurrently for lower latency
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
    def _normalize_urdu_punctuation(text: str) -> str:
        """
        Convert English punctuation marks to Urdu equivalents.
        Only fires for Urdu text; skips numbers and English abbreviations.
        """
        for pattern, replacement in _EN_TO_UR_PUNCT:
            text = pattern.sub(replacement, text)
        # Normalise spacing: punctuation should be followed by one space, not zero or many
        text = re.sub(r'([،۔؟])\s*', r'\1 ', text)
        return text.strip()

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

    async def _synthesize_raw(self, text: str) -> bytes:
        """Synthesise a single text segment via Edge-TTS."""
        cfg = VOICE_CONFIG.get(self._language, VOICE_CONFIG["en"])
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
                logger.warning(f"[TTS] Empty audio for: {text[:50]}")
            return data
        except Exception as e:
            logger.error(f"[TTS] Edge-TTS failed: {e}")
            return b""
