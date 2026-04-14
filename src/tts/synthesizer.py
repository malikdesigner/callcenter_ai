"""
Text-to-Speech using Edge-TTS (Microsoft Neural Voices).
Uses plain-text Communicate with per-language rate/pitch/volume tuning.

Voice choices:
  EN  — en-US-JennyNeural   (warm, professional phone voice)
  UR  — ur-PK-UzmaNeural    (best Pakistani Urdu neural voice available)
"""

import io
import edge_tts
from loguru import logger


# ── Voice & prosody configuration ─────────────────────────────────────────────

VOICE_CONFIG = {
    "en": {
        "voice":  "en-US-JennyNeural",
        "rate":   "-5%",        # Slightly slower = clearer on phone
        "pitch":  "+0Hz",
        "volume": "+0%",
    },
    "ur": {
        "voice":  "ur-PK-UzmaNeural",
        "rate":   "-10%",       # Urdu needs more breathing room for clarity
        "pitch":  "+1Hz",       # Slightly warmer tone
        "volume": "+0%",
    },
}


class VoiceSynthesizer:
    def __init__(self):
        self._language = "en"
        logger.info(
            f"VoiceSynthesizer initialized — "
            f"EN: {VOICE_CONFIG['en']['voice']} | "
            f"UR: {VOICE_CONFIG['ur']['voice']}"
        )

    def load(self):
        pass  # Edge-TTS is cloud-based; nothing to pre-load

    def set_language(self, lang: str):
        self._language = lang

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text → speech bytes using Edge-TTS with rate/pitch prosody.
        """
        text = text.strip()
        if not text:
            return b""

        cfg = VOICE_CONFIG.get(self._language, VOICE_CONFIG["en"])
        logger.debug(f"[TTS] {self._language.upper()} | {cfg['voice']} | {text[:70]}...")

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
            if data:
                return data
            logger.warning(f"[TTS] Edge-TTS returned empty audio for: {text[:50]}")
            return b""
        except Exception as e:
            logger.error(f"[TTS] Edge-TTS failed: {e}")
            return b""

    async def synthesize_to_file(self, text: str, output_path: str):
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"[TTS] saved to {output_path}")
