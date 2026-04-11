"""
Text-to-Speech using Edge-TTS (Cloud-based, Ultra-fast).
Provides professional neural voices with <200ms latency.
"""

import io
import os
import edge_tts
from loguru import logger
from config.settings import settings

class VoiceSynthesizer:
    def __init__(self):
        # We use consistent voices for the receptionist
        # AvaNeural is one of the most natural English voices available
        self.voice_map = {
            "en": "en-US-AvaNeural",
            "ur": "ur-PK-UzmaNeural"
        }
        self._language = "en"
        logger.info(f"VoiceSynthesizer initialized (Primary: Edge-TTS)")

    def load(self):
        """No pre-loading needed for Edge-TTS."""
        pass

    def set_language(self, lang: str):
        """Switch language, e.g. 'en' or 'ur'."""
        self._language = lang

    async def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using Edge-TTS.
        Extremely low latency, cloud-generated.
        """
        if not text.strip():
            return b""

        logger.debug(f"[TTS] synthesizing (Edge-TTS): {text[:80]}...")
        
        voice = self.voice_map.get(self._language, self.voice_map["en"])
        
        try:
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"[TTS] Edge-TTS failed: {e}")
            return b""

    async def synthesize_to_file(self, text: str, output_path: str):
        """Synthesize and write directly to a file."""
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"[TTS] saved to {output_path}")
