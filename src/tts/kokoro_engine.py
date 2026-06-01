"""
Kokoro TTS engine — local, free, fast, high quality.

Supported languages: English (American/British), Spanish, French, Hindi, etc.
Urdu script: NOT supported by Kokoro — falls back to Edge-TTS automatically.

Install:
  pip install kokoro soundfile
  # Windows also needs espeak-ng: https://github.com/espeak-ng/espeak-ng/releases

Voices (English female, suitable for Pakistani receptionist):
  bf_alice  — British female, natural and clear
  bf_emma   — British female, warm
  af_bella  — American female, expressive
  af_heart  — American female, gentle
"""

import io
import asyncio
from loguru import logger


class KokoroEngine:
    """
    Wraps kokoro.KPipeline for async use.
    Falls back gracefully if kokoro/espeak-ng is not installed.
    """

    _instance = None  # singleton: load once, reuse everywhere

    def __init__(self, voice: str = "bf_alice", speed: float = 1.0):
        self._voice  = voice
        self._speed  = speed
        self._pipeline_en = None
        self._pipeline_hi = None  # Hindi — closer to Urdu phonetics
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from kokoro import KPipeline
            self._pipeline_en = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
            # Attempt Hindi pipeline for Roman Urdu (shares phoneme set)
            try:
                self._pipeline_hi = KPipeline(lang_code="h", repo_id="hexgrad/Kokoro-82M")
            except Exception:
                self._pipeline_hi = None
            self._available = True
            logger.info(f"[Kokoro] Loaded — voice={self._voice}")
        except ImportError:
            logger.warning("[Kokoro] Not installed. Run: pip install kokoro soundfile")
        except Exception as e:
            logger.warning(f"[Kokoro] Failed to load: {e}. Edge-TTS will be used instead.")

    @property
    def available(self) -> bool:
        return self._available

    def synthesize_sync(self, text: str, lang: str = "en") -> bytes:
        """
        Blocking synthesis — call from thread executor.
        Returns WAV bytes (24 kHz mono).
        lang: "en" | "ro" (Roman Urdu, uses Hindi pipeline if available)
        """
        if not self._available:
            return b""

        try:
            import numpy as np
            import soundfile as sf

            pipeline = self._pipeline_en
            if lang == "ro" and self._pipeline_hi:
                pipeline = self._pipeline_hi

            audio_chunks = []
            for _, _, audio in pipeline(text, voice=self._voice, speed=self._speed):
                audio_chunks.append(audio)

            if not audio_chunks:
                return b""

            audio_data = np.concatenate(audio_chunks)
            buf = io.BytesIO()
            sf.write(buf, audio_data, 24000, format="WAV")
            buf.seek(0)
            return buf.read()

        except Exception as e:
            logger.error(f"[Kokoro] Synthesis failed: {e}")
            return b""

    async def synthesize(self, text: str, lang: str = "en") -> bytes:
        """Async wrapper — runs blocking synthesis in thread executor."""
        if not self._available or not text.strip():
            return b""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_sync, text, lang)


# Module-level singleton
_kokoro: KokoroEngine | None = None


def get_kokoro(voice: str = "bf_alice", speed: float = 1.0) -> KokoroEngine:
    global _kokoro
    if _kokoro is None:
        _kokoro = KokoroEngine(voice=voice, speed=speed)
    return _kokoro
