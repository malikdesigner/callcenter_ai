"""
Text-to-Speech using Coqui XTTS v2.
Clones voice from a WAV sample. Supports English and Urdu.
No API key needed — runs fully local on GPU.
"""

import io
import os
import torch
import numpy as np
import soundfile as sf
from loguru import logger
from config.settings import settings


def _ensure_wav(path: str) -> str:
    """
    XTTS v2 requires a WAV file for speaker_wav.
    If the input is MP3/M4A/etc., convert it once and cache the WAV.
    Returns the path to a valid WAV file.
    """
    if path.lower().endswith(".wav"):
        return path

    wav_path = os.path.splitext(path)[0] + "_converted.wav"
    if os.path.exists(wav_path):
        return wav_path

    logger.info(f"Converting voice sample to WAV: {path} → {wav_path}")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(wav_path, format="wav")
        logger.info("Conversion done.")
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}. Trying ffmpeg fallback...")
        os.system(f'ffmpeg -y -i "{path}" -ar 24000 -ac 1 "{wav_path}"')

    return wav_path


class VoiceSynthesizer:
    def __init__(self):
        raw_path = settings.voice_sample_path
        self.voice_sample = _ensure_wav(raw_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._language = "en"
        self._tts = None  # Lazy-loaded on first use

    def _load(self):
        if self._tts is not None:
            return
        from TTS.api import TTS
        logger.info("Loading XTTS v2 (first run downloads ~2 GB)...")
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        logger.info("XTTS v2 loaded.")

    def set_language(self, lang: str):
        """Switch language, e.g. 'en' or 'ur'."""
        self._language = lang

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using the cloned voice.
        Returns WAV bytes ready to stream over WebSocket.
        """
        self._load()

        if not text.strip():
            return b""

        logger.debug(f"[TTS] synthesizing: {text[:80]}...")

        wav: list = self._tts.tts(
            text=text,
            speaker_wav=self.voice_sample,
            language=self._language,
        )

        wav_array = np.array(wav, dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, wav_array, samplerate=24000, format="WAV")
        buf.seek(0)
        return buf.read()

    def synthesize_to_file(self, text: str, output_path: str):
        """Synthesize and write directly to a file."""
        self._load()
        self._tts.tts_to_file(
            text=text,
            speaker_wav=self.voice_sample,
            language=self._language,
            file_path=output_path,
        )
        logger.info(f"[TTS] saved to {output_path}")
