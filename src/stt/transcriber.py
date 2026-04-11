"""
Speech-to-Text using faster-whisper.
Runs locally on GPU. No API key needed.
"""

from faster_whisper import WhisperModel
import numpy as np
from loguru import logger
from config.settings import settings


class Transcriber:
    def __init__(self):
        logger.info(
            f"Loading Whisper '{settings.whisper_model}' on {settings.whisper_device}..."
        )
        self.model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        self._language = "en"
        logger.info("Whisper loaded.")

    def set_language(self, lang: str):
        """Switch language, e.g. 'en' or 'ur'."""
        self._language = lang

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy float32 audio array.
        Audio must be mono, 16 kHz, float32 in range [-1, 1].
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalise if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # initial_prompt biases Whisper toward medical vocabulary,
        # preventing mishearing e.g. "appointment" → "apartment"
        medical_prompt = (
            "This is a hospital appointment booking call. "
            "Words include: appointment, doctor, department, patient, "
            "cardiology, orthopedic, pediatric, gynecology, neurology, "
            "general medicine, eye, ophthalmology, blood test, surgery, "
            "morning, afternoon, 9 AM, 10 AM, 2 PM, 3 PM, phone number."
        )

        segments, info = self.model.transcribe(
            audio,
            beam_size=1,           # greedy decoding — 3x faster, fine for conversation
            best_of=1,
            language=self._language,
            initial_prompt=medical_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=200),
            without_timestamps=True,
        )

        text = " ".join(seg.text for seg in segments).strip()
        logger.debug(f"[STT] transcript='{text}' lang={info.language} prob={info.language_probability:.2f}")
        return text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file on disk."""
        segments, _ = self.model.transcribe(path, beam_size=5, language=self._language)
        return " ".join(seg.text for seg in segments).strip()
