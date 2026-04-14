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
        device = settings.whisper_device
        compute = settings.whisper_compute_type
        logger.info(f"Loading Whisper '{settings.whisper_model}' on {device} ({compute})...")
        try:
            self.model = WhisperModel(
                settings.whisper_model,
                device=device,
                compute_type=compute,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                logger.warning("[STT] CUDA out of memory — retrying Whisper on CPU (int8)")
                self.model = WhisperModel(
                    settings.whisper_model,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise
        self._language = "en"
        logger.info("Whisper loaded.")

    def set_language(self, lang: str):
        """Switch language, e.g. 'en' or 'ur'."""
        self._language = lang

    def transcribe(self, audio: np.ndarray, language: str = None, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy float32 audio array.
        Audio must be mono, 16 kHz, float32 in range [-1, 1].
        Pass language explicitly to avoid race conditions when shared across sessions.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalise if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Language-specific initial prompts bias Whisper toward medical vocabulary
        lang = language or self._language
        if lang == "ur":
            # Urdu medical vocabulary — significantly improves Urdu STT accuracy
            initial_prompt = (
                "یہ ایک ہسپتال کی اپائنٹمنٹ بکنگ کال ہے۔ "
                "الفاظ میں شامل ہیں: اپائنٹمنٹ، ڈاکٹر، شعبہ، مریض، نام، فون نمبر، "
                "کارڈیالوجی، آرتھوپیڈک، نیورولوجی، جنرل میڈیسن، گائناکالوجی، پیڈیاٹرک، "
                "بخار، درد، کھانسی، سر درد، پیٹ درد، آنکھ، دانت، ہڈی، دل، "
                "صبح، دوپہر، شام، تاریخ، وقت، کل، پرسوں۔"
            )
        else:
            initial_prompt = (
                "This is a hospital appointment booking call. "
                "Words include: appointment, doctor, department, patient, "
                "cardiology, orthopedic, pediatric, gynecology, neurology, "
                "general medicine, ophthalmology, blood test, surgery, fever, "
                "pain, cough, headache, morning, afternoon, 9 AM, 2 PM, phone number."
            )

        segments, info = self.model.transcribe(
            audio,
            beam_size=3,           # beam_size=3 gives meaningfully better accuracy vs greedy
            best_of=3,
            language=lang,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            without_timestamps=True,
            condition_on_previous_text=False,  # prevents repetition artifacts
        )

        text = " ".join(seg.text for seg in segments).strip()
        logger.debug(f"[STT] transcript='{text}' lang={info.language} prob={info.language_probability:.2f}")
        return text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file on disk."""
        segments, _ = self.model.transcribe(path, beam_size=5, language=self._language)
        return " ".join(seg.text for seg in segments).strip()
