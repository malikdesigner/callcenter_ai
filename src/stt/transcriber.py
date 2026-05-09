"""
Speech-to-Text using faster-whisper.
Runs locally on GPU. No API key needed.
"""

from faster_whisper import WhisperModel
import numpy as np
from loguru import logger
from config.settings import settings

# ── Urdu post-transcription corrections ───────────────────────────────────────
# Common Whisper mishearings specific to Pakistani Urdu phone calls.
# Add new entries here as you discover them from call logs.
_UR_CORRECTIONS = {
    # Appointment variants
    "اپورٹ": "اپائنٹمنٹ",
    "اپوائنٹ": "اپائنٹمنٹ",
    "اپائنٹ": "اپائنٹمنٹ",
    "پوائنٹمنٹ": "اپائنٹمنٹ",
    # Mobile/phone
    "مبائل": "موبائل",
    "موبائیل": "موبائل",
    "مبائیل": "موبائل",
    # Common name mishearings
    "نابید": "نوید",
    "نبید": "نوید",
    "نائبید": "نوید",
    # Medical terms
    "کانسی": "کھانسی",
    "کانسہ": "کھانسی",
    "بوخار": "بخار",
    "ہسپتال": "ہسپتال",  # keep normalised form
}

def _normalize_urdu(text: str) -> str:
    for wrong, right in _UR_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


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

    def transcribe(self, audio: np.ndarray, language: str = None, sample_rate: int = 16000,
                   return_language: bool = False, phone_mode: bool = False):
        """
        Transcribe a numpy float32 audio array.
        Audio must be mono, 16 kHz, float32 in range [-1, 1].
        Pass language explicitly to avoid race conditions when shared across sessions.
        phone_mode=True uses looser thresholds for 8 kHz upsampled phone audio.
        If return_language=True, returns (text, detected_lang) tuple instead of just text.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalise if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        lang = language or self._language

        # Phone audio (8 kHz upsampled) has lower SNR — relax rejection thresholds
        # so Whisper doesn't silently discard valid speech as noise.
        if phone_mode:
            no_speech_thr = 0.5          # was 0.8 — phone codec noise fools Whisper
            log_prob_thr  = -1.2         # was -0.7 — accept lower-confidence tokens
            comp_ratio_thr = 2.4         # was 1.9 — phone compression artefacts inflate ratio
        else:
            no_speech_thr  = 0.8
            log_prob_thr   = -0.7
            comp_ratio_thr = 1.9

        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            best_of=5,
            temperature=0,
            language=lang,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            without_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=comp_ratio_thr,
            log_prob_threshold=log_prob_thr,
            no_speech_threshold=no_speech_thr,
        )

        parts = []
        for seg in segments:
            t = seg.text.strip()
            if t and not t.startswith("[") and not t.startswith("("):
                parts.append(t)

        text = " ".join(parts).strip()

        # Apply language-specific post-processing corrections
        if lang == "ur" and text:
            text = _normalize_urdu(text)

        logger.debug(f"[STT] transcript='{text}' lang={info.language} prob={info.language_probability:.2f}")
        if return_language:
            return text, info.language
        return text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file on disk."""
        segments, _ = self.model.transcribe(path, beam_size=5, language=self._language)
        return " ".join(seg.text for seg in segments).strip()
