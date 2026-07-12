"""
Speech-to-Text using faster-whisper.
Runs locally on GPU. No API key needed.
"""

import os
import importlib

from faster_whisper import WhisperModel
import numpy as np
from loguru import logger
from config.settings import settings

def _register_cuda_dlls():
    """
    On Windows, pip-installed NVIDIA packages put their DLLs in a subfolder
    that Windows doesn't search by default. Register them explicitly so
    ctranslate2 can find cublas64_12.dll and cudnn DLLs at runtime.
    """
    if os.name != "nt":
        return
    for pkg_name in ("nvidia_cublas_cu12", "nvidia_cudnn_cu12", "nvidia_cuda_runtime_cu12"):
        try:
            pkg = importlib.import_module(pkg_name)
            pkg_dir = os.path.dirname(pkg.__file__)
            for sub in ("bin", "lib", ""):
                dll_dir = os.path.join(pkg_dir, sub) if sub else pkg_dir
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
                    logger.debug(f"[STT] Registered DLL dir: {dll_dir}")
        except ImportError:
            pass

_register_cuda_dlls()

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

def _is_hallucination(text: str) -> bool:
    """
    Detect Whisper hallucinations: repeated words/phrases that indicate
    the model is looping on background noise rather than real speech.
    Examples: "Dr. Dr. Dr. Dr." / "the the the" / "ہاں ہاں ہاں ہاں"
    """
    words = text.strip().split()
    if len(words) < 4:
        return False
    # Count how often the most frequent word appears
    from collections import Counter
    counts = Counter(w.lower().strip(".,?!") for w in words)
    top_word, top_count = counts.most_common(1)[0]
    # If one word makes up >60% of all words → hallucination
    if top_count / len(words) > 0.60:
        return True
    # Detect short phrase repetition: "Dr. Smith Dr. Smith Dr. Smith"
    for phrase_len in (2, 3):
        if len(words) >= phrase_len * 3:
            phrases = [" ".join(words[i:i+phrase_len]) for i in range(0, len(words)-phrase_len+1, phrase_len)]
            phrase_counts = Counter(p.lower() for p in phrases)
            top_phrase, top_phrase_count = phrase_counts.most_common(1)[0]
            if top_phrase_count >= 3:
                return True
    return False


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
            error_str = str(e).lower()
            # Fall back to CPU if CUDA memory error OR CUDA library missing
            if device == "cuda" and ("out of memory" in error_str or "cublas" in error_str or "cuda" in error_str):
                logger.warning(f"[STT] CUDA initialization failed ({error_str}) — falling back to CPU (int8)")
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
                   return_language: bool = False, return_confidence: bool = False,
                   phone_mode: bool = False):
        """
        Transcribe a numpy float32 audio array.
        Audio must be mono, 16 kHz, float32 in range [-1, 1].
        Pass language explicitly to avoid race conditions when shared across sessions.
        phone_mode=True uses looser thresholds for 8 kHz upsampled phone audio.
        If return_language=True, returns (text, detected_lang) tuple.
        If return_confidence=True, returns (text, {"avg_logprob": float, "no_speech_prob": float}).
        Both flags may be combined: returns (text, detected_lang, confidence_dict).
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
        seg_logprobs: list[float] = []
        seg_no_speech: list[float] = []
        try:
            for seg in segments:
                t = seg.text.strip()
                if t and not t.startswith("[") and not t.startswith("("):
                    parts.append(t)
                seg_logprobs.append(seg.avg_logprob)
                seg_no_speech.append(seg.no_speech_prob)
        except RuntimeError as e:
            error_str = str(e).lower()
            if "cublas" in error_str or "cuda" in error_str:
                logger.error(f"[STT] CUDA runtime error during transcription — switching model to CPU: {e}")
                self.model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
                # Return empty so STT filter requests a repeat from the user
            else:
                raise

        text = " ".join(parts).strip()

        # Segment-level confidence metrics (averaged across all segments)
        avg_logprob   = sum(seg_logprobs) / len(seg_logprobs) if seg_logprobs else -1.5
        max_no_speech = max(seg_no_speech) if seg_no_speech else 1.0
        confidence = {"avg_logprob": avg_logprob, "no_speech_prob": max_no_speech}

        # Reject Whisper hallucinations (looping repeated words from background noise)
        if text and _is_hallucination(text):
            logger.warning(f"[STT] Hallucination detected, discarding: '{text[:60]}...'")
            text = ""

        # Apply language-specific post-processing corrections
        if lang == "ur" and text:
            text = _normalize_urdu(text)

        logger.debug(
            f"[STT] transcript='{text}' lang={info.language} "
            f"prob={info.language_probability:.2f} "
            f"logprob={avg_logprob:.2f} no_speech={max_no_speech:.2f}"
        )

        if return_language and return_confidence:
            return text, info.language, confidence
        if return_language:
            return text, info.language
        if return_confidence:
            return text, confidence
        return text

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file on disk."""
        segments, _ = self.model.transcribe(path, beam_size=5, language=self._language)
        return " ".join(seg.text for seg in segments).strip()
