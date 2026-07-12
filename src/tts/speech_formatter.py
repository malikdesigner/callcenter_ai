"""
Speech formatting middleware.

Converts LLM text output into speech-optimized form before TTS synthesis.
This layer is what separates "LLM text → TTS" (robotic) from
"conversational speech generation" (human-like).

Pipeline position:
  LLM output → SpeechFormatter → TTS prep → Edge-TTS (Microsoft Neural Voices)

What it does:
  1. Detects sentence tone (micro | question | confirm | apology | neutral)
     Used to drive dynamic TTS prosody (rate/pitch per tone type).
  2. Converts formal written Urdu/Roman Urdu to spoken Pakistani forms.
     E.g. "براہ کرم" → removed, "معذرت" → "سوری", "جناب" → "سر".
  3. Removes verbose constructions that no call-center agent would ever say.

What it does NOT do:
  - Punctuation cleanup (handled by _prepare_urdu_for_tts in synthesizer)
  - Phonetic normalization (آپ کا→آپکا, ڈاکٹر→ڈاکڑ — also in synthesizer)
"""

import re
from typing import Tuple

# ── Formal → Spoken Urdu conversion map ──────────────────────────────────────
# Order matters: longer/more-specific phrases must precede shorter sub-phrases.
_UR_FORMAL_TO_SPOKEN = [
    # Multi-word phrases first
    ("جناب والا",           "سر"),
    ("معذرت خواہ ہوں",     "سوری"),
    ("معافی چاہتا ہوں",    "سوری"),
    ("معافی چاہتی ہوں",    "سوری"),
    ("تشریف لائیں",        "آئیں"),
    ("تشریف لے جائیں",    "جائیں"),
    ("براہ مہربانی",        ""),
    ("براہ کرم",            ""),
    # Single formal words
    ("جناب",               "سر"),
    ("محترم",              ""),
    ("معافی",              "سوری"),
    ("لہذا",               ""),
    ("چنانچہ",             ""),
    ("بہرحال",             ""),
    ("اپوائنٹمنٹ",         "وقت"),
]

# ── Formal → Spoken Roman Urdu conversion map ─────────────────────────────────
_RO_FORMAL_TO_SPOKEN = [
    ("janab wala",            "sir"),
    ("maafi chahta hoon",     "sori"),
    ("mazrat khwah hoon",     "sori"),
    ("appointment",           "waqt"),
]

# ── Tone detection patterns ───────────────────────────────────────────────────

# Micro-response: very short affirmatives / fillers — should be fast
_MICRO_UR = {"جی", "اچھا", "ہاں", "ہمم", "جی جی", "ٹھیک", "اوکے", "اوکے جی"}
_MICRO_RO = {"ji", "acha", "theek", "han", "hmm", "okay", "ok", "ji ji"}
_MICRO_EN = {"ok", "okay", "sure", "right", "got it", "i see", "understood", "alright"}

# Apology — slower, warmer, lower pitch
_APOLOGY_UR = ["سوری", "معذرت", "دستیاب نئیں", "نئیں ہے", "مسئلہ ہے", "نئیں ہوسکا", "خالی نئیں"]
_APOLOGY_RO = ["sori", "sorry", "available nahi", "nahi hai", "masla hai", "nahi ho saka"]
_APOLOGY_EN = ["sorry", "apologize", "unfortunately", "not available", "couldn't", "issue", "unable"]

# Confirmation — assertive, slightly faster
_CONFIRM_UR = ["بالکل جی", "کنفرم", "بکنگ", "محفوظ", "نمبر ہے", "ہو گئی", "ہوگئی", "بک ہو"]
_CONFIRM_RO = ["bilkul ji", "confirm", "booking", "number hai", "ho gai", "hogai", "book ho"]
_CONFIRM_EN = ["confirmed", "booked", "all set", "perfect", "your appointment", "reference is"]

# Question — slightly slower, slight pitch rise
_QUESTION_UR = ["کیا", "کون", "کب", "کہاں", "کیسے", "کونسا", "کتنا", "کونسے", "کدھر"]
_QUESTION_RO = ["kya", "kaun", "kab", "kahan", "kaise", "konsa", "kitna", "konse", "kidhar"]


# ── Tone-to-prosody table ─────────────────────────────────────────────────────
# Urdu/RO base: rate=+5%, pitch=-2Hz (UzmaNeural default in this system).
# English base: rate=-5%, pitch=+0Hz (JennyNeural default).
TONE_PROSODY = {
    "ur": {
        # Natural Pakistani Urdu speech is measured and deliberate — not rushed.
        # UzmaNeural sounds most human at slightly slower rates with a warmer pitch.
        "micro":    {"rate": "+5%",  "pitch": "+0Hz"},   # fillers: جی، اچھا — slightly faster
        "question": {"rate": "-5%",  "pitch": "+2Hz"},   # questions: slower, rise at end
        "confirm":  {"rate": "-2%",  "pitch": "-4Hz"},   # confirmation: clear and warm
        "apology":  {"rate": "-10%", "pitch": "-5Hz"},   # apology: slow, soft
        "empathetic": {"rate": "-8%", "pitch": "-5Hz"},  # empathetic: soft comfort tone
        "neutral":  {"rate": "-3%",  "pitch": "-5Hz"},   # default: natural pace, warm
    },
    "ro": {
        "micro":    {"rate": "+5%",  "pitch": "+0Hz"},
        "question": {"rate": "-5%",  "pitch": "+2Hz"},
        "confirm":  {"rate": "-2%",  "pitch": "-4Hz"},
        "apology":  {"rate": "-10%", "pitch": "-5Hz"},
        "empathetic": {"rate": "-8%", "pitch": "-5Hz"},
        "neutral":  {"rate": "-3%",  "pitch": "-5Hz"},
    },
    "en": {
        "micro":    {"rate": "+2%",  "pitch": "+0Hz"},
        "question": {"rate": "-8%",  "pitch": "+2Hz"},
        "confirm":  {"rate": "-3%",  "pitch": "-1Hz"},
        "apology":  {"rate": "-10%", "pitch": "-2Hz"},
        "empathetic": {"rate": "-12%", "pitch": "-1Hz"},
        "neutral":  {"rate": "-5%",  "pitch": "+0Hz"},
    },
}


class SpeechFormatter:
    """
    Stateless formatter — safe to use as a module-level singleton.
    Call format(text, lang) before every TTS synthesis.
    """

    def format(self, text: str, lang: str, policy_tone: str = None) -> Tuple[str, str]:
        """
        Apply formal→spoken rewriting and detect prosody tone.

        Returns:
            (formatted_text, tone)  where tone ∈ {micro, question, confirm, apology, empathetic, neutral}
        """
        text = text.strip()
        if not text:
            return text, "neutral"

        if lang == "ur":
            text = self._rewrite_ur(text)
        elif lang == "ro":
            text = self._rewrite_ro(text)

        # Allow policy layer to dictate tone (e.g. empathy for symptoms)
        if policy_tone in ["CONFUSED", "ASK_SYMPTOM"]:
            return text, "empathetic"

        tone = self._detect_tone(text, lang)
        return text, tone

    # ── Tone classifier ───────────────────────────────────────────────────────

    def _detect_tone(self, text: str, lang: str) -> str:
        t = text.strip()
        t_lower = t.lower()
        words = set(t_lower.split())
        word_count = len(t.split())

        # Micro: ≤ 3 words AND is a filler/ack
        if word_count <= 3:
            if lang == "ur" and any(w in t for w in _MICRO_UR):
                return "micro"
            if lang == "ro" and (words & _MICRO_RO):
                return "micro"
            if lang == "en" and (words & _MICRO_EN):
                return "micro"

        # Apology
        if lang == "ur" and any(w in t for w in _APOLOGY_UR):
            return "apology"
        if lang == "ro" and any(w in t_lower for w in _APOLOGY_RO):
            return "apology"
        if lang == "en" and any(w in t_lower for w in _APOLOGY_EN):
            return "apology"

        # Confirmation
        if lang == "ur" and any(w in t for w in _CONFIRM_UR):
            return "confirm"
        if lang == "ro" and any(w in t_lower for w in _CONFIRM_RO):
            return "confirm"
        if lang == "en" and any(w in t_lower for w in _CONFIRM_EN):
            return "confirm"

        # Question: terminal ? / ؟ OR starts with question word
        if t.endswith(("؟", "?")):
            return "question"
        if lang == "ur":
            # Question words at start or after ... pause
            for qw in _QUESTION_UR:
                if t.startswith(qw) or f"... {qw}" in t or f" {qw} " in t:
                    return "question"
        if lang == "ro" and any(w in t_lower for w in _QUESTION_RO):
            return "question"
        if lang == "en":
            for qend in ("which", "when", "where", "how", "what", "who"):
                if t_lower.endswith(qend) or f" {qend} " in t_lower:
                    return "question"

        return "neutral"

    # ── Urdu rewriter ─────────────────────────────────────────────────────────

    def _rewrite_ur(self, text: str) -> str:
        for formal, spoken in sorted(_UR_FORMAL_TO_SPOKEN, key=lambda x: len(x[0]), reverse=True):
            text = text.replace(formal, spoken)
        # Collapse spaces left by removals
        text = re.sub(r' {2,}', ' ', text).strip()
        return text

    # ── Roman Urdu rewriter ───────────────────────────────────────────────────

    def _rewrite_ro(self, text: str) -> str:
        t_lower = text.lower()
        for formal, spoken in sorted(_RO_FORMAL_TO_SPOKEN, key=lambda x: len(x[0]), reverse=True):
            idx = t_lower.find(formal)
            while idx != -1:
                text = text[:idx] + spoken + text[idx + len(formal):]
                t_lower = text.lower()
                idx = t_lower.find(formal, idx + max(1, len(spoken)))
        text = re.sub(r' {2,}', ' ', text).strip()
        return text


# Module-level singleton — import and use directly
formatter = SpeechFormatter()
