"""
Rule-based intent detection and language identification.
Runs before the LLM to guide the state machine — no API call needed.
"""

import re
from typing import Optional


# ── Language detection ─────────────────────────────────────────────────────────

_LANG_SWITCH_EN = [
    "english please", "speak english", "in english", "switch to english",
    "english mein", "english me", "talk in english", "english bol",
]
_LANG_SWITCH_UR = [
    "اردو", "اردو میں", "اردو بولیں", "اردو میں بات", "urdu please", "urdu mein",
]


def detect_language_preference(text: str) -> Optional[str]:
    """
    Detect whether the user prefers Urdu or English at the start of a bilingual call.
    Returns 'ur', 'en', or None (ambiguous).
    """
    t = text.strip()
    t_lower = t.lower()

    ur_signals = [
        "urdu", "اردو", "urdoo", "urdou", "اردو میں", "urdu mein", "urdu me",
        "urdhu", "urdo", "oordoo", "ardo", "ordoo", "orda", "urdū",
        # Common Whisper phonetic mis-transcriptions of "اردو" on phone audio
        "or do", "ordo", "ardo", "urdo", "oordo", "erdo", "ordu",
        "اردو میں بات", "اردو بولیں", "اردو چاہیے",
    ]
    if any(s in t_lower or s in t for s in ur_signals):
        return "ur"

    en_signals = [
        "english", "eng", "angrezi", "انگریزی", "english mein", "english me",
        "inglis", "inglish", "ingrezi", "angrezee",
    ]
    if any(s in t_lower for s in en_signals):
        return "en"

    # Infer from script: any Urdu/Arabic characters → user is speaking Urdu
    arabic_chars = sum(1 for c in t if '؀' <= c <= 'ۿ')
    if arabic_chars >= 1:
        return "ur"

    # Do NOT assume English from Latin text alone — Whisper hallucinates Latin
    # words from Urdu speech (e.g. "Geodude"). Let the caller use audio_language
    # (Whisper's detected language) as the tiebreaker instead.
    return None


def detect_language_switch(text: str) -> Optional[str]:
    """
    Returns 'en', 'ur', or None if no switch is detected.
    Also auto-detects from Unicode script when no explicit command given.
    """
    t_lower = text.lower()

    # Explicit commands first
    if any(p in t_lower for p in _LANG_SWITCH_EN):
        return "en"
    if any(p in text for p in _LANG_SWITCH_UR):
        return "ur"

    # Script-based detection: Urdu/Arabic characters \u2192 Urdu.
    # Do NOT auto-detect English from absence of Arabic script \u2014 a Urdu-speaking
    # user routinely says numbers, names, and medical terms in Latin chars.
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars > 3:
        return "ur"

    return None


# ── Intent constants ───────────────────────────────────────────────────────────

_CONFIRM_EN = {
    "yes", "yep", "yeah", "correct", "right", "ok", "okay", "sure",
    "confirm", "go ahead", "book it", "please book", "sounds good",
    "perfect", "great", "fine", "agreed", "proceed", "that's right",
    "absolutely", "definitely", "do it", "book", "alright",
}
_CONFIRM_UR = {
    "ہاں", "جی", "جی ہاں", "ٹھیک ہے", "بالکل", "درست", "کریں",
    "بک کریں", "ہاں جی", "ٹھیک", "اچھا", "بک کر دیں", "کنفرم",
    "ضرور", "چلو", "بھیج دو", "ہاں،",
    # Common Whisper mis-transcriptions of ہاں جی
    "ہانجی", "ہانچی", "آنجی", "آنچی", "ہاں جی۔", "جی صحیح",
    "جی ٹھیک", "جی بالکل", "جی ضرور", "جی اچھا",
}
_DENY_EN = {
    "no", "nope", "wrong", "incorrect", "change", "wait",
    "hold on", "not right", "actually", "different", "mistake",
}
_DENY_UR = {
    "نہیں", "غلط", "نا", "رکیں", "بدلو", "بدلیں", "درست نہیں",
    "نہیں،", "نہیں۔", "نای",
}

_DEPT_SYMPTOMS_EN = {
    "heart", "cardiac", "chest", "bone", "bones", "joint", "joints",
    "fracture", "child", "children", "baby", "pregnancy", "brain",
    "headache", "migraine", "eye", "eyes", "fever", "pain", "cough",
    "cardiology", "orthopedic", "pediatric", "gynecology", "neurology",
    "general", "doctor", "department", "specialist", "medicine",
    "stomach", "abdomen", "skin", "dental", "teeth", "ear", "nose",
}
_DEPT_SYMPTOMS_UR = {
    "دل", "سینہ", "ہڈی", "جوڑ", "بچہ", "بچے", "حمل", "دماغ",
    "سر درد", "آنکھ", "بخار", "درد", "کھانسی", "ڈاکٹر", "شعبہ",
    "پیٹ", "دانت", "کان", "ناک", "جلد",
}

_DATE_EN = {
    "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "next week", "next monday",
    "next tuesday", "next wednesday", "next thursday", "next friday",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
}
_DATE_UR = {
    "کل", "پرسوں", "آج", "پیر", "منگل", "بدھ", "جمعرات",
    "جمعہ", "ہفتہ", "اتوار", "اگلے ہفتے",
}

_TIME_EN = {"am", "pm", "morning", "afternoon", "evening", "o'clock", "half past", "half"}
_TIME_UR = {"صبح", "دوپہر", "شام", "رات", "بجے", "بج"}

# ── Symptom → Department keyword map ──────────────────────────────────────────
# Maps individual keywords to canonical department names.
# Used to pre-fill department without waiting for LLM to extract it.
_SYMPTOM_DEPT_UR: dict[str, str] = {
    # General / Internal Medicine
    "بخار": "general", "بوخار": "general", "نزلہ": "general", "زکام": "general",
    "کھانسی": "general", "کف": "general", "کمزوری": "general", "تھکاوٹ": "general",
    "شوگر": "general", "ذیابیطس": "general", "بلڈ پریشر": "general", "بی پی": "general",
    "الرجی": "general", "خارش": "general", "جلد": "general",
    # Cardiology
    "دل": "cardiology", "سینہ": "cardiology", "سینے": "cardiology",
    "دھڑکن": "cardiology", "سینے میں درد": "cardiology",
    # Orthopedic
    "ہڈی": "orthopedic", "ہڈیاں": "orthopedic", "جوڑ": "orthopedic",
    "گھٹنا": "orthopedic", "کمر": "orthopedic", "ریڑھ": "orthopedic",
    "فریکچر": "orthopedic", "موچ": "orthopedic",
    # Pediatric
    "بچہ": "pediatric", "بچے": "pediatric", "بچی": "pediatric",
    "شیرخوار": "pediatric", "بچوں": "pediatric",
    # Gynecology
    "حمل": "gynecology", "ماہواری": "gynecology", "خواتین": "gynecology",
    # Neurology
    "مرگی": "neurology", "فالج": "neurology", "یادداشت": "neurology",
    "سردرد": "neurology", "آدھے سر": "neurology",
}
_SYMPTOM_DEPT_EN: dict[str, str] = {
    "fever": "general", "cough": "general", "cold": "general", "flu": "general",
    "sugar": "general", "diabetes": "general", "bp": "general", "pressure": "general",
    "heart": "cardiology", "cardiac": "cardiology", "chest": "cardiology",
    "bone": "orthopedic", "joint": "orthopedic", "knee": "orthopedic", "fracture": "orthopedic",
    "child": "pediatric", "baby": "pediatric", "infant": "pediatric",
    "pregnancy": "gynecology", "gynae": "gynecology",
    "brain": "neurology", "migraine": "neurology", "epilepsy": "neurology",
}


def detect_symptom_department(text: str) -> Optional[str]:
    """
    Scans text for symptom keywords and returns the most likely department.
    Returns None if no match found.
    """
    # Multi-word phrases first (more specific)
    for phrase in sorted(_SYMPTOM_DEPT_UR, key=len, reverse=True):
        if phrase in text:
            return _SYMPTOM_DEPT_UR[phrase]
    t_lower = text.lower()
    for phrase in sorted(_SYMPTOM_DEPT_EN, key=len, reverse=True):
        if phrase in t_lower:
            return _SYMPTOM_DEPT_EN[phrase]
    return None

_DIGIT_DATE_RE = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
    re.IGNORECASE,
)


# ── Main detector ──────────────────────────────────────────────────────────────

def detect_intent(text: str, flow_state: str = "", current_language: str = "en") -> str:
    """
    Classify user message into one of:
      provide_name | provide_phone | describe_symptoms | choose_date
      choose_time | confirm | deny | change_language_en | change_language_ur | unclear

    flow_state is the current conversation state (e.g. 'collect_name') which
    helps resolve ambiguous short responses.

    current_language prevents mis-classifying normal Urdu/English responses as
    language-switch requests — only trigger a switch when the detected script
    actually differs from the active language.
    """
    t = text.strip()
    t_lower = t.lower()
    words = set(t_lower.split())

    # 1. Language switch — only if the detected language differs from current
    lang = detect_language_switch(t)
    if lang == "en" and current_language != "en":
        return "change_language_en"
    if lang == "ur" and current_language != "ur":
        return "change_language_ur"

    # 2. Confirmation
    if words & _CONFIRM_EN or any(w in t for w in _CONFIRM_UR):
        # Don't mis-classify "no" → confirm when denial words also present
        if not (words & _DENY_EN or any(w in t for w in _DENY_UR)):
            return "confirm"

    # 3. Denial / correction
    if words & _DENY_EN or any(w in t for w in _DENY_UR):
        return "deny"

    # 4. Phone: mostly digits, 7-15 total
    digits = "".join(c for c in t if c.isdigit())
    non_space = t.replace(" ", "").replace("-", "")
    if 7 <= len(digits) <= 15 and len(non_space) > 0:
        digit_ratio = len(digits) / len(non_space)
        if digit_ratio >= 0.5:
            return "provide_phone"

    # 5. Time (check before date to catch "9 am tomorrow" → time first)
    if words & _TIME_EN or any(w in t for w in _TIME_UR):
        return "choose_time"

    # 6. Date
    if words & _DATE_EN or any(w in t for w in _DATE_UR) or _DIGIT_DATE_RE.search(t):
        return "choose_date"

    # 7. Symptoms / department
    if words & _DEPT_SYMPTOMS_EN or any(w in t for w in _DEPT_SYMPTOMS_UR):
        return "describe_symptoms"

    # 8. Likely a name: short response, no digits, asked for name
    if flow_state == "collect_name":
        word_list = [w for w in t.split() if w]
        if 1 <= len(word_list) <= 4 and not any(c.isdigit() for c in t):
            return "provide_name"

    return "unclear"
