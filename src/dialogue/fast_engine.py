"""
Fast Dialogue Engine — instant responses for structured booking data.

Standard booking flow: name → phone → [Ollama for symptoms/doctor] → date → time → book

This engine handles the structured steps (name, phone, date, time, booking confirmation)
without any LLM call, giving sub-100ms responses. Ollama is only called for steps that
require actual reasoning: symptom understanding, doctor recommendation, or anything
the user says that deviates from the expected answer (questions, corrections, confusion).
"""

import re
from datetime import date, timedelta
from typing import Optional

from loguru import logger
from sqlmodel import Session, select

from src.appointment.models import Doctor, engine
from src.appointment.booking import BookingSystem

# ── Patterns ──────────────────────────────────────────────────────────────────

# Pakistani mobile numbers: 03xx-xxxxxxx (11 digits) or +9203xx... or 3xx-xxxxxxx
_PHONE_RE = re.compile(
    r'(?:\+?92|0)?'              # optional country code
    r'(3\d{2})'                  # network prefix: 3xx
    r'[\s\-]?'
    r'(\d{7})',                  # 7 digits
)
# Also catch plain 11-digit strings like "03001234567"
_PHONE_11_RE = re.compile(r'\b(0\d{10})\b')

_URDU_QUESTION_WORDS = {
    "کیوں", "کیا", "کیسے", "کون", "کب", "کہاں",
    "مطلب", "سمجھ", "بتائیں", "بتاؤ", "ہے کیا",
}
_EN_QUESTION_WORDS = {
    "why", "what", "how", "who", "when", "where", "which",
    "explain", "tell me", "don't understand",
}

# Relative date words → day offset
_URDU_REL_DATE = {
    "آج": 0, "اج": 0,
    "کل": 1, "کال": 1,
    "پرسوں": 2, "پرسو": 2,
    "نرسوں": 3,
}
_EN_REL_DATE = {
    "today": 0, "aaj": 0,
    "tomorrow": 1, "kal": 1, "kl": 1,
    "day after": 2, "parso": 2, "parson": 2,
}

# Urdu day names → isoweekday (Mon=1)
_URDU_DAY = {
    "پیر": 1, "منگل": 2, "بدھ": 3, "جمعرات": 4,
    "جمعہ": 5, "ہفتہ": 6, "اتوار": 7,
}
_EN_DAY = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
    "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7,
    "somwar": 1, "mangal": 2, "budh": 3, "jumerat": 4,
    "jumma": 5, "hafta": 6, "itwar": 7,
}

# Urdu number words → integer
_URDU_NUMS = {
    "ایک": 1, "دو": 2, "تین": 3, "چار": 4, "پانچ": 5,
    "چھ": 6, "چھے": 6, "سات": 7, "آٹھ": 8, "نو": 9,
    "دس": 10, "گیارہ": 11, "بارہ": 12,
}
_URDU_MERIDIEM = {"صبح": "AM", "دوپہر": "PM", "شام": "PM", "رات": "PM"}

# Words that indicate the user is NOT answering the booking question
_CONFUSION_SIGNALS = {
    "ur": {"مطلب", "سمجھ", "سمجھا", "سمجھی", "کیا مراد", "نہیں پتا", "پتا نہیں",
           "کیوں", "کیسے", "کب", "کیا ہے", "بتائیں", "بتاؤ", "کون"},
    "en": {"what", "why", "how", "explain", "confused", "don't know", "don't understand",
           "not sure", "no idea", "tell me", "who", "which"},
}

# Name filler words to strip
_NAME_FILLERS = {
    "ur": ["میرا نام ہے", "میرا نام", "نام ہے", "نام", "میں", "جی ہاں", "جی", "ہاں", "سر", "صاحب", "ہے", "ہوں"],
    "en": ["my name is", "i am", "i'm", "this is", "name is", "it's", "its"],
    "ro": ["mera naam hai", "mera naam", "naam hai", "naam", "main", "ji haan", "ji", "haan", "sar", "sahab", "hai", "hoon"],
}


class FastDialogueEngine:
    """Instant responses for structured booking steps. No LLM required."""

    FAST_STATES = {"collect_name", "collect_phone", "collect_date", "collect_time", "confirm_booking"}

    # Affirmative words in all three languages — user is saying YES
    _YES_WORDS = {
        # Urdu
        "ہاں", "جی", "جی ہاں", "ٹھیک ہے", "بالکل", "ضرور", "ٹھیک", "کریں", "بک کریں",
        "کنفرم", "اچھا", "چلے گا", "ٹھیک رہے گا",
        # Roman Urdu
        "haan", "ji", "ji haan", "theek hai", "bilkul", "zaroor", "theek", "karein",
        "book karein", "confirm", "acha", "chalega", "theek rahega",
        # English
        "yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "go ahead", "please do",
        "that's fine", "sounds good", "perfect", "great",
    }

    def __init__(self, language: str = "ur"):
        self.language = language
        self._booking = BookingSystem()

    def try_handle(
        self,
        transcript: str,
        flow_state: str,
        collected: dict,
    ) -> Optional[dict]:
        """
        Returns {"speech": str, "action": str, "data": dict} or None.
        None means: let Ollama handle this turn.
        """
        if flow_state not in self.FAST_STATES:
            return None

        t = transcript.strip()
        if not t:
            return None

        # Confirmation state — user is saying yes/no to booking summary
        if flow_state == "confirm_booking":
            return self._handle_confirmation(t, collected)

        # If the user seems confused or is asking a question → Ollama
        if self._seems_confused(t):
            logger.debug(f"[Fast] Detected confusion/question — deferring to Ollama")
            return None

        if flow_state == "collect_name":
            return self._handle_name(t, collected)
        if flow_state == "collect_phone":
            return self._handle_phone(t, collected)
        if flow_state == "collect_date":
            return self._handle_date(t, collected)
        if flow_state == "collect_time":
            return self._handle_time(t, collected)

        return None

    # ── Confirmation handler ───────────────────────────────────────────────────

    def _handle_confirmation(self, text: str, collected: dict) -> Optional[dict]:
        """
        Handles the confirm_booking state.
        YES → action='save' (agent executes booking instantly).
        NO / unclear → None (Ollama handles the correction).
        """
        low = text.lower().strip()
        # Check for any yes-word
        for word in self._YES_WORDS:
            if word in low or word in text:
                logger.info("[Fast] Confirmation received — triggering booking")
                return {"speech": "", "action": "save", "data": {}}
        # Negative signals → let Ollama handle (correction or clarification needed)
        _NO_WORDS = {"نہیں", "نئیں", "نہ", "غلط", "بدلو", "no", "nahi", "nahi", "galat", "badlo", "change"}
        for word in _NO_WORDS:
            if word in low or word in text:
                return None  # Ollama handles the correction
        # Ambiguous — let Ollama decide
        return None

    # ── Confusion detection ────────────────────────────────────────────────────

    def _seems_confused(self, text: str) -> bool:
        """True if the user is asking a question or seems confused."""
        low = text.lower()
        signals = _CONFUSION_SIGNALS.get(self.language, set()) | _CONFUSION_SIGNALS.get("en", set())
        # Word-boundary check
        for word in signals:
            if re.search(rf'\b{re.escape(word)}\b', low):
                return True
        # Urdu signals (no word boundaries needed — script based)
        if self.language in ("ur", "ro"):
            for word in _CONFUSION_SIGNALS["ur"]:
                if word in text:
                    return True
        # Short question-like transcript (e.g. "کیا؟")
        if text.endswith("?") or text.endswith("؟"):
            return True
        return False

    # ── Name handler ──────────────────────────────────────────────────────────

    def _handle_name(self, text: str, collected: dict) -> Optional[dict]:
        name = self._extract_name(text)
        if not name:
            return None  # can't extract → Ollama

        if self.language == "ur":
            speech = f"جی {name} صاحب ... آپکا موبائل نمبر بتا دیجیے"
        elif self.language == "ro":
            speech = f"Ji {name} sahab ... aapka mobile number bata dijiye"
        else:
            speech = f"Got it, {name}. Could I get your mobile number please?"

        logger.info(f"[Fast] Name extracted: {name!r}")
        return {"speech": speech, "action": "ask", "data": {"patient_name": name}}

    # Words that indicate the text is NOT a name
    _MEDICAL_WORDS = {
        "ڈاکٹر", "ڈاکڑ", "ڈاکھٹر", "ہسپتال", "اپوٹمنٹ", "اپائنٹمنٹ",
        "ملنا", "بنا", "بنو", "بنے", "چاہیے", "چاہ", "کوئی", "ہلو",
        "سلام", "علیکم", "مسئلہ", "تکلیف", "مریض", "doctor", "hospital",
        "appointment", "problem",
    }

    def _extract_name(self, text: str) -> Optional[str]:
        """Strip known filler phrases and return the likely name."""
        # Reject immediately if text contains exclamation marks or greetings —
        # these are sentences, not name answers.
        if "!" in text or text.count(" ") > 6:
            return None

        result = text.strip()
        fillers = _NAME_FILLERS.get(self.language, []) + _NAME_FILLERS["en"]
        for filler in sorted(fillers, key=len, reverse=True):
            result = re.sub(re.escape(filler), "", result, flags=re.IGNORECASE).strip()
        # Remove trailing/leading punctuation
        result = result.strip(".,،۔!؟?").strip()
        # Reject empty, too short, or contains phone digits
        if not result or len(result) < 2 or re.search(r'\d{4,}', result):
            return None
        # Reject if more than 4 words (it's a sentence, not a name)
        if len(result.split()) > 4:
            return None
        # Reject if any medical/action word is present
        for word in self._MEDICAL_WORDS:
            if word in result:
                return None
        # Reject common non-name Urdu words
        _NON_NAMES = {"ہاں", "جی", "ٹھیک", "بالکل", "اچھا", "yes", "no", "okay", "sure", "ہلو", "hello"}
        if result.lower() in _NON_NAMES or result in _NON_NAMES:
            return None
        # Title-case for English/Roman names
        if self.language == "en":
            result = result.title()
        return result or None

    # ── Phone handler ─────────────────────────────────────────────────────────

    def _handle_phone(self, text: str, collected: dict) -> Optional[dict]:
        phone = self._extract_phone(text)
        if not phone:
            return None  # can't extract → Ollama

        name = collected.get("patient_name", "")
        if self.language == "ur":
            speech = f"شکریہ ... آپکو کیا تکلیف ہے؟"
        elif self.language == "ro":
            speech = "Shukriya ... aapko kya takleef hai?"
        else:
            speech = "Thank you. What's the reason for your visit — any symptoms or concerns?"

        logger.info(f"[Fast] Phone extracted: {phone!r}")
        return {"speech": speech, "action": "ask", "data": {"patient_phone": phone}}

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract Pakistani mobile number. Accepts 10-11 digits (Whisper may drop a digit)."""
        # Remove spaces/dashes between digits first
        clean = re.sub(r'[\s\-]', '', text)

        m = _PHONE_RE.search(clean)
        if m:
            return f"0{m.group(1)}{m.group(2)}"

        m = _PHONE_11_RE.search(clean)
        if m:
            digits = m.group(1)
            if digits.startswith("03"):
                return digits

        # Fallback: extract any contiguous digit sequence of 10-11 digits
        digits_only = re.sub(r'\D', '', text)
        if 10 <= len(digits_only) <= 11:
            if not digits_only.startswith("0"):
                digits_only = "0" + digits_only
            # Accept both 10 and 11 digit Pakistani numbers starting with 03
            if digits_only.startswith("03"):
                return digits_only

        return None

    # ── Date handler ──────────────────────────────────────────────────────────

    def _handle_date(self, text: str, collected: dict) -> Optional[dict]:
        appt_date = self._extract_date(text)
        if not appt_date:
            return None  # can't parse → Ollama

        date_iso = appt_date.isoformat()
        doctor = collected.get("doctor_name", "")

        # Fetch available slots from DB for this doctor + date
        slots_text = ""
        if doctor:
            try:
                slots = self._booking.get_available_slots(doctor, appt_date)
                if slots:
                    top = slots[:4]
                    slots_text = "، ".join(top) if self.language == "ur" else " / ".join(top)
                else:
                    # No slots → let Ollama handle (needs to suggest alternatives)
                    logger.info(f"[Fast] No slots for {doctor} on {date_iso} — deferring to Ollama")
                    return None
            except Exception as e:
                logger.warning(f"[Fast] Slot lookup failed: {e}")
                return None

        # Format the date nicely
        date_label = self._format_date_label(appt_date)

        if self.language == "ur":
            if slots_text:
                speech = f"جی ... {date_label} کو ... {doctor} کے ساتھ یہ وقت دستیاب ہے ... {slots_text} ... کون سا وقت چاہیے؟"
            else:
                speech = f"جی ... {date_label} کو ... کون سا وقت چاہیے؟"
        elif self.language == "ro":
            if slots_text:
                speech = f"Ji ... {date_label} ko ... {doctor} ke saath yeh waqt available hai ... {slots_text} ... kaunsa waqt chahiye?"
            else:
                speech = f"Ji ... {date_label} ko ... kaunsa waqt chahiye?"
        else:
            if slots_text:
                speech = f"Sure, for {date_label} — {doctor} has these slots available: {slots_text}. Which time works for you?"
            else:
                speech = f"Sure, for {date_label} — what time would you like?"

        logger.info(f"[Fast] Date extracted: {date_iso}")
        return {"speech": speech, "action": "ask", "data": {"appointment_date": date_iso}}

    def _extract_date(self, text: str) -> Optional[date]:
        """Parse relative or absolute date references."""
        today = date.today()
        low = text.lower().strip()

        # 1. Urdu relative dates
        for word, offset in _URDU_REL_DATE.items():
            if word in text:
                return today + timedelta(days=offset)

        # 2. English/Roman relative dates
        for word, offset in _EN_REL_DATE.items():
            if re.search(rf'\b{re.escape(word)}\b', low):
                return today + timedelta(days=offset)

        # 3. Urdu day names → next occurrence
        for day_name, iso_day in _URDU_DAY.items():
            if day_name in text:
                return self._next_weekday(today, iso_day)

        # 4. English/Roman day names
        for day_name, iso_day in _EN_DAY.items():
            if re.search(rf'\b{re.escape(day_name)}\b', low):
                return self._next_weekday(today, iso_day)

        # 5. Numeric date patterns: DD/MM, DD-MM, "15 june", "june 15"
        MONTHS = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10,
            "november": 11, "december": 12,
        }
        # "15 june" or "june 15"
        for mon_name, mon_num in MONTHS.items():
            m = re.search(rf'(\d{{1,2}})\s+{mon_name}', low)
            if not m:
                m = re.search(rf'{mon_name}\s+(\d{{1,2}})', low)
            if m:
                day_num = int(m.group(1))
                year = today.year
                try:
                    d = date(year, mon_num, day_num)
                    if d < today:
                        d = date(year + 1, mon_num, day_num)
                    return d
                except ValueError:
                    pass

        # "15/6" or "15-6"
        m = re.search(r'(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?', low)
        if m:
            try:
                day_num, mon_num = int(m.group(1)), int(m.group(2))
                year = int(m.group(3)) if m.group(3) else today.year
                if year < 100:
                    year += 2000
                d = date(year, mon_num, day_num)
                if d < today:
                    d = date(year + 1, mon_num, day_num)
                return d
            except ValueError:
                pass

        return None

    def _next_weekday(self, from_date: date, iso_day: int) -> date:
        """Return the next occurrence of iso_day (1=Mon, 7=Sun) on or after from_date."""
        days_ahead = iso_day - from_date.isoweekday()
        if days_ahead <= 0:
            days_ahead += 7
        return from_date + timedelta(days=days_ahead)

    def _format_date_label(self, d: date) -> str:
        """Human-readable date label in the appropriate language."""
        today = date.today()
        delta = (d - today).days
        if self.language == "ur":
            if delta == 0:
                return "آج"
            if delta == 1:
                return "کل"
            if delta == 2:
                return "پرسوں"
            DAY_NAMES_UR = {1: "پیر", 2: "منگل", 3: "بدھ", 4: "جمعرات",
                             5: "جمعہ", 6: "ہفتہ", 7: "اتوار"}
            return DAY_NAMES_UR.get(d.isoweekday(), d.strftime("%d/%m"))
        elif self.language == "ro":
            if delta == 0: return "aaj"
            if delta == 1: return "kal"
            if delta == 2: return "parso"
            DAY_NAMES_RO = {1: "Somwar", 2: "Mangal", 3: "Budh", 4: "Jumerat",
                             5: "Jumma", 6: "Hafta", 7: "Itwar"}
            return DAY_NAMES_RO.get(d.isoweekday(), d.strftime("%d/%m"))
        else:
            if delta == 0: return "today"
            if delta == 1: return "tomorrow"
            return d.strftime("%A, %B %d")

    # ── Time handler ──────────────────────────────────────────────────────────

    def _handle_time(self, text: str, collected: dict) -> Optional[dict]:
        time_str = self._extract_time(text)
        if not time_str:
            return None  # can't parse → Ollama

        doctor = collected.get("doctor_name", "")
        appt_date_str = collected.get("appointment_date", "")

        # Validate the slot exists in DB
        if doctor and appt_date_str:
            try:
                appt_date = date.fromisoformat(appt_date_str)
                available = self._booking.get_available_slots(doctor, appt_date)
                if available and time_str not in available:
                    # Close match: find nearest slot
                    nearest = self._nearest_slot(time_str, available)
                    if nearest:
                        time_str = nearest
                    else:
                        # No match → let Ollama handle slot correction
                        return None
            except Exception as e:
                logger.warning(f"[Fast] Slot validation failed: {e}")

        name = collected.get("patient_name", "")
        doctor_disp = doctor or ""

        if self.language == "ur":
            speech = f"جی ... {time_str} بجے بکنگ ہو رہی ہے ... ایک لمحہ"
        elif self.language == "ro":
            speech = f"Ji ... {time_str} baje booking ho rahi hai ... ek lamha"
        else:
            speech = f"Perfect — booking you in for {time_str}. One moment..."

        logger.info(f"[Fast] Time extracted: {time_str!r}")
        return {"speech": speech, "action": "ask", "data": {"appointment_time": time_str}}

    def _extract_time(self, text: str) -> Optional[str]:
        """Parse a spoken time into HH:MM AM/PM format."""
        low = text.lower().strip()

        # Detect meridiem from Urdu time-of-day words
        meridiem = None
        for word, hint in _URDU_MERIDIEM.items():
            if word in text:
                meridiem = hint
                break
        if not meridiem:
            if re.search(r'\bam\b|morning|صبح', low):
                meridiem = "AM"
            elif re.search(r'\bpm\b|afternoon|evening|shaam|شام|دوپہر', low):
                meridiem = "PM"

        # Detect hour from Urdu number words
        hour = None
        for word, val in _URDU_NUMS.items():
            if word in text:
                hour = val
                break

        # Detect hour from digits: "10", "10:30", "10 baje"
        minute = 0
        if hour is None:
            # "10:30" pattern
            m = re.search(r'\b(\d{1,2}):(\d{2})\b', low)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2))
            else:
                # "10 baje" or just "10"
                m = re.search(r'\b(\d{1,2})\b', low)
                if m:
                    hour = int(m.group(1))

        if hour is None:
            return None

        # Resolve AM/PM
        if meridiem is None:
            # Default heuristic: 9-12 → AM, 1-8 → PM
            if 1 <= hour <= 8:
                meridiem = "PM"
            else:
                meridiem = "AM"

        if meridiem == "PM" and hour != 12:
            hour_24 = hour + 12
        elif meridiem == "AM" and hour == 12:
            hour_24 = 0
        else:
            hour_24 = hour

        return f"{hour_24:02d}:{minute:02d} {'AM' if hour_24 < 12 else 'PM'}"

    def _nearest_slot(self, requested: str, available: list) -> Optional[str]:
        """Find the closest available slot to what the user requested."""
        if not available:
            return None

        def parse_minutes(slot: str) -> int:
            m = re.search(r'(\d{1,2}):(\d{2})', slot)
            if m:
                return int(m.group(1)) * 60 + int(m.group(2))
            return 0

        req_mins = parse_minutes(requested)
        best = min(available, key=lambda s: abs(parse_minutes(s) - req_mins))
        diff = abs(parse_minutes(best) - req_mins)
        # Only snap if within 30 minutes
        return best if diff <= 30 else None
