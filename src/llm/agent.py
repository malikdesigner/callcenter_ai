"""
Hospital receptionist LLM agent powered by Ollama (local, no API key).
Manages multi-turn conversation and triggers appointment booking actions.
"""

import json
import re
from datetime import date, datetime
from typing import Optional

from loguru import logger
from sqlmodel import Session, select

from config.settings import settings
from src.appointment.booking import (
    BookingSystem,
    resolve_department,
)
from src.appointment.models import Appointment, Doctor, Department, engine
from src.llm.intent import detect_intent, detect_language_switch, detect_language_preference, detect_symptom_department

# ── Role & Strict Flow ────────────────────────────────────────────────────────

PLACEHOLDERS = {"unknown", "patient name", "phone number", "n/a", "none", "tbd", "placeholder"}
# reason is optional — patient may not give one explicitly
REQUIRED_FIELDS = ["patient_name", "patient_phone", "department", "doctor_name", "appointment_date", "appointment_time"]

# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(lang: str = "en") -> str:
    with Session(engine) as session:
        doctors = session.exec(select(Doctor).where(Doctor.is_active == True)).all()
        depts = session.exec(select(Department)).all()

    # Build doctor list with specialty so LLM can match symptoms to the right doctor
    doctor_info = []
    for dept in depts:
        dept_docs = [f"{d.name} ({d.specialty})" for d in doctors if d.department_name == dept.name]
        if dept_docs:
            doctor_info.append(f"{dept.name.title()}: " + ", ".join(dept_docs))
    doctor_list_str = " | ".join(doctor_info)

    today = date.today().isoformat()
    tomorrow = (date.today() + __import__('datetime').timedelta(days=1)).isoformat()
    current_time = datetime.now().strftime("%I:%M %p")

    if lang == "ur":
        return f"""You are Sara — receptionist at {settings.hospital_name}. Natural Pakistani phone manner. Talk like a human, not a form bot.

Today: {today} | Time: {current_time} | Tomorrow: {tomorrow}
Doctors: {doctor_list_str}

Booking flow: name → phone → problem → recommend doctor → date → time → confirm → book

━━━ STEP 1: IDENTIFY WHAT THE USER IS DOING ━━━
Every turn, first decide the user's intent:
• ANSWERING   — giving info you asked for → acknowledge + ask next thing
• CONFUSED    — doesn't understand (مطلب / سمجھ نہیں / کیا مراد / کیا ہے یہ) → explain briefly with examples
• QUESTIONING — asking about you/process/hospital (کیوں / کیا آپ / کیسے / مجھے نہیں پتا) → answer briefly + continue
• CORRECTING  — fixing earlier info (غلط / بدلو / نہیں / دوبارہ) → accept gracefully + re-collect
• GREETING    — hello/salam → greet back naturally

━━━ STEP 2: RESPOND BY INTENT ━━━

CONFUSED → explain with simple examples, then re-ask differently (never repeat same words):
  کیا تکلیف ہے مطلب   → یعنی کیا مسئلہ ہے ... بخار ... کھانسی ... درد یا کچھ اور
  سمجھ نہیں آئی        → جی سر ... کیا تکلیف ہے آپکو
  نام مطلب             → آپکا اپنا نام ... جیسے احمد یا علی

QUESTIONING → answer briefly then continue:
  کیا آپ انسان ہیں      → جی سر ... میں ہسپتال کی اسسٹنٹ ہوں
  مجھے نہیں پتا کونسا ڈاکٹر → کوئی مسئلہ نہیں ... مسئلہ بتا دیجیے میں بتا دیتی ہوں
  ڈاکٹر کیوں چاہیے     → جی سر ... جس مسئلے کیلئے چیک اپ کروانا ہو

CORRECTING → accept + re-collect:
  نمبر غلط ہوگیا       → کوئی بات نہیں ... دوبارہ بتا دیجیے

━━━ BOOKING RULES ━━━
• Symptoms → empathy + recommend SPECIFIC DOCTOR by name. NEVER "کونسے شعبے میں جانا ہے"
  ✓ اچھا جی ... ڈاکڑ [نام] ہیں اس کے لیے ... وقت لوں
• One thing per turn. Acknowledge before asking.
• NEVER repeat exact same question twice — rephrase.
• action="end" ONLY after booking done + patient says goodbye.

━━━ OUTPUT RULES ━━━
1. Urdu script only (ا ب پ...). Zero English or Roman Urdu.
2. Max 8-10 words. Very short — like a real call-center agent.
3. Pauses with ... only. NO ۔ . ! ? ، ,
4. Tone: جی سر ... اچھا جی ... ٹھیک ہے ... ایک سیکنڈ جی
5. Phonetic forms: آپکا | آپکو | آپکے | نئیں | ڈاکڑ | موبایل
6. NO formal: براہ کرم / معافی / تکلیف

BOOKING EXAMPLES:
سلام:    السلام علیکم ... میں سارہ ہوں ... آپکا نام کیا ہے
فون:     اچھا جی [نام] صاحب ... موبایل نمبر بتائیں
تکلیف:  اچھا جی ... ڈاکڑ [نام] ہیں اس کے لیے ... وقت لوں
تاریخ:  ٹھیک ہے ... کونسے دن آنا ہے
وقت:    یہ وقت ہیں ... [سلاٹس] ... کونسا ٹھیک ہے
تصدیق: [نام] صاحب ... ڈاکڑ [نام] ... [تاریخ] ... [وقت] پر ... کنفرم کروں
اختتام: جی ... شکریہ ... اللہ حافظ

JSON only: {{"speech":"...","action":"ask|confirm|save|end","data":{{"patient_name":"","patient_phone":"","department":"","doctor_name":"","appointment_date":"","appointment_time":"","reason":""}}}}"""

    if lang == "ro":
        return f"""You are Sara — receptionist at {settings.hospital_name}. Natural Pakistani phone manner in Roman Urdu (Latin script). Talk like a human, not a form bot.

Today: {today} | Time: {current_time} | Tomorrow: {tomorrow}
Doctors: {doctor_list_str}

Booking flow: naam → phone → problem → doctor recommend → date → waqt → confirm → book

━━━ STEP 1: USER KI NIYAT PAHCHANEIN ━━━
Har turn mein pehle decide karein:
• ANSWERING   — info de raha hai → acknowledge + agla sawal
• CONFUSED    — samajh nahi aaya (matlab / samajh nahi / kya hai yeh) → briefly explain + examples
• QUESTIONING — apke barey mein pooch raha hai (kya aap insaan / kyun / kaise) → jawab do + continue
• CORRECTING  — pehli baat theek kar raha hai (galat / badlo / nahi) → accept + re-collect
• GREETING    — hello/salam → naturally greet back

━━━ STEP 2: NIYAT KE MUTABIQ JAWAB ━━━

CONFUSED → simple examples ke saath samjhaein, phir alag andaz mein poochein (same words repeat mat karein):
  matlab kya hai        → yani kya masla hai ... bukhaar ... khansi ... dard ya kuch aur
  samajh nahi aaya      → ji sir ... kya takleef hai aapko
  naam matlab           → aapka apna naam ... jaise Ahmad ya Ali

QUESTIONING → briefly answer then continue:
  kya aap insaan hain   → ji sir ... main hospital ki assistant hoon
  nahi pata konsa doctor → koi masla nahi ... masla bata dein main bata deti hoon
  doctor kyun chahiye   → ji sir ... jis masle ke liye check-up karwana ho

CORRECTING → accept + re-collect:
  number galat hogaya   → koi baat nahi ... dobara bata dein

━━━ BOOKING RULES ━━━
• Symptoms → empathy + recommend SPECIFIC DOCTOR by name. NEVER "konse department mein jana hai"
  ✓ Acha ji ... Dr [naam] hain is ke liye ... waqt loon
• One thing per turn. Acknowledge before asking.
• NEVER repeat exact same question twice — rephrase.
• action="end" ONLY after booking done + patient says goodbye.

━━━ OUTPUT RULES ━━━
1. Roman Urdu only (Latin script). Zero Urdu script or English.
2. Max 8-10 words. Very short — like a real call-center agent.
3. Pauses with ... only. NO periods, commas, exclamation marks.
4. Tone: ji sir ... acha ji ... theek hai ... ek second ji
5. Common forms: aapka | aapko | nahi | doctor | mobile
6. NO formal: meherbani / maafi / izaazat

BOOKING EXAMPLES:
Salam:    Assalam o alaikum ... main Sara hoon ... aapka naam
Phone:    Acha ji [naam] sahab ... mobile number bataein
Takleef:  Acha ji ... Dr [naam] hain is ke liye ... waqt loon
Date:     Theek hai ... konse din aana hai
Waqt:     Yeh waqt hain ... [slots] ... konsa theek hai
Confirm:  [naam] sahab ... Dr [naam] ... [date] ... [waqt] par ... confirm karoon
Ikhtitaam: Ji ... shukriya ... Allah hafiz

JSON only: {{"speech":"...","action":"ask|confirm|save|end","data":{{"patient_name":"","patient_phone":"","department":"","doctor_name":"","appointment_date":"","appointment_time":"","reason":""}}}}"""

    return f"""You are {settings.receptionist_name}, receptionist at {settings.hospital_name}. Warm, human, conversational — never robotic or form-like.
Today: {today} | Time: {current_time} | Tomorrow: {tomorrow}
Doctors & specialties: {doctor_list_str}

Booking flow: name → phone → symptoms → recommend doctor → date → time → confirm → book

STEP 1 — IDENTIFY USER INTENT before responding:
• ANSWERING   — giving info you asked for → acknowledge + ask next
• CONFUSED    — doesn't understand ("what do you mean", "I don't get it") → explain simply with examples
• QUESTIONING — asking about you/process ("are you human", "why", "I don't know which doctor") → answer + continue
• CORRECTING  — fixing info ("wrong number", "actually", "no wait") → accept + re-collect
• GREETING    — hello → greet naturally

STEP 2 — RESPOND BY INTENT:
CONFUSED    → Explain with simple examples, re-ask in different words (never same sentence twice)
              "What do you mean by symptoms?" → "Things like headache, fever, cough — what's bothering you?"
QUESTIONING → Answer briefly, continue naturally
              "Are you a human?" → "I'm the hospital assistant — let me help you book."
              "I don't know which doctor" → "No problem — tell me what's bothering you, I'll recommend the right one."
CORRECTING  → "Of course, go ahead." → accept correction, re-collect field

BOOKING RULES:
1. Symptoms → empathy first, then recommend SPECIFIC DOCTOR by name. NEVER "which department?"
   ✓ "For headaches, Dr. Kamran Baig is our neurologist — shall I book with him?"
   ✗ "Which department would you like?"
2. One question per turn. Acknowledge before asking.
3. NEVER repeat the same question twice — rephrase.
4. action="end" ONLY after booking complete + patient says goodbye.

action: ask=collecting info | confirm=all ready, read back | save=patient confirmed | end=after booking+goodbye
JSON only: {{"speech":"...","action":"ask|confirm|save|end","data":{{"patient_name":"","patient_phone":"","department":"","doctor_name":"","appointment_date":"","appointment_time":"","reason":""}}}}"""

# ── Agent class ────────────────────────────────────────────────────────────────

class HospitalAgent:
    def __init__(self, language: str = "en"):
        # "bi" mode = start by asking user which language they prefer
        self._bilingual_mode: bool = (language == "bi")
        self.language: str = "en" if self._bilingual_mode else language
        self._language_chosen: bool = not self._bilingual_mode

        self.booking = BookingSystem()
        self._history: list = []
        self._collected: dict = {}
        # State machine
        self._flow_state: str = "greeting"
        self._awaiting_confirmation: bool = False
        # Robustness tracking
        self._consecutive_failures: int = 0
        self._last_asked_state: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Reset state for a new call."""
        self._history = []
        self._collected = {}
        self._flow_state = "greeting"
        self._awaiting_confirmation = False
        self._consecutive_failures = 0
        self._last_asked_state = ""
        if self._bilingual_mode:
            self.language = "en"
            self._language_chosen = False

    async def get_greeting(self) -> str:
        if self._bilingual_mode and not self._language_chosen:
            return (
                f"To continue in English, say English. "
                f"اردو میں جاری رکھنے کے لیے اردو کہیں۔"
            )

        prompt = "A new caller just connected. Greet them naturally like a real receptionist — brief and warm."
        if self.language == "ur":
            prompt = "New call arrived. Greet in short conversational Urdu script, Pakistani call-center style. Max 6-8 words, use ... for pauses, no other punctuation. Example: السلام علیکم ... میں سارہ ہوں ... آپکا نام کیا ہے"
        elif self.language == "ro":
            prompt = "New call arrived. Greet in short conversational Roman Urdu (Latin script), Pakistani call-center style. Max 6-8 words, use ... for pauses, no other punctuation. Example: Assalam o alaikum ... main Sara hoon ... aapka naam"
        result = await self._chat(prompt)

        fallback_en = f"Thank you for calling {settings.hospital_name}, this is {settings.receptionist_name}. How can I help you?"
        fallback_ur = f"السلام علیکم ... میں سارہ ہوں {settings.hospital_name} سے ... آپکا نام کیا ہے"
        fallback_ro = f"Assalam o alaikum ... main Sara hoon {settings.hospital_name} se ... aapka naam"

        if self.language == "ur":
            return result.get("speech", fallback_ur)
        if self.language == "ro":
            return result.get("speech", fallback_ro)
        return result.get("speech", fallback_en)

    async def process_turn(self, user_text: str) -> dict:
        """
        Process one user turn.
        Returns dict with 'speech', 'action', 'data'.
        May execute a booking/cancellation as a side effect.
        """
        result = await self._chat(user_text)

        # Merge collected data
        if result.get("data"):
            for k, v in result["data"].items():
                if v:
                    self._collected[k] = v

        action = result.get("action", "none")

        if action == "book_appointment":
            result = self._execute_booking(result)
        elif action == "cancel_appointment":
            result = self._execute_cancellation(result)

        return result

    async def process_turn_stream(self, user_text: str, audio_language: str = None):
        """
        Yields strings (sentences) as they are generated by the LLM.
        The very last item yielded will be the complete response dict.
        """
        full_json_str = ""
        last_yielded_speech_idx = 0
        speech_buffer = ""

        # ── Bilingual mode: first turn is language selection ──────────────
        if self._bilingual_mode and not self._language_chosen:
            pref = detect_language_preference(user_text)

            # Whisper's audio-level language detection is the ground truth when
            # text-based detection fails (e.g. hallucinated Latin words from Urdu speech)
            if pref is None and audio_language == "ur":
                # Whisper's audio-level detection is reliable for Urdu — use it
                # even when the transcript is a phonetic mis-transcription like "or do"
                pref = "ur"
                logger.info(f"[Agent] Bilingual: Whisper detected Urdu audio (transcript='{user_text}')")

            if pref == "ur":
                self.language = "ur"   # so TTS uses Urdu voice for the farewell line
                speech = "بالکل جی ... ابھی اردو سروس سے جوڑتی ہوں"
                target_port = 8001
                logger.info("[Agent] Bilingual: Urdu detected → redirecting to port 8001")
            elif pref == "en":
                self.language = "en"
                speech = "Perfect! Connecting you to English service now."
                target_port = 8000
                logger.info("[Agent] Bilingual: English detected → redirecting to port 8000")
            else:
                speech = "سمجھی نئیں ... اردو یا English کہیں / Please say Urdu or English"
                logger.info(f"[Agent] Bilingual: unclear (transcript='{user_text}', audio_lang={audio_language})")
                yield speech
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": speech})
                yield {"speech": speech, "action": "ask", "data": {}}
                return

            # Language confirmed — play farewell then redirect browser to the right server
            yield speech
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": speech})
            yield {"speech": speech, "action": "redirect", "data": {"port": target_port}}
            return

        # ── Symptom → department pre-fill (no LLM needed) ────────────────
        if "department" not in self._collected or not self._collected["department"]:
            dept = detect_symptom_department(user_text)
            if dept:
                self._collected["department"] = dept
                logger.info(f"[Agent] Symptom keyword → department='{dept}'")

        # ── Intent detection ──────────────────────────────────────────────
        intent = detect_intent(user_text, self._flow_state, self.language)
        logger.debug(f"[Agent] Intent={intent} | State={self._flow_state} | Awaiting={self._awaiting_confirmation}")

        # ── Language switching — explicit commands only ───────────────────
        # We rely on detect_intent (not detect_language_switch) for this because
        # detect_language_switch's script heuristic falsely returns "en" for
        # phone numbers and names that contain no Arabic characters.
        if intent == "change_language_en" and self.language != "en":
            self.language = "en"
            logger.info("[Agent] Language switched to 'en' (explicit command)")
        elif intent == "change_language_ur" and self.language != "ur":
            self.language = "ur"
            logger.info("[Agent] Language switched to 'ur' (explicit command)")

        # ── Failure tracking ──────────────────────────────────────────────
        # Confusion/questions are NOT failures — the user is engaging, just
        # not answering the slot question yet.
        user_intent_type = self._classify_user_intent(user_text)
        is_non_answer = user_intent_type in ("CONFUSED", "QUESTIONING", "CORRECTING")
        if not is_non_answer and (intent == "unclear" or len(user_text.strip()) < 2):
            self._consecutive_failures += 1
        elif not is_non_answer:
            self._consecutive_failures = 0

        # ── Confirmation gate: if we are waiting for user's yes/no ────────
        if self._awaiting_confirmation:
            if intent == "confirm":
                # User said yes → book without calling LLM again
                logger.info("[Agent] Confirmation received — executing booking directly")
                result = self._execute_booking({"speech": "", "action": "save", "data": {}})
                self._awaiting_confirmation = False
                self._flow_state = "post_booking"
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": result.get("speech", "")})
                yield result
                return
            elif intent == "deny":
                # User said no → clear last confirmed data and re-ask
                logger.info("[Agent] Denial received — resetting confirmation state")
                self._awaiting_confirmation = False
                self._collected.pop("appointment_time", None)
                self._collected.pop("appointment_date", None)
                # Fall through so LLM generates a helpful "What would you like to change?" response

        # ── Correction detection ──────────────────────────────────────────
        # If the user is correcting a previously collected field, clear it so
        # the LLM is forced to re-collect rather than silently ignoring the fix.
        _CORRECTION_WORDS = [
            # Urdu
            "نہیں", "نای", "نا", "غلط", "نہیں،", "نہیں۔",
            "بدلو", "بدلیں", "درست", "صحیح نہیں",
            # English
            "no", "not", "wrong", "incorrect", "change", "actually",
        ]
        user_lower = user_text.lower()
        is_correction = any(w in user_lower or w in user_text for w in _CORRECTION_WORDS)

        if is_correction:
            # Detect which field is being corrected from the user's message
            _TIME_WORDS = ["بجے", "وقت", "time", "am", "pm", "بج", "صبح", "شام", "دوپہر"]
            _DATE_WORDS = ["تاریخ", "date", "کل", "پرسوں", "آج", "tomorrow"]
            _DOCTOR_WORDS = ["ڈاکٹر", "doctor", "dr"]
            if any(w in user_text.lower() or w in user_text for w in _TIME_WORDS):
                self._collected.pop("appointment_time", None)
                logger.debug("[Agent] Correction detected — cleared appointment_time")
            if any(w in user_text.lower() for w in _DATE_WORDS):
                self._collected.pop("appointment_date", None)
                logger.debug("[Agent] Correction detected — cleared appointment_date")
            if any(w in user_text.lower() for w in _DOCTOR_WORDS):
                self._collected.pop("doctor_name", None)
                logger.debug("[Agent] Correction detected — cleared doctor_name")

        async for content in self._chat_stream(user_text):
            full_json_str += content

            match = re.search(r'"speech"\s*:\s*"(.*?)"', full_json_str, re.DOTALL)
            if match:
                current_speech = match.group(1)
                new_symbols = current_speech[last_yielded_speech_idx:]
                
                if new_symbols:
                    # 1. Unescape JSON literals (like \u0627 or \n)
                    try:
                        # Wrap in quotes to make it a valid JSON string literal for decoding
                        clean_symbols = json.loads(f'"{new_symbols}"')
                    except:
                        # Fallback if the partial chunk ends in a backslash
                        clean_symbols = new_symbols.replace("\\n", "\n").replace("\\t", " ")
                    
                    # 2. Strip HTML tags (like <u>) that LLM might hallucinate
                    clean_symbols = re.sub(r'<.*?>', '', clean_symbols)
                    
                    speech_buffer += clean_symbols
                    last_yielded_speech_idx += len(new_symbols)
                    
                    # Look for sentence boundaries in our buffer
                    # Includes Urdu full stop ۔ (U+06D4) and Urdu ? ؟ (U+061F)
                    sentences = re.split(r'(?<=[.!?۔؟])\s+', speech_buffer)
                    if len(sentences) > 1:
                        # Yield all complete sentences
                        for s in sentences[:-1]:
                            yield s
                        # Keep the last (potentially incomplete) one
                        speech_buffer = sentences[-1]

        # Final cleanup for the last sentence
        if speech_buffer.strip():
            yield speech_buffer.strip()

        # Parse final JSON to get result and update state
        logger.debug(f"[LLM] Raw Response: {full_json_str}")
        try:
            result = json.loads(full_json_str)
        except:
            match = re.search(r'"speech"\s*:\s*"(.*?)"', full_json_str, re.DOTALL)
            speech = match.group(1) if match else "I'm sorry, I'm having trouble thinking."
            result = {"speech": speech, "action": "ask", "data": {}}

        # ── State Security ──
        # Filter and Merge collected data
        if result.get("data"):
            for k, v in result["data"].items():
                val = str(v).strip() if v else ""
                # Ignore placeholders and empty strings
                if val and val.lower() not in PLACEHOLDERS:
                    self._collected[k] = val
        
        self._normalise_collected()
        
        # Determine logical next action
        missing = self._get_missing_requirements()
        action = result.get("action", "ask")

        # GATEKEEPER: Prevent save/confirm if critical info missing or invalid
        if action in ["save", "confirm"] and missing:
            logger.warning(f"[Agent] Blocking {action} due to missing/invalid fields: {missing}")
            # Phone-specific error messages take priority
            if "_phone_too_short" in missing:
                phone_val = self._collected.get("patient_phone", "")
                self._collected.pop("patient_phone", None)  # force re-collect
                if self.language == "ur":
                    result["speech"] = f"نمبر مکمل نئیں ... پورا موبایل نمبر بتائیں"
                elif self.language == "ro":
                    result["speech"] = f"Number mukammal nahi ... pura mobile number bataein"
                else:
                    result["speech"] = f"That number doesn't look right — it's too short. Please give me your full phone number (at least 10 digits)."
            elif "_phone_too_long" in missing:
                phone_val = self._collected.get("patient_phone", "")
                self._collected.pop("patient_phone", None)
                if self.language == "ur":
                    result["speech"] = f"نمبر لمبا لگتا ہے ... صحیح نمبر بتائیں"
                elif self.language == "ro":
                    result["speech"] = f"Number lamba lagta hai ... sahi number bataein"
                else:
                    result["speech"] = f"That number seems too long. Could you double-check and give me your correct phone number?"
            else:
                real_missing = [f for f in missing if not f.startswith("_")]
                labels = [f.replace('patient_', '').replace('_', ' ') for f in real_missing]
                if self.language == "ur":
                    result["speech"] = f"ابھی {' اور '.join(labels)} چاہیے ... بتا سکتے ہیں"
                elif self.language == "ro":
                    result["speech"] = f"Abhi {' aur '.join(labels)} chahiye ... bata sakte hain"
                else:
                    result["speech"] = f"I still need {', and '.join(labels)} before I can book. Could you provide that?"
            action = "ask"
            result["action"] = "ask"

        if action == "save":
            logger.info(f"[Agent] Executing booking — collected: {self._collected}")
            result = self._execute_booking(result)
            self._awaiting_confirmation = False
            self._flow_state = "post_booking"
        elif action == "confirm" and not missing:
            if self._flow_state == "post_booking":
                # Booking is already done — LLM is spuriously re-reading details.
                # Treat it as a normal post-booking reply instead.
                logger.warning("[Agent] Blocking re-confirmation in post_booking state → converting to ask")
                result["action"] = "ask"
            else:
                logger.info("[Agent] All info ready — waiting for user confirmation")
                self._awaiting_confirmation = True
                self._flow_state = "confirm_booking"
        elif action == "end":
            # Only allow ending the call after booking is complete
            if self._flow_state == "post_booking":
                result["action"] = "end"
                self._flow_state = "done"
            else:
                # LLM tried to end mid-conversation — block it and keep collecting
                logger.warning("[Agent] Blocking premature 'end' — booking not complete, converting to 'ask'")
                result["action"] = "ask"
                self._flow_state = self._compute_flow_state()
        else:
            # action == "ask" or similar — update flow state based on what's still missing
            self._flow_state = self._compute_flow_state()

        # Update history — keep last 8 turns (16 messages) to save tokens
        self._history.append({"role": "user", "content": user_text})
        # Only store speech in history, not full JSON, to reduce token count
        self._history.append({"role": "assistant", "content": result.get("speech", "")})
        if len(self._history) > 16:
            self._history = self._history[-16:]

        yield result

    # ── Normalisation ──────────────────────────────────────────────────────────

    def _normalise_collected(self):
        """
        Sanitise self._collected after each LLM turn.
        Handles time formats (including Urdu number words), department aliases, and date shortcuts.
        """
        # ── Time normalisation ────────────────────────────────────────────────
        _TIME_MAP = {
            (9, 0): "09:00 AM",  (9, 30): "09:30 AM",
            (10, 0): "10:00 AM", (10, 30): "10:30 AM",
            (11, 0): "11:00 AM", (11, 30): "11:30 AM",
            (12, 0): "12:00 PM", (12, 30): "12:30 PM",
            (13, 0): "01:00 PM", (13, 30): "01:30 PM",
            (14, 0): "02:00 PM", (14, 30): "02:30 PM",
            (15, 0): "03:00 PM", (15, 30): "03:30 PM",
            (16, 0): "04:00 PM", (16, 30): "04:30 PM",
            (17, 0): "05:00 PM", (17, 30): "05:30 PM",
        }
        _VALID_SLOTS = set(_TIME_MAP.values())

        # Urdu/Hindi number words → integer hour
        _URDU_NUMBERS = {
            "ایک": 1, "دو": 2, "تین": 3, "چار": 4, "پانچ": 5,
            "چھ": 6, "سات": 7, "آٹھ": 8, "نو": 9, "دس": 10,
            "گیارہ": 11, "بارہ": 12,
        }
        # Urdu time-of-day words → meridiem hint
        _URDU_MERIDIEM = {
            "صبح": "am", "دوپہر": "pm", "شام": "pm", "رات": "pm",
        }

        raw_time = self._collected.get("appointment_time", "")
        if raw_time and raw_time not in _VALID_SLOTS:
            t = raw_time.strip()
            hour, minute, meridiem = None, 0, None

            # ── Urdu number words ─────────────────────────────────────────
            # Detect meridiem hint first ("صبح دس" → am 10, "شام چار" → pm 4)
            for word, hint in _URDU_MERIDIEM.items():
                if word in t:
                    meridiem = hint
                    break

            for word, val in _URDU_NUMBERS.items():
                if word in t:
                    hour = val
                    break

            # ── English/numeric formats ───────────────────────────────────
            t_lower = t.lower()
            if hour is None:
                half = re.search(r'half\s+past\s+(\d+)|(\d+)\s+thirty', t_lower)
                if half:
                    hour = int(half.group(1) or half.group(2))
                    minute = 30

            if hour is None:
                m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', t_lower)
                if m:
                    hour = int(m.group(1))
                    minute = int(m.group(2) or 0)
                    if m.group(3):
                        meridiem = m.group(3)

            if hour is not None:
                # Apply meridiem
                if meridiem == "pm" and hour < 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                elif meridiem is None and hour <= 6:
                    hour += 12  # assume afternoon for ambiguous low hours

                # Snap minute to nearest 0 or 30
                minute = 30 if minute >= 15 else 0
                canonical = _TIME_MAP.get((hour, minute))
                if canonical:
                    self._collected["appointment_time"] = canonical
                    logger.debug(f"[Agent] Time normalised: '{raw_time}' → '{canonical}'")
                else:
                    logger.warning(f"[Agent] Time '{raw_time}' (hour={hour}) could not map to a slot")

        # ── Department normalisation ─────────────────────────────────────────
        from src.appointment.booking import resolve_department
        raw_dept = self._collected.get("department", "")
        if raw_dept:
            resolved = resolve_department(raw_dept)
            if resolved:
                self._collected["department"] = resolved

        # ── Phone normalisation ──────────────────────────────────────────────
        # Strip Urdu comma separators (،), spaces, dashes — keep digits only
        raw_phone = self._collected.get("patient_phone", "")
        if raw_phone:
            clean_phone = re.sub(r'[^\d]', '', raw_phone)
            if clean_phone != raw_phone and len(clean_phone) >= 7:
                self._collected["patient_phone"] = clean_phone
                logger.debug(f"[Agent] Phone normalised: '{raw_phone}' → '{clean_phone}'")

        # ── Date normalisation ───────────────────────────────────────────────
        raw_date = self._collected.get("appointment_date", "")
        if raw_date:
            from datetime import date, timedelta
            today = date.today()
            d = raw_date.lower().strip()
            # English shortcuts
            if d in ("today", "آج"):
                self._collected["appointment_date"] = today.isoformat()
            elif d in ("tomorrow", "next day", "کل"):
                self._collected["appointment_date"] = (today + timedelta(days=1)).isoformat()
            elif d in ("day after tomorrow", "day after", "پرسوں", "پرسو"):
                self._collected["appointment_date"] = (today + timedelta(days=2)).isoformat()
            # Already YYYY-MM-DD — leave as-is

    # ── Internal helpers ───────────────────────────────────────────────────────

    def get_transcript(self) -> str:
        """Converts self._history into a clean text transcript."""
        lines = []
        for msg in self._history:
            if self.language == "ur":
                role = "مریض" if msg["role"] == "user" else "سارہ"
            elif self.language == "ro":
                role = "Mareez" if msg["role"] == "user" else "Sara"
            else:
                role = "Patient" if msg["role"] == "user" else "Sara AI"
            
            content = msg["content"]
            
            # Assistant content is JSON — extract the speech
            if msg["role"] == "assistant":
                try:
                    data = json.loads(content)
                    content = data.get("speech", content)
                except:
                    pass
            
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _get_missing_requirements(self) -> list:
        """Helper to find which required fields are still missing, placeholders, or invalid."""
        missing = []
        for field in REQUIRED_FIELDS:
            val = self._collected.get(field, "")
            if not val or val.lower() in PLACEHOLDERS:
                missing.append(field)
                continue

            # Phone: Pakistani numbers are 10–13 digits (local/country-code formats)
            if field == "patient_phone":
                clean = "".join(filter(str.isdigit, val))
                if len(clean) < 10:
                    missing.append("_phone_too_short")
                elif len(clean) > 13:
                    missing.append("_phone_too_long")
        return missing

    def _compute_flow_state(self) -> str:
        """Helper to determine the current state based on what's missing."""
        missing = self._get_missing_requirements()
        if not missing:
            return "confirm_booking"

        # Determine logical next step based on what's missing
        if "patient_name" in missing:
            return "collect_name"
        if "patient_phone" in missing or "_phone_too_short" in missing or "_phone_too_long" in missing:
            return "collect_phone"
        if "department" in missing and "doctor_name" in missing:
            return "ask_information" # wait, does "collect_department" exist in intent.py? intent.py supports provide_name, provide_phone, describe_symptoms, choose_date, choose_time. So flow_state can be 'collect_name' to help intent.py.
        if "doctor_name" in missing:
            return "ask_information"
        if "appointment_date" in missing:
            return "collect_date"
        if "appointment_time" in missing:
            return "collect_time"
        return "ask_information"

    @staticmethod
    def _classify_user_intent(text: str) -> str:
        """
        Fast heuristic classification of the user's turn.
        Returns: CONFUSED | QUESTIONING | CORRECTING | ANSWERING
        Used to signal the LLM that slot-filling should pause this turn.
        """
        t = text.strip()
        t_lower = t.lower()

        _CONFUSED_UR = ["مطلب", "سمجھ نہیں", "کیا مراد", "سمجھائیں", "نہیں سمجھا", "نہیں سمجھی", "کیا ہے یہ", "یہ کیا"]
        _CONFUSED_RO = ["matlab", "samajh nahi", "kya muraad", "samjhaein", "nahi samjha", "nahi samjhi", "kya hai yeh", "kya matlab"]
        _CONFUSED_EN = ["what do you mean", "what does that mean", "don't understand", "i don't get", "what is that", "confused", "explain"]
        if any(w in t for w in _CONFUSED_UR) or any(w in t_lower for w in _CONFUSED_RO) or any(w in t_lower for w in _CONFUSED_EN):
            return "CONFUSED"

        _QUESTION_UR = ["کیا آپ انسان", "آپ کون", "مجھے نہیں پتا", "نہیں پتا کونسا", "ڈاکٹر کیوں", "کیوں چاہیے", "کیسے", "کب تک", "کتنا وقت"]
        _QUESTION_RO = ["kya aap insaan", "aap kaun", "mujhe nahi pata", "nahi pata konsa", "doctor kyun", "kyun chahiye", "kaise", "kab tak", "kitna waqt"]
        _QUESTION_EN = ["are you human", "who are you", "why do you", "i don't know which", "i'm not sure which", "how long", "what happens"]
        if any(w in t for w in _QUESTION_UR) or any(w in t_lower for w in _QUESTION_RO) or any(w in t_lower for w in _QUESTION_EN):
            return "QUESTIONING"

        _CORRECT_UR = ["غلط", "بدلو", "بدلیں", "دوبارہ بتاتا", "دوبارہ بتاتی", "غلط ہوگیا", "غلط دیا", "صحیح نئیں"]
        _CORRECT_RO = ["galat", "badlo", "badlein", "dobara bata", "galat hogaya", "galat diya", "theek nahi"]
        _CORRECT_EN = ["wrong number", "that's wrong", "i said", "actually it's", "no wait", "correction", "i meant"]
        if any(w in t for w in _CORRECT_UR) or any(w in t_lower for w in _CORRECT_RO) or any(w in t_lower for w in _CORRECT_EN):
            return "CORRECTING"

        return "ANSWERING"

    def _build_context_injection(self, user_text: str = "") -> str:
        """
        Inject live DB availability and collected state into the system prompt.
        - If doctor + date are both known → show exact free slots for that combo.
        - If only department is known → show next available across all doctors.
        - If requested slot is already booked → warn the LLM explicitly.
        """
        from datetime import date as _date
        lines = []

        # ── Classify user intent so LLM knows how to respond this turn ───────
        user_intent = self._classify_user_intent(user_text) if user_text else "ANSWERING"
        non_answering = user_intent in ("CONFUSED", "QUESTIONING", "CORRECTING")
        if non_answering:
            lines.append(
                f"USER_INTENT: {user_intent} — "
                + {
                    "CONFUSED":    "The user does not understand. Explain first, then gently re-ask.",
                    "QUESTIONING": "The user is asking a question. Answer it briefly, then continue naturally.",
                    "CORRECTING":  "The user is correcting earlier info. Acknowledge it and re-collect that field.",
                }[user_intent]
            )

        # Post-booking state: appointment is confirmed, handle follow-up questions
        if self._flow_state == "post_booking":
            appt_id = self._collected.get("appointment_id", "")
            lines.append(
                f"BOOKING_COMPLETE: Appointment #{appt_id} is confirmed. "
                f"Answer follow-up questions naturally. "
                f"When patient says goodbye/thanks/done → action='end'."
            )
            return "\n\n".join(lines)

        if self._collected:
            lines.append(f"CURRENT_STATE: {json.dumps(self._collected)}")

        missing = self._get_missing_requirements()
        if missing:
            lines.append(f"STILL_NEEDED: {', '.join(missing)}")
            if non_answering:
                # User is confused/questioning — don't force slot-filling this turn
                lines.append(
                    "NOTE: Handle the user's intent above first. "
                    "Only ask for missing info after you have addressed their confusion/question."
                )
            else:
                if self.language in ("ur", "ro"):
                    lines.append("RULE: Pehle achi tarah jawab dein, phir alag andaz mein missing info maangein. Ek hi phrase dobara mat bolein.")
                else:
                    lines.append("RULE: Acknowledge what they said, then ask for the missing info in a natural, non-repetitive way.")
        else:
            lines.append("ALL_INFO_COLLECTED: All info ready — proceed to confirmation.")

        # ── Repetition guard ──────────────────────────────────────────────────
        if self._last_asked_state and self._last_asked_state == self._flow_state:
            if self.language in ("ur", "ro"):
                lines.append("REPHRASE: Yeh sawal pehle poocha ja chuka hai. Bilkul alag aur seedha andaz mein poochein.")
            else:
                lines.append("REPHRASE: You already asked this exact thing. Use completely different words.")
        self._last_asked_state = self._flow_state

        # ── Failure counter hint ──────────────────────────────────────────────
        if self._consecutive_failures >= 2:
            if self.language == "ur":
                lines.append(
                    f"USER_STRUGGLING ({self._consecutive_failures} baar): "
                    "Bahut chota aur saadha poochein. "
                    "Agar phir bhi nahi samjhe: 'بخار ... درد ... یا کچھ اور'"
                )
            elif self.language == "ro":
                lines.append(
                    f"USER_STRUGGLING ({self._consecutive_failures} baar): "
                    "Bahut chota aur saadha poochein. "
                    "Agar phir bhi nahi samjhe: 'bukhaar ... dard ... ya kuch aur'"
                )
            else:
                lines.append(
                    f"USER_STRUGGLING ({self._consecutive_failures} turns): Simplify drastically. "
                    "Try multiple choice: 'Is it fever, pain, or something else?'"
                )

        doctor = self._collected.get("doctor_name", "")
        dept = self._collected.get("department", "")
        raw_date = self._collected.get("appointment_date", "")
        requested_time = self._collected.get("appointment_time", "")

        # ── Case 1: Doctor + Date known → show exact slots from DB ───────────
        if doctor and raw_date:
            try:
                appt_date = _date.fromisoformat(raw_date)
                available_slots = self.booking.get_available_slots(doctor, appt_date)
                if available_slots:
                    slots_str = ", ".join(available_slots)
                    lines.append(
                        f"AVAILABLE_SLOTS for {doctor} on {raw_date}: {slots_str}\n"
                        f"STRICT RULE: You MUST only offer slots from this list. "
                        f"Do NOT invent or suggest any other time."
                    )
                    # Warn if the user's requested time is not in the available list
                    if requested_time and requested_time not in available_slots:
                        lines.append(
                            f"WARNING: Requested time '{requested_time}' is NOT available for {doctor} on {raw_date}. "
                            f"Tell the user this slot is taken and offer alternatives from the list above. "
                            f"Do NOT book this slot. Set action='ask'."
                        )
                else:
                    lines.append(
                        f"NO_SLOTS_AVAILABLE: {doctor} has no free slots on {raw_date}. "
                        f"Tell the user this and suggest checking another date or doctor. "
                        f"Do NOT set action='save' or 'confirm'."
                    )
            except (ValueError, Exception) as e:
                logger.warning(f"[Agent] Context slot check failed: {e}")

        # ── Case 2: Only department known → show next available ───────────────
        elif dept:
            # Inject RECOMMEND_DOCTOR when doctor not yet chosen
            if not doctor:
                try:
                    with Session(engine) as session:
                        dept_docs = session.exec(
                            select(Doctor).where(
                                Doctor.department_name == dept,
                                Doctor.is_active == True,
                            )
                        ).all()
                    if dept_docs:
                        doc_names = ", ".join(f"{d.name} ({d.specialty})" for d in dept_docs)
                        if self.language in ("ur", "ro"):
                            lines.append(
                                f"RECOMMEND_DOCTOR: {dept} department mein yeh doctors hain: {doc_names}. "
                                f"Patient ki takleef ke mutabiq ek doctor ka naam khud suggest karein. "
                                f"'Kaun sa department chahiye?' mat poochein."
                            )
                        else:
                            lines.append(
                                f"RECOMMEND_DOCTOR: Doctors in {dept}: {doc_names}. "
                                f"Pick the best match for the patient's symptoms and recommend them by name. "
                                f"Do NOT ask 'which department do you want?'"
                            )
                except Exception as e:
                    logger.warning(f"[Agent] RECOMMEND_DOCTOR lookup failed: {e}")

            availability = self.booking.get_next_available(dept, days_ahead=5)
            if availability:
                av_lines = []
                for item in availability[:4]:
                    av_lines.append(
                        f"  {item['doctor']} | {item['date']} ({item['date_iso']}) | "
                        + ", ".join(item["slots"][:4])
                    )
                lines.append(
                    "AVAILABLE_SLOTS (next 5 days):\n" + "\n".join(av_lines) + "\n"
                    "STRICT RULE: Only offer these specific slots. Use date_iso format for appointment_date."
                )
            else:
                lines.append(
                    f"NO_AVAILABILITY: No slots available in {dept} for the next 5 days. "
                    f"Inform the user and suggest calling back later."
                )

        return "\n\n".join(lines)

    async def _chat(self, user_input: str) -> dict:
        """
        Single-turn async chat. Accumulates the stream and parses JSON.
        Returns a dict with at least {"speech": str, "action": str, "data": dict}.
        """
        full_raw = ""
        async for delta in self._chat_stream(user_input):
            full_raw += delta

        try:
            result = json.loads(full_raw)
            # Some models wrap the response one level deeper
            if "speech" not in result:
                for v in result.values():
                    if isinstance(v, dict) and "speech" in v:
                        result = v
                        break
            if not isinstance(result.get("speech"), str):
                raise ValueError("speech not a string")
        except Exception:
            m = re.search(r'"speech"\s*:\s*"(.*?)"', full_raw, re.DOTALL)
            if m:
                speech = m.group(1)
            elif self.language == "ur":
                speech = f"السلام علیکم ... {settings.hospital_name} سے سارہ ہوں ... کیا مدد کروں"
            elif self.language == "ro":
                speech = f"Assalam o alaikum ... {settings.hospital_name} se Sara hoon ... kya madad karoon"
            else:
                speech = f"Thank you for calling {settings.hospital_name}. How can I help?"
            result = {"speech": speech, "action": "none", "data": {}}

        return result

    async def _chat_stream(self, user_input: str):
        """
        Yields raw content delta strings (str) from the LLM.
        Priority: Gemini → Groq → Ollama (local) → HuggingFace.
        Add GEMINI_API_KEY or GROQ_API_KEY to .env to enable cloud providers.
        """
        system = _build_system_prompt(self.language)
        ctx = self._build_context_injection(user_input)
        if ctx:
            system += f"\n\n--- LIVE CONTEXT ---\n{ctx}"

        messages = [{"role": "system", "content": system}]
        messages += self._history
        messages.append({"role": "user", "content": user_input})

        from openai import AsyncOpenAI

        # ── Build provider list in priority order ──────────────────────────
        # Each key in .env enables the matching provider.
        # Priority: Gemini → HuggingFace/hf-inference → Ollama (local).
        providers = []

        # HuggingFace free inference is disabled — their endpoint format and model
        # availability changes too frequently to be reliable. Use Ollama instead.

        # ── Try cloud providers first ──────────────────────────────────────
        for client, model_name, provider_name, use_json_format in providers:
            logger.info(f"[LLM] Trying {provider_name} → {model_name}")
            try:
                stream = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={"type": "json_object"} if use_json_format else None,
                    temperature=0.2,
                    max_tokens=350,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  # success
            except Exception as e:
                err_str = str(e)
                # Skip provider permanently this session if quota is exhausted (limit: 0)
                if "limit: 0" in err_str or "tokens per day" in err_str.lower():
                    logger.warning(f"[LLM] {provider_name} quota exhausted — skipping for session.")
                else:
                    logger.error(f"[LLM] {provider_name} failed: {e}. Trying next.")

        # ── 4. Ollama (local) — auto-detect installed models ──────────────
        from ollama import AsyncClient
        ollama_options = {"temperature": 0.2, "num_predict": 350, "num_ctx": 2048}
        if settings.ollama_num_gpu >= 0:
            ollama_options["num_gpu"] = settings.ollama_num_gpu

        client = AsyncClient(host=settings.ollama_host)

        # Query which models are actually installed to avoid wasting time on missing ones
        try:
            installed_info = await client.list()
            installed_names = [m["model"] for m in installed_info.get("models", [])]
            logger.info(f"[LLM] Installed Ollama models: {installed_names}")
        except Exception:
            installed_names = []  # Ollama not running — will fail at chat stage

        # Build candidate list: configured model first, then any installed model as fallback
        candidates = []
        if not installed_names or settings.ollama_model in installed_names:
            candidates.append(settings.ollama_model)
        # Add any other installed model as emergency fallback
        for name in installed_names:
            if name not in candidates:
                candidates.append(name)

        if not candidates:
            candidates = [settings.ollama_model]  # last-resort attempt

        for ollama_model in candidates:
            logger.info(f"[LLM] Trying Ollama → {ollama_model}")
            try:
                async for chunk in await client.chat(
                    model=ollama_model,
                    messages=messages,
                    format="json",
                    options=ollama_options,
                    stream=True,
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                return  # success
            except Exception as e:
                err_str = str(e).lower()
                logger.error(f"[LLM] Ollama/{ollama_model} failed: {e}.")
                # VRAM exhausted — retry the same model on CPU (slow but functional)
                if "resource limitations" in err_str or "out of memory" in err_str:
                    logger.warning(f"[LLM] Ollama/{ollama_model} VRAM error — retrying on CPU (slow)")
                    try:
                        cpu_opts = {**ollama_options, "num_gpu": 0}
                        async for chunk in await client.chat(
                            model=ollama_model,
                            messages=messages,
                            format="json",
                            options=cpu_opts,
                            stream=True,
                        ):
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                yield delta
                        return  # success on CPU
                    except Exception as e2:
                        logger.error(f"[LLM] Ollama/{ollama_model} CPU fallback failed: {e2}.")

        # ── 5. All providers failed ────────────────────────────────────────
        logger.error("[LLM] All providers failed.")
        if self.language == "ur":
            speech = "سوری ... ابھی تکنیکی مسئلہ ہے ... تھوڑی دیر بعد کوشش کریں"
        elif self.language == "ro":
            speech = "Sori ... abhi technical masla hai ... thodi der baad koshish karein"
        else:
            speech = (
                f"Thank you for calling {settings.hospital_name}. "
                f"This is {settings.receptionist_name}. "
                f"I'm experiencing a brief technical issue. Please hold for a moment."
            )
        yield json.dumps({"speech": speech, "action": "ask", "data": {}})

    def _execute_booking(self, result: dict) -> dict:
        info = self._collected
        try:
            apt_date = datetime.strptime(info["appointment_date"], "%Y-%m-%d").date()
            appt = self.booking.book_appointment(
                patient_name=info["patient_name"],
                patient_phone=info["patient_phone"],
                doctor_name=info["doctor_name"],
                department=info["department"],
                appointment_date=apt_date,
                appointment_time=info["appointment_time"],
                reason=info.get("reason", ""),
                transcript=self.get_transcript(),
            )

            if self.language == "ur":
                result["speech"] = (
                    f"بالکل جی ... {appt.patient_name} صاحب ... "
                    f"ڈاکڑ {appt.doctor_name} کے ساتھ ... "
                    f"{appt.appointment_date.strftime('%d %B')} کو ... "
                    f"{appt.appointment_time} بجے ... "
                    f"بکنگ نمبر {appt.id} ہے ... کوئی اور بات"
                )
            elif self.language == "ro":
                result["speech"] = (
                    f"Bilkul ji ... {appt.patient_name} sahab ... "
                    f"Dr {appt.doctor_name} ke saath ... "
                    f"{appt.appointment_date.strftime('%d %B')} ko ... "
                    f"{appt.appointment_time} baje ... "
                    f"booking number {appt.id} hai ... koi aur baat"
                )
            else:
                result["speech"] = (
                    f"Perfect, you're all set! "
                    f"{appt.patient_name}, your appointment with {appt.doctor_name} "
                    f"is confirmed for {appt.appointment_date.strftime('%A, %B %d')} "
                    f"at {appt.appointment_time}. "
                    f"Your booking reference is #{appt.id}. "
                    f"Is there anything else I can help you with?"
                )
            result["action"] = "ask"
            result["data"]["appointment_id"] = appt.id

        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"[Booking] Failed: {error_msg}")

            if "phone number" in error_msg.lower():
                if self.language == "ur":
                    result["speech"] = "نمبر غلط لگتا ہے ... دوبارہ بتائیں"
                elif self.language == "ro":
                    result["speech"] = "Number galat lagta hai ... dobara bataein"
                else:
                    result["speech"] = "That phone number doesn't look right. Could you give me a valid number?"

            elif "not available" in error_msg.lower() or "slot" in error_msg.lower():
                # Fetch real alternatives from DB
                doctor = info.get("doctor_name", "")
                dept = info.get("department", "")
                try:
                    apt_date = datetime.strptime(info["appointment_date"], "%Y-%m-%d").date()
                    free_slots = self.booking.get_available_slots(doctor, apt_date)
                except Exception:
                    free_slots = []

                # Clear the bad time so LLM re-asks
                self._collected.pop("appointment_time", None)

                if free_slots:
                    slots_str_ur = "، ".join(free_slots[:4])
                    slots_str_en = ", ".join(free_slots[:4])
                    if self.language == "ur":
                        result["speech"] = (
                            f"یہ وقت نئیں ہے ... یہ دستیاب ہیں ... {slots_str_ur} ... کونسا ٹھیک ہے"
                        )
                    elif self.language == "ro":
                        result["speech"] = (
                            f"Yeh waqt nahi hai ... yeh available hain ... {slots_str_en} ... konsa theek hai"
                        )
                    else:
                        result["speech"] = (
                            f"Sorry, {info.get('appointment_time', 'that slot')} isn't available for {doctor}. "
                            f"Available times are: {slots_str_en}. Which works for you?"
                        )
                else:
                    # No slots at all for this doctor on this date — suggest other doctors
                    alternatives = self.booking.get_next_available(dept, days_ahead=5)
                    self._collected.pop("appointment_date", None)
                    if alternatives:
                        if self.language == "ur":
                            alt_lines = [f"{a['doctor']} {a['date']} کو {a['slots'][0]} بجے" for a in alternatives[:2] if a["doctor"] != doctor]
                        elif self.language == "ro":
                            alt_lines = [f"{a['doctor']} {a['date']} ko {a['slots'][0]} baje" for a in alternatives[:2] if a["doctor"] != doctor]
                        else:
                            alt_lines = [f"{a['doctor']} on {a['date']} at {a['slots'][0]}" for a in alternatives[:2] if a["doctor"] != doctor]
                        alt_str = " ... ".join(alt_lines) if alt_lines else ""
                        if self.language == "ur":
                            result["speech"] = (
                                f"ڈاکڑ {doctor} اس دن نئیں ہیں ... {alt_str} ... کوئی اور دن"
                            ) if alt_str else f"ڈاکڑ {doctor} دستیاب نئیں ... کوئی اور دن لیتے ہیں"
                        elif self.language == "ro":
                            result["speech"] = (
                                f"Dr {doctor} us din nahi hain ... {alt_str} ... koi aur din"
                            ) if alt_str else f"Dr {doctor} available nahi ... koi aur din lete hain"
                        else:
                            result["speech"] = (
                                f"Sorry, {doctor} has no availability on that date. "
                                f"Other options: {alt_str}. Would any of those work?"
                            )
                    else:
                        if self.language == "ur":
                            result["speech"] = f"ڈاکڑ {doctor} ابھی دستیاب نئیں ... کوئی اور ڈاکڑ لیتے ہیں"
                        elif self.language == "ro":
                            result["speech"] = f"Dr {doctor} abhi available nahi ... koi aur doctor lete hain"
                        else:
                            result["speech"] = f"Sorry, {doctor} isn't available then. Would you like to try a different doctor?"
            else:
                if self.language == "ur":
                    result["speech"] = "بکنگ نئیں ہوسکی ... دوبارہ کوشش کریں"
                elif self.language == "ro":
                    result["speech"] = "Booking nahi ho saki ... dobara koshish karein"
                else:
                    result["speech"] = f"I'm sorry, I couldn't complete the booking. Shall we try again?"
            result["action"] = "ask"

        except KeyError as e:
            missing_field = str(e).strip("'")
            logger.warning(f"[Booking] Missing field: {missing_field}")
            if self.language == "ur":
                result["speech"] = f"بکنگ کے لیے {missing_field.replace('_', ' ')} درکار ہے ... بتائیں"
            elif self.language == "ro":
                result["speech"] = f"Booking ke liye {missing_field.replace('_', ' ')} chahiye ... bataein"
            else:
                result["speech"] = f"I still need your {missing_field.replace('_', ' ')} to complete the booking."
            result["action"] = "ask"

        return result

    def _execute_cancellation(self, result: dict) -> dict:
        appt_id = self._collected.get("appointment_id")
        if appt_id:
            success = self.booking.cancel_appointment(int(appt_id))
            if success:
                result["speech"] = (
                    f"Your appointment #{appt_id} has been successfully cancelled. "
                    f"Is there anything else I can help you with?"
                )
            else:
                result["speech"] = (
                    f"I couldn't find a confirmed appointment with that reference. "
                    f"Could you double-check the booking ID?"
                )
        return result
