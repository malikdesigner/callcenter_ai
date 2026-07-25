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
from src.llm.intent import detect_intent, detect_language_switch, detect_language_preference, detect_symptom_department, validate_slot_input

# ── Session-level Gemini key exhaustion tracking ──────────────────────────────
# Keys are added here when their daily quota is confirmed exhausted (limit: 0).
# Avoids re-trying dead keys on every call within the same server session.
_gemini_exhausted_keys: set = set()

# ── Role & Strict Flow ────────────────────────────────────────────────────────

PLACEHOLDERS = {"unknown", "patient name", "phone number", "n/a", "none", "tbd", "placeholder"}
# reason is optional — patient may not give one explicitly
REQUIRED_FIELDS = ["patient_name", "patient_phone", "department", "doctor_name", "appointment_date", "appointment_time"]

# ── Hospital knowledge base (editable by admin) ────────────────────────────────

import os as _os

_KNOWLEDGE_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), "data", "hospital_knowledge.json")
_hospital_knowledge: dict = {}

def _load_hospital_knowledge() -> dict:
    """Load admin-editable hospital knowledge from data/hospital_knowledge.json."""
    global _hospital_knowledge
    try:
        with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            _hospital_knowledge = json.load(f)
        logger.info(f"[Knowledge] Loaded hospital knowledge ({len(_hospital_knowledge)} sections)")
    except FileNotFoundError:
        logger.warning(f"[Knowledge] hospital_knowledge.json not found — non-booking queries will use generic responses")
        _hospital_knowledge = {}
    except Exception as e:
        logger.error(f"[Knowledge] Failed to load hospital_knowledge.json: {e}")
        _hospital_knowledge = {}
    return _hospital_knowledge

def _build_knowledge_block(lang: str = "ur") -> str:
    """Convert hospital knowledge to a compact prompt block for Ollama."""
    kb = _hospital_knowledge or _load_hospital_knowledge()
    if not kb:
        return ""

    lines = ["━━━ HOSPITAL KNOWLEDGE (answer info queries from this) ━━━"]

    hosp = kb.get("hospital", {})
    if hosp:
        lines.append(f"Hospital: {hosp.get('name','')} | {hosp.get('address','')} | Tel: {hosp.get('phone','')} | Emergency: {hosp.get('emergency','')}")

    timings = kb.get("timings", {})
    if timings:
        t_parts = [f"{k}: {v}" for k, v in timings.items()]
        lines.append("Timings: " + " | ".join(t_parts))

    fees = kb.get("fees", {})
    if fees:
        f_parts = [f"{k}: {v}" for k, v in fees.items()]
        lines.append("Fees: " + " | ".join(f_parts))

    policies = kb.get("policies", {})
    if policies:
        p_parts = [f"{k}: {v}" for k, v in policies.items()]
        lines.append("Policies: " + " | ".join(p_parts))

    faqs = kb.get("faqs", [])
    if faqs:
        faq_parts = [f"Q: {f['q']} → A: {f['a']}" for f in faqs]
        lines.append("FAQs:\n" + "\n".join(faq_parts))

    out_of_scope = kb.get("out_of_scope_response", {})
    if out_of_scope:
        oos = out_of_scope.get(lang, out_of_scope.get("en", ""))
        if oos:
            lines.append(f"OUT_OF_SCOPE RESPONSE (if asked something unrelated to hospital): \"{oos}\"")

    lines.append(
        "INFO QUERY RULE: If user asks anything from the knowledge above → answer it naturally and warmly, "
        "then return to booking flow if appointment was in progress. "
        "If completely out of scope (car, politics, etc.) → use OUT_OF_SCOPE RESPONSE."
    )
    return "\n".join(lines)

# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(lang: str = "en") -> str:
    with Session(engine) as session:
        doctors = session.exec(select(Doctor).where(Doctor.is_active == True)).all()
        depts = session.exec(select(Department)).all()

    # Build doctor list with specialty + fee so LLM can match symptoms and answer fee queries
    doctor_info = []
    for dept in depts:
        dept_docs = []
        for d in doctors:
            if d.department_name == dept.name:
                fee_part = f", fee={d.fee}" if d.fee else ""
                dept_docs.append(f"{d.name} ({d.specialty}{fee_part})")
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

━━━ BOOKING FLOW — STRICT ORDER ━━━
1. نام ملا → فون نمبر مانگیں
2. فون ملا → تکلیف پوچھیں: "آپکو کیا تکلیف ہے؟"
   NEVER offer a menu like "معلومات چاہیے یا اپوائنٹمنٹ" — just ask for symptoms directly.
3. تکلیف بتائی → empathy + SPECIFIC DOCTOR recommend کریں by name
   ✓ "اچھا جی ... اس کے لیے ڈاکڑ [نام] ہیں ... کیا ان سے وقت لوں؟"
   ✗ NEVER "کونسے شعبے میں جانا ہے"
4. Doctor confirm → تاریخ پوچھیں
5. تاریخ ملی → دستیاب slot دکھائیں (صرف 3 slots)
6. Slot confirm → booking confirm کریں → action="save"
• One thing per turn. Acknowledge before asking.
• NEVER repeat exact same question twice — rephrase.
• action="end" ONLY after booking done + patient says goodbye.

━━━ OUTPUT RULES ━━━
1. Urdu script only (ا ب پ...). Zero English or Roman Urdu.
2. NATURAL CONVERSATIONAL TONE: Talk like a real, polite Pakistani receptionist. Use complete, warm sentences.
   ✓ السلام علیکم، میں سارہ بات کر رہی ہوں، بتائیے میں آپ کی کیا مدد کر سکتی ہوں؟
   ✓ جی بالکل، کوئی مسئلہ نہیں، آپ کا فون نمبر کیا ہے؟
   ✗ جناب میں آپ کی کیا مدد کرسکتی ہوں (too formal)
   ✗ جی سر ... کیا مسئلہ ہے (too abrupt/robotic)
3. Punctuation: Use ، (Urdu comma) for ALL pauses and between clauses. Use ؟ for questions. NEVER use ۔ — it causes a dead halt. Every mid-sentence pause must be ،
4. Empathy: Show empathy when hearing symptoms. "اچھا، مجھے سن کر افسوس ہوا، کوئی فکر کی بات نہیں،"
5. JSON FORMAT: Output a single compact line of JSON — no newlines, no extra spaces. {{"speech":"...","action":"...","data":{{...}}}}
6. Tone words: جی بالکل، ضرور، فکر نہ کریں، مہربانی، شکریہ۔
7. BANNED words (Too formal/bookish): براہ کرم / معافی / جناب / محترم / لہذا / اپوائنٹمنٹ۔

VARIED FILLERS — rotate naturally:
  اچھا | جی بالکل | ٹھیک ہے | میں سمجھ سکتی ہوں | ضرور

UNCLEAR AUDIO — if user speech is unrecognizable or nonsensical:
  → معذرت، مجھے آپ کی آواز واضح نہیں آئی۔ کیا آپ دہرا سکتے ہیں؟
WRONG TYPE — if user gives digits when you asked for name, or gibberish for date:
  → [acknowledge politely] ... براہ مہربانی [نام/تاریخ/وقت] دوبارہ بتائیں۔
NAME RULE — patient_name must be a real name (1-4 words, no digit strings).
SLOTS RULE — NEVER list more than 3 time slots. Pick the 3 most convenient ones.

CONVERSATIONAL EXAMPLES (copy this natural style — note: ، not ۔):
سلام:    السلام علیکم، میں سارہ بات کر رہی ہوں، بتائیے میں آپ کی کیا مدد کر سکتی ہوں؟
فون:     جی بالکل، آپ کا مکمل فون نمبر کیا ہو گا؟
تکلیف:  اچھا، فکر نہ کریں، ڈاکٹر [نام] ان مسائل کے ماہر ہیں، کیا میں ان کے ساتھ آپ کا وقت بک کر دوں؟
تاریخ:  ضرور، آپ کس دن آنا پسند کریں گے؟
وقت:    ٹھیک ہے، میرے پاس یہ اوقات دستیاب ہیں: [3 سلاٹس]، ان میں سے کون سا وقت آپ کے لیے بہتر رہے گا؟
تصدیق:  ٹھیک ہے [نام] صاحب، میں نے ڈاکٹر [نام] کے ساتھ آپ کا وقت [تاریخ] کو [وقت] کے لیے بک کر دیا ہے، کیا میں اسے کنفرم کر دوں؟
اختتام: بہت شکریہ آپ کا، اللہ حافظ،

{_build_knowledge_block("ur")}

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

━━━ BOOKING FLOW — STRICT ORDER ━━━
1. Naam mila → phone number maangein
2. Phone mila → takleef poochein: "Aapko kya takleef hai?"
   NEVER offer a menu like "maloomat chahiye ya appointment" — seedha symptoms poochein.
3. Takleef batayi → empathy + SPECIFIC DOCTOR naam se recommend karein
   ✓ "Acha ji ... is ke liye Dr [naam] hain ... kya unse waqt loon?"
   ✗ NEVER "konse department mein jana hai"
4. Doctor confirm → date poochein
5. Date mili → available slots dikhayein (sirf 3 slots)
6. Slot confirm → booking confirm karein → action="save"
• One thing per turn. Acknowledge before asking.
• NEVER repeat exact same question twice — rephrase.
• action="end" ONLY after booking done + patient says goodbye.

━━━ OUTPUT RULES ━━━
1. Roman Urdu only (Latin script). Zero Urdu script or English.
2. NATURAL CONVERSATIONAL TONE: Talk like a real, polite Pakistani receptionist. Use complete, warm sentences.
   ✓ Assalam o alaikum, main Sara baat kar rahi hoon. Bataiye main aapki kya madad kar sakti hoon?
   ✓ Ji bilkul, koi masla nahi. Aapka phone number kya hai?
   ✗ Janab main aapki kya madad kar sakti hoon (too formal)
   ✗ Ji sir ... kya masla hai (too abrupt/robotic)
3. Punctuation: Use proper punctuation (. ? ,) so the TTS engine can pace the speech naturally.
4. Empathy: Show empathy when hearing symptoms. "Acha, mujhe sun kar afsos hua. Koi fikar ki baat nahi..."
5. Tone words: Ji bilkul, zaroor, fikar na karein, meherbani, shukriya.
6. BANNED (Too formal/bookish): bara-e-meherbani / maafi / janab / muhtaram / lihaza / appointment.

VARIED FILLERS — rotate naturally:
  acha | ji bilkul | theek hai | main samajh sakti hoon | zaroor

UNCLEAR AUDIO — if user speech is unrecognizable or nonsensical:
  → Maazrat, mujhe aapki awaaz wazeh nahi aayi. Kya aap dohra sakte hain?
WRONG TYPE — digits when asked for name, or gibberish for date:
  → [acknowledge politely] ... meherbani farma kar [naam/date/waqt] dobara bataein.
NAME RULE — patient_name must be a real name (1-4 words, no digit strings only).
SLOTS RULE — NEVER list more than 3 time slots. Pick the 3 most convenient ones.

CONVERSATIONAL EXAMPLES (copy this natural style exactly):
Salam:    Assalam o alaikum, main Sara baat kar rahi hoon. Bataiye main aapki kya madad kar sakti hoon?
Phone:    Ji bilkul. Aapka mukammal phone number kya hoga?
Takleef:  Acha, fikar na karein. Dr [naam] is ke mahir hain. Kya main unke sath aapka waqt book kar doon?
Date:     Zaroor. Aap kis din aana pasand karein ge?
Waqt:     Theek hai. Mere paas yeh waqt hain... [3 slots]. In mein se konsa waqt aap ke liye behtar rahe ga?
Confirm:  Theek hai [naam] sahab. Main ne Dr [naam] ke sath aapka waqt [date] ko [waqt] ke liye book kar diya hai. Kya main isay confirm kar doon?
Ikhtitaam: Bahut shukriya aapka. Allah hafiz.

{_build_knowledge_block("ro")}

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

{_build_knowledge_block("en")}

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
        # Fast dialogue engine (instant responses, no Ollama needed for structured steps)
        from src.dialogue.fast_engine import FastDialogueEngine
        self._fast_engine = FastDialogueEngine(language=self.language)

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
        fallback_ur = f"السلام علیکم، میں سارہ بات کر رہی ہوں۔ بتائیے میں آپ کی کیا مدد کر سکتی ہوں؟"
        fallback_ro = f"Assalam o alaikum, main Sara baat kar rahi hoon. Bataiye main aapki kya madad kar sakti hoon?"

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

    async def process_turn_stream(self, user_text: str, audio_language: str = None, policy_state: str = "", nlg_constraint: str = ""):
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
                lang = "ur"
                speech = "جی ضرور!"
                logger.info("[Agent] Bilingual: Urdu detected → switching to Urdu in-place")
            elif pref == "en":
                lang = "en"
                speech = "Sure!"
                logger.info("[Agent] Bilingual: English detected → switching to English in-place")
            else:
                speech = "سمجھی نئیں ... اردو یا English کہیں / Please say Urdu or English"
                logger.info(f"[Agent] Bilingual: unclear (transcript='{user_text}', audio_lang={audio_language})")
                yield speech
                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": speech})
                yield {"speech": speech, "action": "ask", "data": {}}
                return

            # Language confirmed — switch in-place, no disconnect
            self.language = lang
            self._language_chosen = True
            self._bilingual_mode = False
            yield speech
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": speech})
            yield {"speech": speech, "action": "switch_language", "data": {"language": lang}}
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
            self._fast_engine.language = "en"
            logger.info("[Agent] Language switched to 'en' (explicit command)")
        elif intent == "change_language_ur" and self.language != "ur":
            self.language = "ur"
            self._fast_engine.language = "ur"
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
            # Fast engine checks for YES words (more reliable than intent detector)
            fast_confirm = self._fast_engine.try_handle(user_text, "confirm_booking", self._collected)
            if fast_confirm is not None and fast_confirm.get("action") == "save":
                intent = "confirm"  # unify the path below

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

        # ── Fast path: instant response for structured booking steps ─────────
        # Skip Ollama entirely for name/phone/date/time collection when the
        # user gives a clear, structured answer. Ollama only gets called for
        # symptoms, doctor recommendation, confusion, and corrections.
        if not is_correction and not is_non_answer:
            fast = self._fast_engine.try_handle(
                user_text, self._flow_state, self._collected
            )
            if fast is not None:
                logger.info(f"[Fast] Handled instantly — state={self._flow_state}, action={fast.get('action')}")
                for k, v in fast.get("data", {}).items():
                    if v:
                        self._collected[k] = v
                self._normalise_collected()

                # Fast engine confirmed booking — execute it directly
                if fast.get("action") == "save":
                    fast = self._execute_booking(fast)
                    self._awaiting_confirmation = False
                    self._flow_state = "post_booking"
                    self._history.append({"role": "user", "content": user_text})
                    self._history.append({"role": "assistant", "content": fast.get("speech", "")})
                    if len(self._history) > 16:
                        self._history = self._history[-16:]
                    if fast.get("speech"):
                        yield fast["speech"]
                    yield fast
                    return
                else:
                    self._flow_state = self._compute_flow_state()

                # If all info is now collected, yield the fast ack then chain immediately
                # to the LLM for the confirmation summary — no need to wait for another turn.
                if self._flow_state == "confirm_booking":
                    self._history.append({"role": "user", "content": user_text})
                    self._history.append({"role": "assistant", "content": fast.get("speech", "")})
                    if len(self._history) > 16:
                        self._history = self._history[-16:]
                    if fast.get("speech"):
                        yield fast["speech"]
                    # Fall through to LLM with a synthetic trigger so it generates the summary
                    user_text = "جی" if self.language in ("ur", "ro") else "okay"
                    # (don't return — fall through to _chat_stream below)
                else:
                    self._history.append({"role": "user", "content": user_text})
                    self._history.append({"role": "assistant", "content": fast.get("speech", "")})
                    if len(self._history) > 16:
                        self._history = self._history[-16:]
                    if fast.get("speech"):
                        yield fast["speech"]
                    yield fast
                    return

        async for content in self._chat_stream(user_text, policy_state=policy_state, nlg_constraint=nlg_constraint):
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
                    
                    # Yield on sentence/clause boundaries — ، (Urdu comma) added so
                    # comma-delimited Urdu phrases stream to TTS without waiting for ؟
                    sentences = re.split(r'(?<=[.!?۔؟،])\s+', speech_buffer)
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            if s.strip():
                                yield s.strip()
                        speech_buffer = sentences[-1]

        # Yield whatever remains in the buffer
        _buffer_yielded = speech_buffer.strip()
        if _buffer_yielded:
            yield _buffer_yielded

        # Parse final JSON — strip markdown fences Gemini sometimes adds
        logger.debug(f"[LLM] Raw Response: {full_json_str}")
        _json_text = full_json_str.strip()
        if _json_text.startswith("```"):
            _json_text = re.sub(r'^```(?:json)?\s*', '', _json_text)
            _json_text = re.sub(r'\s*```\s*$', '', _json_text).strip()
        try:
            result = json.loads(_json_text)
        except Exception as _json_err:
            logger.warning(f"[LLM] JSON parse error: {_json_err} | len={len(_json_text)} | tail={_json_text[-80:]!r}")
            # Try clean closed match first, then greedy partial extraction for truncated JSON
            closed = re.search(r'"speech"\s*:\s*"(.*?)"(?=\s*[,}])', _json_text, re.DOTALL)
            if closed:
                speech = closed.group(1)
            else:
                partial = re.search(r'"speech"\s*:\s*"(.+)', _json_text, re.DOTALL)
                if partial:
                    raw = partial.group(1)
                    end = raw.find('"')
                    speech = raw[:end] if end != -1 else raw  # use whole tail if no closing quote
                else:
                    speech = ""
            if speech:
                logger.warning(f"[LLM] JSON parse failed — recovered speech ({len(speech)} chars)")
            result = {"speech": speech, "action": "ask", "data": {}}

        # Safety net: if nothing was yielded to TTS during streaming, yield the parsed speech.
        # Gemini without response_format sends full JSON in 1-2 large chunks,
        # so streaming sentence extraction sees no boundary → speech_buffer stays whole.
        _result_speech = result.get("speech", "").strip()
        _nothing_yielded = not _buffer_yielded and last_yielded_speech_idx == 0
        _is_error_msg = "trouble thinking" in _result_speech or "technical" in _result_speech
        if _nothing_yielded and _result_speech and not _is_error_msg:
            logger.debug(f"[LLM] Yielding speech from parsed JSON ({len(_result_speech)} chars)")
            yield _result_speech

        # ── State Security ──
        # Filter and Merge collected data
        if result.get("data"):
            for k, v in result["data"].items():
                val = str(v).strip() if v else ""
                # Ignore placeholders and empty strings
                if val and val.lower() not in PLACEHOLDERS:
                    self._collected[k] = val

        # ── Doctor name auto-extraction fallback ──────────────────────────────
        # When Ollama recommends a doctor in speech but forgets to put it in data,
        # scan the speech for known doctor names and auto-populate collected.
        if not self._collected.get("doctor_name"):
            self._try_extract_doctor_from_speech(result.get("speech", ""))

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

    def _try_extract_doctor_from_speech(self, speech: str):
        """
        Scan Ollama's speech for known doctor names and auto-populate collected.
        Called when doctor_name is missing after Ollama's turn — prevents state sticking
        at ask_information when Ollama says the doctor name in speech but not in data JSON.
        """
        if not speech:
            return
        with Session(engine) as session:
            doctors = session.exec(select(Doctor).where(Doctor.is_active == True)).all()
        for doc in doctors:
            if doc.name and doc.name in speech:
                self._collected["doctor_name"] = doc.name
                if doc.department_name and not self._collected.get("department"):
                    self._collected["department"] = doc.department_name
                logger.info(f"[Agent] Auto-extracted doctor from speech: {doc.name!r}")
                return

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

        # ── Name validation: reject digit strings stored as names ────────────
        # Happens when Whisper transcribes a phone number spoken during name collection.
        raw_name = self._collected.get("patient_name", "")
        if raw_name:
            name_digits = "".join(c for c in raw_name if c.isdigit())
            if len(name_digits) >= 6:
                logger.warning(f"[Agent] Rejecting digit-string as name: '{raw_name}'")
                self._collected.pop("patient_name", None)

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

    def _build_context_injection(self, user_text: str = "", policy_state: str = "", nlg_constraint: str = "") -> str:
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
            
        if policy_state:
            lines.append(f"POLICY_STATE: {policy_state}")
        if nlg_constraint:
            lines.append(f"POLICY_CONSTRAINT: {nlg_constraint}\nCRITICAL RULE: You MUST follow this policy constraint strictly.")

        missing = self._get_missing_requirements()
        if missing:
            lines.append(f"STILL_NEEDED: {', '.join(missing)}")

            # ── ask_information: Ollama's ONE job — get symptoms + recommend doctor ──
            if self._flow_state == "ask_information" and not self._collected.get("doctor_name"):
                try:
                    with Session(engine) as session:
                        all_docs = session.exec(select(Doctor).where(Doctor.is_active == True)).all()
                    doc_list = " | ".join(f"{d.name} ({d.specialty})" for d in all_docs)
                except Exception:
                    doc_list = ""
                if self.language in ("ur", "ro"):
                    lines.append(
                        f"YOUR ONLY JOB THIS TURN:\n"
                        f"1. User ne symptoms bataye → ek specific doctor ka naam choose karo\n"
                        f"2. data.doctor_name aur data.department JSON mein zaroor likhein\n"
                        f"Available doctors: {doc_list}\n"
                        f"Example: agar bukhar/khansee → General Medicine doctor\n"
                        f"Example: agar seenay mein dard → Cardiology doctor\n"
                        f"DO NOT ask 'konsa department?' — seedha doctor suggest karo"
                    )
                else:
                    lines.append(
                        f"YOUR ONLY JOB THIS TURN:\n"
                        f"1. Hear the symptoms → pick ONE specific doctor from the list\n"
                        f"2. MUST include data.doctor_name AND data.department in JSON\n"
                        f"Available doctors: {doc_list}\n"
                        f"NEVER ask 'which department?' — recommend the doctor directly"
                    )

            if non_answering:
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
            if self.language in ("ur", "ro"):
                lines.append(
                    "ALL_INFO_COLLECTED: Saari maloomat mil gayi — confirmation ke liye tayar.\n"
                    "CONFIRMATION RULE: Sirf naam, doctor, date, time ka mukhtasar khulasa karo. "
                    "Fee aur WhatsApp instructions system khud add kar dega — tum mat batao."
                )
            else:
                lines.append(
                    "ALL_INFO_COLLECTED: All info ready — proceed to confirmation.\n"
                    "CONFIRMATION RULE: Read back name, doctor, date, and time only. "
                    "Fee and WhatsApp instructions are added automatically — do not include them."
                )

        if self.language == "ur":
            lines.append("CRITICAL LANGUAGE RULE: Respond in highly natural, fluent Pakistani Urdu. Do NOT translate English phrases literally. Use proper Urdu grammar, sentence structure, and vocabulary as a real human receptionist would.")


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

        # ── Slot semantic validation ──────────────────────────────────────────
        # If the user's response doesn't match what was asked for, inject a hint
        # so the LLM asks them to clarify rather than silently accepting garbage.
        if user_text and self._flow_state in ("collect_name", "collect_date", "collect_time"):
            is_valid, reason = validate_slot_input(user_text, self._flow_state, self.language)
            if not is_valid:
                _SLOT_LABELS = {
                    "collect_name": {"ur": "نام", "ro": "naam", "en": "name"},
                    "collect_date": {"ur": "تاریخ", "ro": "date", "en": "date"},
                    "collect_time": {"ur": "وقت", "ro": "waqt", "en": "time"},
                }
                label = _SLOT_LABELS.get(self._flow_state, {}).get(self.language, self._flow_state)
                if reason == "digit_as_name":
                    lines.append(
                        f"SLOT_MISMATCH: You asked for the patient's name but received "
                        f"'{user_text}' which looks like a phone number, not a name. "
                        f"Do NOT store this as the name. Ask for their name again clearly."
                    )
                elif reason == "no_date_words":
                    lines.append(
                        f"SLOT_MISMATCH: You asked for a date/day but '{user_text}' "
                        f"contains no recognizable date. Ask them to repeat the day clearly."
                    )
                elif reason == "no_time_words":
                    lines.append(
                        f"SLOT_MISMATCH: You asked for a time but '{user_text}' "
                        f"contains no recognizable time. Ask them to repeat which time slot."
                    )

        # ── Case 1: Doctor + Date known → show exact slots from DB ───────────
        if doctor and raw_date:
            try:
                appt_date = _date.fromisoformat(raw_date)
                available_slots = self.booking.get_available_slots(doctor, appt_date)
                if available_slots:
                    slots_str = ", ".join(available_slots[:4])  # cap at 4 — never read 15 slots aloud
                    lines.append(
                        f"AVAILABLE_SLOTS for {doctor} on {raw_date}: {slots_str}\n"
                        f"STRICT RULE: Offer AT MOST 3 specific slots. Do NOT list all of them. "
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

    async def _chat_stream(self, user_input: str, policy_state: str = "", nlg_constraint: str = ""):
        """
        Yields raw content delta strings (str) from the LLM.
        Urdu / Roman Urdu : Gemini 2.5 Flash (primary) → Ollama (fallback).
        English            : Ollama only.
        Set GEMINI_API_KEY in .env to enable Gemini.
        """
        system = _build_system_prompt(self.language)
        ctx = self._build_context_injection(user_input, policy_state, nlg_constraint)
        if ctx:
            system += f"\n\n--- LIVE CONTEXT ---\n{ctx}"

        messages = [{"role": "system", "content": system}]
        messages += self._history
        messages.append({"role": "user", "content": user_input})

        from openai import AsyncOpenAI

        # ── Gemini — primary for Urdu / Roman Urdu (rotates across up to 6 keys) ──
        # English skips Gemini entirely and goes straight to Ollama.
        global _gemini_exhausted_keys
        if self.language in ("ur", "ro"):
            all_keys = [k for k in [
                settings.gemini_api_key,
                settings.gemini_api_key1,
                settings.gemini_api_key2,
                settings.gemini_api_key3,
                settings.gemini_api_key4,
                settings.gemini_api_key5,
            ] if k]
            live_keys = [k for k in all_keys if k not in _gemini_exhausted_keys]

            for api_key in live_keys:
                tag = f"...{api_key[-6:]}"
                gemini_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                )
                logger.info(f"[LLM] Trying Gemini ({settings.gemini_model}) [key {tag}]")
                try:
                    for attempt in range(2):  # retry once on empty response
                        stream = await asyncio.wait_for(
                            gemini_client.chat.completions.create(
                                model=settings.gemini_model,
                                messages=messages,
                                temperature=0.2 if attempt == 0 else 0.4,
                                max_tokens=1200,
                                stream=True,
                            ),
                            timeout=12.0,
                        )
                        chars_received = 0
                        async for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield delta
                                chars_received += len(delta)
                        if chars_received > 0:
                            break
                        logger.warning(f"[LLM] Gemini key {tag} empty response (attempt {attempt+1}/2) — retrying.")
                    if chars_received == 0:
                        logger.warning(f"[LLM] Gemini key {tag} empty after 2 attempts — trying next key.")
                        continue
                    logger.info(f"[LLM] ✓ Answered by Gemini ({settings.gemini_model}) [key {tag}]")
                    return
                except asyncio.TimeoutError:
                    logger.warning(f"[LLM] Gemini key {tag} timed out (12s) — trying next key.")
                except Exception as e:
                    err_str = str(e)
                    _daily_exhausted = (
                        "limit: 0" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "exceeded your current quota" in err_str
                        or "GenerateRequestsPerDay" in err_str
                    )
                    if _daily_exhausted:
                        _gemini_exhausted_keys.add(api_key)
                        remaining = len([k for k in all_keys if k not in _gemini_exhausted_keys])
                        logger.warning(f"[LLM] Gemini key {tag} daily quota exhausted ({remaining} key(s) left) — reason: {e}")
                    elif "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        logger.warning(f"[LLM] Gemini key {tag} rate-limited — reason: {e}")
                    else:
                        logger.error(f"[LLM] Gemini key {tag} failed — reason: {e}")

            if all_keys:
                logger.warning("[LLM] All Gemini keys failed — trying Groq.")

        # ── Groq — fast cloud fallback (Urdu + English) ───────────────────────
        if settings.groq_api_key:
            groq_client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            logger.info(f"[LLM] Trying Groq ({settings.groq_model})")
            try:
                stream = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=settings.groq_model,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=600,
                        stream=True,
                    ),
                    timeout=10.0,
                )
                chars_received = 0
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                        chars_received += len(delta)
                if chars_received > 0:
                    logger.info(f"[LLM] ✓ Answered by Groq ({settings.groq_model})")
                    return
                logger.warning("[LLM] Groq returned empty response — falling back to Ollama.")
            except asyncio.TimeoutError:
                logger.warning("[LLM] Groq timed out (10s) — falling back to Ollama.")
            except Exception as e:
                logger.error(f"[LLM] Groq failed: {e} — falling back to Ollama.")

        # ── Ollama (local) — primary for English, fallback for Urdu/RO ────────
        from ollama import AsyncClient
        ollama_options = {"temperature": 0.2, "num_predict": 350, "num_ctx": 2048}
        if settings.ollama_num_gpu >= 0:
            ollama_options["num_gpu"] = settings.ollama_num_gpu

        ollama_client = AsyncClient(host=settings.ollama_host)

        try:
            installed_info = await ollama_client.list()
            installed_names = [m["model"] for m in installed_info.get("models", [])]
            logger.debug(f"[LLM] Installed Ollama models: {installed_names}")
        except Exception:
            installed_names = []

        candidates = []
        if not installed_names or settings.ollama_model in installed_names:
            candidates.append(settings.ollama_model)
        for name in installed_names:
            if name not in candidates:
                candidates.append(name)
        if not candidates:
            candidates = [settings.ollama_model]

        for ollama_model in candidates:
            logger.info(f"[LLM] Trying Ollama → {ollama_model}")
            try:
                async for chunk in await ollama_client.chat(
                    model=ollama_model,
                    messages=messages,
                    format="json",
                    options=ollama_options,
                    stream=True,
                ):
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                logger.info(f"[LLM] ✓ Answered by Ollama ({ollama_model})")
                return
            except Exception as e:
                err_str = str(e).lower()
                logger.error(f"[LLM] Ollama/{ollama_model} failed: {e}.")
                if "resource limitations" in err_str or "out of memory" in err_str:
                    logger.warning(f"[LLM] Ollama/{ollama_model} VRAM error — retrying on CPU (slow)")
                    try:
                        cpu_opts = {**ollama_options, "num_gpu": 0}
                        async for chunk in await ollama_client.chat(
                            model=ollama_model,
                            messages=messages,
                            format="json",
                            options=cpu_opts,
                            stream=True,
                        ):
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                yield delta
                        logger.info(f"[LLM] ✓ Answered by Ollama/{ollama_model} (CPU fallback)")
                        return
                    except Exception as e2:
                        logger.error(f"[LLM] Ollama/{ollama_model} CPU fallback failed: {e2}.")

        # ── All providers failed ───────────────────────────────────────────────
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
                    f"بالکل جی، {appt.patient_name} صاحب، "
                    f"آپ کا وقت بک ہوگیا، "
                    f"ڈاکڑ {appt.doctor_name} کے ساتھ، "
                    f"{appt.appointment_date.strftime('%d %B')} کو، "
                    f"{appt.appointment_time}، "
                    f"بکنگ نمبر {appt.id}، "
                    f"آپ کو ابھی WhatsApp پیغام آئے گا، "
                    f"فیس جمع کروانے کا اسکرین شاٹ بھیجیں، "
                    f"کیا کوئی اور مدد چاہیے؟"
                )
            elif self.language == "ro":
                result["speech"] = (
                    f"Bilkul ji. {appt.patient_name} sahab, "
                    f"aapki appointment book ho gayi hai. "
                    f"Dr {appt.doctor_name} ke saath, "
                    f"{appt.appointment_date.strftime('%d %B')} ko, "
                    f"{appt.appointment_time} baje. "
                    f"Booking number {appt.id}. "
                    f"Aapko abhi WhatsApp message aayega — "
                    f"fee jama kerwane ka screenshot bhej dein. "
                    f"Kya koi aur madad chahiye?"
                )
            else:
                result["speech"] = (
                    f"You're all set, {appt.patient_name}! "
                    f"Appointment confirmed with {appt.doctor_name} "
                    f"on {appt.appointment_date.strftime('%A, %B %d')} "
                    f"at {appt.appointment_time}. "
                    f"Booking reference #{appt.id}. "
                    f"You'll receive a WhatsApp message shortly — "
                    f"please send a screenshot of the fee payment. "
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
