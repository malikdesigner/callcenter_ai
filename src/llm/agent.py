"""
Hospital receptionist LLM agent powered by Ollama (local, no API key).
Manages multi-turn conversation and triggers appointment booking actions.
"""

import json
import re
from datetime import date, datetime
from typing import Optional

import ollama
from loguru import logger
from sqlmodel import Session, select

from config.settings import settings
from src.appointment.booking import (
    BookingSystem,
    resolve_department,
)
from src.appointment.models import Appointment, Doctor, Department, engine

# ── Role & Strict Flow ────────────────────────────────────────────────────────

PLACEHOLDERS = {"unknown", "patient name", "phone number", "n/a", "none", "tbd", "placeholder"}
# reason is optional — patient may not give one explicitly
REQUIRED_FIELDS = ["patient_name", "patient_phone", "department", "doctor_name", "appointment_date", "appointment_time"]

# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(lang: str = "en") -> str:
    with Session(engine) as session:
        doctors = session.exec(select(Doctor).where(Doctor.is_active == True)).all()
        depts = session.exec(select(Department)).all()

    doctor_info = []
    for dept in depts:
        dept_docs = [d.name for d in doctors if d.department_name == dept.name]
        if dept_docs:
            doctor_info.append(f"{dept.name.title()}: " + ", ".join(dept_docs))
    doctor_list_str = " | ".join(doctor_info)

    today = date.today().isoformat()
    tomorrow = (date.today() + __import__('datetime').timedelta(days=1)).isoformat()
    current_time = datetime.now().strftime("%I:%M %p")

    if lang == "ur":
        return f"""آپ سارہ ہیں — {settings.hospital_name} کی ریسپشنسٹ۔ عام پاکستانی اردو میں بات کریں۔
آج: {today} | وقت: {current_time} | کل: {tomorrow}
ڈاکٹرز: {doctor_list_str}

ترتیب: نام ← فون نمبر ← تکلیف ← ڈاکٹر/شعبہ ← وقت (صرف دستیاب سلاٹس) ← تصدیق ← بکنگ
ہر بار صرف ایک سوال۔ جو معلومات مل گئی دوبارہ نہ پوچھیں۔ جواب مختصر رکھیں۔
action: ask=معلومات چاہیے | confirm=سب تیار، تصدیق لیں | save=تصدیق ہوگئی، بک کریں | end=کال ختم
صرف JSON آؤٹ پٹ: {{"speech":"...","action":"ask|confirm|save|end","data":{{"patient_name":"","patient_phone":"","department":"","doctor_name":"","appointment_date":"","appointment_time":"","reason":""}}}}"""

    return f"""You are {settings.receptionist_name}, receptionist at {settings.hospital_name}. Speak like a real human on a phone call — warm, brief, natural.
Today: {today} | Time: {current_time} | Tomorrow: {tomorrow}
Doctors: {doctor_list_str}

Flow: name → phone → problem → doctor/dept → time (only from available slots) → confirm → book
One question per turn. Never re-ask info already given. Keep responses short.
action: ask=need info | confirm=all ready, read back & confirm | save=user confirmed, book it | end=call over
JSON only: {{"speech":"...","action":"ask|confirm|save|end","data":{{"patient_name":"","patient_phone":"","department":"","doctor_name":"","appointment_date":"","appointment_time":"","reason":""}}}}"""

# ── Agent class ────────────────────────────────────────────────────────────────

class HospitalAgent:
    def __init__(self, language: str = "en"):
        self.language = language
        self.booking = BookingSystem()
        self._history: list = []
        self._collected: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Reset state for a new call."""
        self._history = []
        self._collected = {}

    async def get_greeting(self) -> str:
        prompt = "A new caller just connected. Greet them naturally like a real receptionist — brief and warm."
        if self.language == "ur":
            prompt = "نیا کال آیا ہے۔ عام پاکستانی ریسپشنسٹ کی طرح مختصر اور دوستانہ انداز میں سلام کریں اور نام پوچھیں۔"
        result = await self._chat(prompt)

        fallback_en = f"Thank you for calling {settings.hospital_name}, this is {settings.receptionist_name}. How can I help you?"
        fallback_ur = f"السلام علیکم، {settings.hospital_name} میں خوش آمدید! میں سارہ بول رہی ہوں۔ آپ کا نام کیا ہے؟"
        
        return result.get("speech", fallback_en if self.language == "en" else fallback_ur)

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

    async def process_turn_stream(self, user_text: str):
        """
        Yields strings (sentences) as they are generated by the LLM.
        The very last item yielded will be the complete response dict.
        """
        full_json_str = ""
        last_yielded_speech_idx = 0
        speech_buffer = ""

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
                    sentences = re.split(r'(?<=[.!?])\s+', speech_buffer)
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
            labels = [f.replace('patient_', '').replace('_', ' ') for f in missing]
            if self.language == "ur":
                result["speech"] = f"بکنگ کے لیے ابھی {' اور '.join(labels)} درکار ہے۔ کیا آپ یہ بتا سکتے ہیں؟"
            else:
                result["speech"] = f"I still need {', and '.join(labels)} before I can book. Could you provide that?"
            action = "ask"
            result["action"] = "ask"

        if action == "save":
            logger.info(f"[Agent] Executing booking — collected: {self._collected}")
            result = self._execute_booking(result)
        elif action == "confirm" and not missing:
            # User confirmed — proceed to book immediately
            logger.info(f"[Agent] Confirm→save — collected: {self._collected}")
            result = self._execute_booking(result)
        elif action == "end":
            result["action"] = "end"
        elif action == "confirm":
            # Still confirming but missing info — already handled by gatekeeper above
            pass

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
            (12, 0): "12:00 PM",
            (14, 0): "02:00 PM", (14, 30): "02:30 PM",
            (15, 0): "03:00 PM", (15, 30): "03:30 PM",
            (16, 0): "04:00 PM", (16, 30): "04:30 PM",
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
            if d in ("tomorrow", "next day"):
                self._collected["appointment_date"] = (today + timedelta(days=1)).isoformat()
            elif d in ("day after tomorrow", "day after"):
                self._collected["appointment_date"] = (today + timedelta(days=2)).isoformat()
            # Already YYYY-MM-DD — leave as-is

    # ── Internal helpers ───────────────────────────────────────────────────────

    def get_transcript(self) -> str:
        """Converts self._history into a clean text transcript."""
        lines = []
        for msg in self._history:
            if self.language == "ur":
                role = "مریض" if msg["role"] == "user" else "سارہ"
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
            
            # Specific validation for Phone
            if field == "patient_phone" and val:
                clean = "".join(filter(str.isdigit, val))
                if not (7 <= len(clean) <= 15):
                    missing.append("a valid phone number (7-15 digits)")
        return missing

    def _build_context_injection(self) -> str:
        """
        Inject live DB availability and collected state into the system prompt.
        - If doctor + date are both known → show exact free slots for that combo.
        - If only department is known → show next available across all doctors.
        - If requested slot is already booked → warn the LLM explicitly.
        """
        from datetime import date as _date
        lines = []

        if self._collected:
            lines.append(f"CURRENT_STATE: {json.dumps(self._collected)}")

        missing = self._get_missing_requirements()
        if missing:
            lines.append(f"MISSING_FOLLOWING_INFO: {', '.join(missing)}")
            if self.language == "ur":
                lines.append("STRICT RULE: آگے بڑھنے سے پہلے 'نام' اور 'فون نمبر' حاصل کرنا لازمی ہے۔")
            else:
                lines.append("STRICT RULE: You MUST collect Name and Phone before Confirm/Save.")
        else:
            if self.language == "ur":
                lines.append("ALL_INFO_COLLECTED: تمام معلومات مل گئی ہیں۔ تصدیق کی طرف بڑھیں۔")
            else:
                lines.append("ALL_INFO_COLLECTED: All info collected — proceed to Confirmation.")

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
                speech = f"{settings.hospital_name} میں کال کرنے کا شکریہ۔ میں آپ کی کیا مدد کر سکتی ہوں؟"
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
        ctx = self._build_context_injection()
        if ctx:
            system += f"\n\n--- LIVE CONTEXT ---\n{ctx}"

        messages = [{"role": "system", "content": system}]
        messages += self._history
        messages.append({"role": "user", "content": user_input})

        from openai import AsyncOpenAI

        # ── Build provider list in priority order ──────────────────────────
        providers = []

        # 1. Gemini (primary — best Urdu quality, free via AI Studio)
        if settings.gemini_api_key:
            providers.append((
                AsyncOpenAI(
                    api_key=settings.gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                settings.gemini_model,
                "Gemini",
                False,  # Gemini does not support json_object format
            ))

        # 2. Groq (secondary — very fast, free tier)
        if settings.groq_api_key:
            providers.append((
                AsyncOpenAI(
                    api_key=settings.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                ),
                settings.groq_model,
                "Groq",
                True,
            ))

        # 3. HuggingFace (tertiary cloud fallback)
        if settings.hugging_face_token:
            providers.append((
                AsyncOpenAI(
                    api_key=settings.hugging_face_token,
                    base_url="https://router.huggingface.co/hf-inference/v1/",
                ),
                settings.huggingface_model,
                "HuggingFace",
                False,  # Qwen does not support json_object format
            ))

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

        # ── 4. Ollama (local fallback) ─────────────────────────────────────
        logger.info(f"[LLM] Trying Ollama → {settings.ollama_model}")
        try:
            from ollama import AsyncClient
            ollama_options = {"temperature": 0.2, "num_predict": 350, "num_ctx": 1024}
            if settings.ollama_num_gpu >= 0:
                ollama_options["num_gpu"] = settings.ollama_num_gpu
            async for chunk in await AsyncClient(host=settings.ollama_host).chat(
                model=settings.ollama_model,
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
            logger.error(f"[LLM] Ollama failed: {e}.")

        # ── 5. All providers failed ────────────────────────────────────────
        logger.error("[LLM] All providers failed.")
        if self.language == "ur":
            speech = (
                f"{settings.hospital_name} میں کال کرنے کا شکریہ۔ "
                f"میں سارہ بات کر رہی ہوں۔ ابھی ایک تکنیکی مسئلہ ہے، "
                f"براہ کرم تھوڑا انتظار کریں۔"
            )
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
                    f"بالکل! آپ کا اپائنٹمنٹ کنفرم ہو گیا ہے۔ "
                    f"{appt.patient_name} صاحب، {appt.doctor_name} کے ساتھ "
                    f"{appt.appointment_date.strftime('%d %B')} کو "
                    f"{appt.appointment_time} پر ملاقات بک ہو گئی ہے۔ "
                    f"بکنگ نمبر #{appt.id} ہے۔ اللہ حافظ!"
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
            result["action"] = "end"
            result["data"]["appointment_id"] = appt.id

        except ValueError as e:
            error_msg = str(e)
            logger.warning(f"[Booking] Failed: {error_msg}")

            if "phone number" in error_msg.lower():
                if self.language == "ur":
                    result["speech"] = "فون نمبر درست نہیں لگتا۔ کیا آپ دوبارہ نمبر بتا سکتے ہیں؟"
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
                    slots_str = ", ".join(free_slots[:4])
                    if self.language == "ur":
                        result["speech"] = (
                            f"معذرت، {info.get('appointment_time', 'یہ وقت')} {doctor} کے لیے دستیاب نہیں ہے۔ "
                            f"دستیاب اوقات یہ ہیں: {slots_str}۔ کون سا وقت ٹھیک رہے گا؟"
                        )
                    else:
                        result["speech"] = (
                            f"Sorry, {info.get('appointment_time', 'that slot')} isn't available for {doctor}. "
                            f"Available times are: {slots_str}. Which works for you?"
                        )
                else:
                    # No slots at all for this doctor on this date — suggest other doctors
                    alternatives = self.booking.get_next_available(dept, days_ahead=5)
                    self._collected.pop("appointment_date", None)
                    if alternatives:
                        alt_lines = [
                            f"{a['doctor']} on {a['date']} at {a['slots'][0]}"
                            for a in alternatives[:3]
                            if a["doctor"] != doctor
                        ]
                        alt_str = "; ".join(alt_lines) if alt_lines else "other times"
                        if self.language == "ur":
                            result["speech"] = (
                                f"معذرت، {doctor} اس تاریخ کو دستیاب نہیں ہیں۔ "
                                f"آپ یہ آپشنز دیکھ سکتے ہیں: {alt_str}۔ کیا کوئی اور ڈاکٹر یا تاریخ ٹھیک رہے گی؟"
                            )
                        else:
                            result["speech"] = (
                                f"Sorry, {doctor} has no availability on that date. "
                                f"Other options: {alt_str}. Would any of those work?"
                            )
                    else:
                        if self.language == "ur":
                            result["speech"] = f"معذرت، {doctor} اس وقت دستیاب نہیں ہیں۔ کیا کوئی اور ڈاکٹر ٹھیک رہے گا؟"
                        else:
                            result["speech"] = f"Sorry, {doctor} isn't available then. Would you like to try a different doctor?"
            else:
                if self.language == "ur":
                    result["speech"] = f"معذرت، بکنگ مکمل نہیں ہو سکی۔ کیا ہم دوبارہ کوشش کریں؟"
                else:
                    result["speech"] = f"I'm sorry, I couldn't complete the booking. Shall we try again?"
            result["action"] = "ask"

        except KeyError as e:
            missing_field = str(e).strip("'")
            logger.warning(f"[Booking] Missing field: {missing_field}")
            if self.language == "ur":
                result["speech"] = f"بکنگ کے لیے {missing_field.replace('_', ' ')} درکار ہے۔ کیا آپ یہ بتا سکتے ہیں؟"
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
