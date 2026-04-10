"""
Hospital receptionist LLM agent powered by Ollama (local, no API key).
Manages multi-turn conversation and triggers appointment booking actions.
"""

import json
from datetime import date, datetime
from typing import Optional

import ollama
from loguru import logger

from config.settings import settings
from src.appointment.booking import (
    DOCTORS,
    BookingSystem,
    resolve_department,
)

# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    dept_list = ", ".join(d.title() for d in DOCTORS)
    today = date.today().strftime("%A, %B %d, %Y")

    return f"""You are {settings.receptionist_name}, a warm and professional AI receptionist at {settings.hospital_name}.

Today's date: {today}

Your responsibilities:
1. Greet the caller and identify their need (book / cancel / check appointment, or general info).
2. Collect information ONE question at a time — never ask multiple questions together.
3. Book, cancel, or look up appointments using the data provided to you.
4. Always confirm details back to the caller before finalising a booking.

Available departments: {dept_list}

Speaking rules:
- Keep responses SHORT (1-3 sentences) — this is a phone call.
- Be friendly but efficient.
- Never make up doctor names or slots — only use those provided in the context.
- If you don't know something, offer to transfer to a human staff member.

ALWAYS reply with a valid JSON object — no extra text outside the JSON:
{{
  "speech": "<exactly what you say to the caller>",
  "action": "<one of: none | book_appointment | cancel_appointment | end_call>",
  "data": {{
    "patient_name": "",
    "patient_phone": "",
    "department": "",
    "doctor_name": "",
    "appointment_date": "",   // YYYY-MM-DD
    "appointment_time": "",   // e.g. 09:00 AM
    "reason": "",
    "appointment_id": null
  }}
}}

Only set action to "book_appointment" when you have ALL of:
patient_name, patient_phone, department, doctor_name, appointment_date, appointment_time.

Only set action to "cancel_appointment" when you have appointment_id confirmed by the caller.

Set action to "end_call" only after you have finished helping the caller and said goodbye.
"""


# ── Agent class ────────────────────────────────────────────────────────────────

class HospitalAgent:
    def __init__(self):
        self.booking = BookingSystem()
        self._history: list = []
        self._collected: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Reset state for a new call."""
        self._history = []
        self._collected = {}

    def get_greeting(self) -> str:
        result = self._chat("A new caller has connected. Please greet them warmly.")
        return result.get("speech", f"Thank you for calling {settings.hospital_name}. This is {settings.receptionist_name}. How can I help you today?")

    def process_turn(self, user_text: str) -> dict:
        """
        Process one user turn.
        Returns dict with 'speech', 'action', 'data'.
        May execute a booking/cancellation as a side effect.
        """
        result = self._chat(user_text)

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

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_context_injection(self) -> str:
        """Inject current availability into the system prompt if department known."""
        lines = []

        if self._collected:
            lines.append(f"Collected so far: {json.dumps(self._collected)}")

        dept = self._collected.get("department", "")
        if dept:
            availability = self.booking.get_next_available(dept, days_ahead=5)
            if availability:
                av_lines = []
                for item in availability[:3]:
                    av_lines.append(
                        f"  {item['doctor']} | {item['date']} | "
                        + ", ".join(item["slots"][:3])
                    )
                lines.append("Available slots:\n" + "\n".join(av_lines))

        return "\n\n".join(lines)

    def _chat(self, user_input: str) -> dict:
        system = _build_system_prompt()
        ctx = self._build_context_injection()
        if ctx:
            system += f"\n\n--- LIVE CONTEXT ---\n{ctx}"

        messages = [{"role": "system", "content": system}]
        messages += self._history
        messages.append({"role": "user", "content": user_input})

        raw = ""
        try:
            response = ollama.chat(
                model=settings.ollama_model,
                messages=messages,
                format="json",
                options={"temperature": 0.2, "num_predict": 300},
            )
            raw = response["message"]["content"]
            result = json.loads(raw)

            # Ensure result is a flat dict with a "speech" string at the top level.
            # Some models wrap the response in an extra layer.
            if "speech" not in result:
                # Try one level deeper
                for v in result.values():
                    if isinstance(v, dict) and "speech" in v:
                        result = v
                        break
                else:
                    raise ValueError("No 'speech' key found in LLM response")

            # Guarantee speech is a plain string, not a nested object
            if not isinstance(result.get("speech"), str):
                result["speech"] = str(result.get("speech", ""))

        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[LLM] JSON parse failed, extracting speech from raw")
            # Try to pull just the speech value via regex
            import re
            m = re.search(r'"speech"\s*:\s*"([^"]+)"', raw)
            speech = m.group(1) if m else (
                f"Thank you for calling {settings.hospital_name}. How can I help you?"
            )
            result = {"speech": speech, "action": "none", "data": {}}

        except Exception as e:
            logger.error(f"[LLM] Ollama error: {e}")
            result = {
                "speech": "I'm sorry, I'm having a technical issue. Please hold.",
                "action": "none",
                "data": {},
            }

        # Update conversation history (keep last 20 turns to avoid token overflow)
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": json.dumps(result)})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        logger.debug(f"[LLM] action={result.get('action')} speech={result.get('speech','')[:80]}")
        return result

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
            )
            result["speech"] = (
                f"Your appointment is confirmed! "
                f"{appt.patient_name}, you're booked with {appt.doctor_name} "
                f"on {appt.appointment_date.strftime('%A, %B %d')} "
                f"at {appt.appointment_time}. "
                f"Your booking reference is #{appt.id}. "
                f"Is there anything else I can help you with?"
            )
            result["data"]["appointment_id"] = appt.id
        except ValueError as e:
            logger.warning(f"[Booking] Failed: {e}")
            result["speech"] = (
                f"I'm sorry, that slot is no longer available. "
                f"Let me suggest another time for you."
            )
            result["action"] = "none"
        except KeyError as e:
            logger.warning(f"[Booking] Missing field: {e}")
            result["speech"] = (
                f"I still need a few more details before I can complete the booking. "
                f"Could you please confirm {e}?"
            )
            result["action"] = "none"
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
