"""
Hospital AI Agent v2

Upgrades over v1:
  • Qwen3:14b — correct Urdu grammar, no hallucinated responses
  • Tool calling — LLM calls real DB tools, never invents slots or doctors
  • Async PostgreSQL via src.db
  • Knowledge base from YAML, not hardcoded
  • Backwards-compatible interface with call_handler.py

Streaming contract (same as original agent.py):
  process_turn_stream() yields:
    str  → sentence/phrase fragment ready for TTS
    dict → final result {"speech", "action", ...}
"""

import json
import re
import yaml
from datetime import date
from pathlib import Path
from typing import AsyncGenerator, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from src.llm.tools import TOOL_DEFINITIONS, execute_tool
from src.llm.quality_checker import QualityChecker, ClaudeFallback

# ── Canned responses: used when model generates garbage ───────────────────────
# Keyed by language → dialogue state → safe fallback phrase.
_CANNED: dict[str, dict[str, str]] = {
    "ur": {
        "GREETING":     "السلام علیکم ... میں سارہ ہوں ... آپکا نام",
        "ASK_NAME":     "آپکا نام بتائیں",
        "ASK_PHONE":    "موبایل نمبر بتائیں",
        "ASK_SYMPTOM":  "کیا تکلیف ہے",
        "ASK_DOCTOR":   "کس ڈاکڑ کو دکھانا ہے",
        "ASK_DATE":     "کونسی تاریخ ٹھیک ہے",
        "ASK_TIME":     "کیا وقت ٹھیک رہے گا",
        "CONFIRMATION": "سب ٹھیک ہے ... کنفرم کروں",
        "END":          "شکریہ ... اللہ حافظ",
        "default":      "ذرا بتائیں ... میں مدد کروں",
    },
    "ro": {
        "GREETING":     "Assalam o alaikum ... main Sara hoon ... aapka naam",
        "ASK_NAME":     "Aapka naam bataein",
        "ASK_PHONE":    "Mobile number bataein",
        "ASK_SYMPTOM":  "Kya takleef hai",
        "ASK_DOCTOR":   "Kis doctor ko dikhana hai",
        "ASK_DATE":     "Konsi date theek hai",
        "ASK_TIME":     "Kya waqt theek rahega",
        "CONFIRMATION": "Sab theek hai ... confirm karoon",
        "default":      "Ji ... zaroor bataein",
    },
    "en": {
        "GREETING":     "Hello, I'm Sara. May I have your name?",
        "ASK_NAME":     "Could you tell me your name?",
        "ASK_PHONE":    "What's your mobile number?",
        "ASK_SYMPTOM":  "What's bothering you?",
        "ASK_DOCTOR":   "Which doctor would you like to see?",
        "ASK_DATE":     "What date works for you?",
        "ASK_TIME":     "What time works for you?",
        "CONFIRMATION": "Shall I go ahead and confirm the booking?",
        "default":      "Could you say that again?",
    },
}


# ── Load hospital info once at module level ────────────────────────────────────
_HOSPITAL_INFO = yaml.safe_load(Path("knowledge_base/hospital.yaml").read_text(encoding="utf-8"))


# ── System prompts ─────────────────────────────────────────────────────────────

def _build_system(lang: str, policy_state: str = "", nlg_constraint: str = "") -> str:
    today = date.today().strftime("%A, %d %B %Y")

    if lang == "ur":
        base = f"""آپ سارہ ہیں — {_HOSPITAL_INFO['name']} کی AI ریسپشنسٹ۔

آپکا کام: اپوائنٹمنٹ بکنگ، کینسل، ری شیڈول، ڈاکٹر تجویز، ہسپتال معلومات۔

━━━ بات چیت کا اصول ━━━
• مختصر اور واضح — ایک جواب میں ایک کام
• ایک وقت میں ایک سوال
• پہلے تصدیق، پھر بکنگ
• ٹول استعمال کریں — خود سے وقت یا ڈاکٹر نہ بنائیں

━━━ درست اردو گرامر ━━━
✓ "السلام علیکم ... میں سارہ ہوں ... آپکا نام"
✓ "اچھا جی ... موبایل نمبر بتائیں"
✓ "آپکو کیا تکلیف ہے"
✓ "ڈاکڑ کامران منگل کو دستیاب ہیں ... وقت لوں"
✓ "سب کنفرم ہے ... بکنگ کروں"

✗ "جیسے جیسے آپ کی تکلیف بتائیں گئیں" — غلط
✗ "براہ کرم اپنی درخواست واضح کریں" — بہت رسمی
✗ "ڈاکٹر نے کوئی مسئلہ نہیں" — بے معنی

━━━ پابندیاں ━━━
• جواب ≤ 12 الفاظ
• صرف اردو رسم الخط
• وقفہ صرف ... سے
• ممنوع: براہ کرم | معافی | جناب | لہذا | اپوائنٹمنٹ
• ہنگامی تکلیف → ایمرجنسی بھیجیں

متبادل فلر (ہر بار بدلتے رہیں): جی سر | اچھا جی | ٹھیک ہے | سمجھ گئی | بالکل جی

آج: {today}"""

    elif lang == "ro":
        base = f"""Aap Sara hain — {_HOSPITAL_INFO['name']} ki AI receptionist.

Kaam: appointment booking, cancel, reschedule, doctor suggest, hospital info.

━━━ Baat cheet ka usool ━━━
• Mukhtasar aur wazeh — ek jawab mein ek kaam
• Ek waqt mein ek sawal
• Pehle tasdeeq, phir booking
• Tool use karein — khud se waqt ya doctor mat banaein

━━━ Output rules ━━━
• Jawab ≤ 12 alfaaz
• Sirf Roman Urdu (Latin)
• Waqfa sirf ... se
• Mamnoo: meherbani | maafi | janab | lihaza | appointment

Filler (har baar badlain): ji sir | acha ji | theek hai | samajh gayi | bilkul ji

Aaj: {today}"""

    else:
        base = f"""You are Sara — AI receptionist at {_HOSPITAL_INFO['name']}.

Role: Book, cancel, reschedule appointments. Suggest doctors. Answer hospital FAQs.

Rules:
• One question per turn. Acknowledge before asking next.
• Always confirm details before booking.
• Use tools — never invent availability or doctor names.
• Max 20 words per response.
• For emergencies → Emergency Department.

Today: {today} | Emergency: {_HOSPITAL_INFO['emergency']}"""

    # Inject dialogue policy constraints if provided
    if policy_state or nlg_constraint:
        extras = []
        if policy_state and policy_state not in ("GREETING", ""):
            extras.append(f"CURRENT_BOOKING_STATE: {policy_state}")
        if nlg_constraint:
            extras.append(f"INSTRUCTION: {nlg_constraint}")
        if extras:
            base += "\n\n" + "\n".join(extras)

    return base


# ── Agent ──────────────────────────────────────────────────────────────────────

class HospitalAgentV2:
    """
    One instance per call session.
    Backwards-compatible with the call_handler.py interface.
    """

    def __init__(self, language: str = "ur", db_session: AsyncSession = None):
        self.language = language
        self._db: Optional[AsyncSession] = db_session

        # Compat attributes expected by call_handler / server
        self._bilingual_mode: bool = (language == "bi")
        self._language_chosen: bool = not self._bilingual_mode
        self._flow_state: str = "greeting"
        self._collected: dict = {}   # kept in sync by call_handler via data_extractor

        self._history: list[dict] = []
        self._quality   = QualityChecker(language=language)
        self._fallback  = ClaudeFallback()
        self._turn_count: int = 0

    def set_db(self, session: AsyncSession):
        self._db = session

    def reset(self):
        self._history = []
        self._collected = {}
        self._flow_state = "greeting"
        if self._bilingual_mode:
            self.language = "en"
            self._language_chosen = False

    # ── Public API (matches call_handler expectations) ─────────────────────────

    async def get_greeting(self) -> str:
        prompt = (
            "مریض نے ابھی کال کی — مختصر، گرم جوشی سے سلام کریں اور نام پوچھیں۔"
            if self.language == "ur"
            else (
                "Naye patient ne call ki — Roman Urdu mein mukhtasar salam karein aur naam poochein."
                if self.language == "ro"
                else "A patient just called. Greet warmly and ask for their name."
            )
        )
        result = await self._complete_simple(prompt)
        fallbacks = {
            "ur": f"السلام علیکم ... میں سارہ ہوں ... آپکا نام",
            "ro": f"Assalam o alaikum ... main Sara hoon ... aapka naam",
            "en": f"Hello! I'm Sara from {_HOSPITAL_INFO['name']}. May I have your name?",
        }
        return result or fallbacks.get(self.language, fallbacks["en"])

    async def process_turn_stream(
        self,
        user_text: str,
        audio_language: str = None,
        policy_state: str = "",
        nlg_constraint: str = "",
    ) -> AsyncGenerator:
        """
        Yields speech fragments (str) then a final result dict.
        Handles bilingual redirect, tool calls, and direct responses.
        """
        # ── Bilingual: first turn picks language ──────────────────────────────
        if self._bilingual_mode and not self._language_chosen:
            from src.llm.intent import detect_language_preference
            pref = detect_language_preference(user_text)
            if pref is None and audio_language == "ur":
                pref = "ur"
            if pref == "ur":
                self.language = "ur"
                speech = "بالکل جی ... ابھی اردو سروس سے جوڑتی ہوں"
                yield speech
                yield {"speech": speech, "action": "redirect", "data": {"port": 8001}}
                return
            elif pref == "en":
                speech = "Perfect! Connecting to English service now."
                yield speech
                yield {"speech": speech, "action": "redirect", "data": {"port": 8000}}
                return
            else:
                speech = "سمجھی نئیں ... اردو یا English کہیں" if not audio_language else "Please say Urdu or English."
                yield speech
                self._history.append({"role": "user",      "content": user_text})
                self._history.append({"role": "assistant",  "content": speech})
                yield {"speech": speech, "action": "ask"}
                return

        # ── Normal turn ───────────────────────────────────────────────────────
        self._history.append({"role": "user", "content": user_text})
        system = _build_system(self.language, policy_state, nlg_constraint)
        messages = [{"role": "system", "content": system}] + self._history[-20:]

        from ollama import AsyncClient
        client = AsyncClient(host=settings.ollama_host)
        opts = {"temperature": settings.ollama_temperature, "num_predict": 512, "num_ctx": 4096}
        if settings.ollama_num_gpu >= 0:
            opts["num_gpu"] = settings.ollama_num_gpu

        logger.info(f"[AgentV2] {settings.ollama_model} | {self.language} | '{user_text[:50]}'")

        # ── First LLM call (non-streaming: needed to detect tool calls) ───────
        try:
            resp = await client.chat(
                model=settings.ollama_model,
                messages=messages,
                tools=TOOL_DEFINITIONS if self._db else None,
                options=opts,
            )
        except Exception as e:
            logger.error(f"[AgentV2] LLM error: {e}")
            speech = self._fallback_speech()
            yield speech
            yield {"speech": speech, "action": "ask"}
            return

        # ollama 0.3.x returns a plain dict; newer versions return pydantic objects.
        # Normalise to dict so the rest of this code works with both.
        msg: dict = resp if isinstance(resp, dict) else resp.__dict__
        msg_body: dict = msg.get("message", msg)   # {"role":…, "content":…, "tool_calls":[…]}

        # ── Handle tool calls ─────────────────────────────────────────────────
        tool_calls = msg_body.get("tool_calls") or []
        if tool_calls:
            self._history.append({
                "role":       "assistant",
                "content":    msg_body.get("content", ""),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                # Support both dict form {"function": {"name":…, "arguments":{…}}}
                # and object form (tc.function.name)
                if isinstance(tc, dict):
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                else:
                    name = tc.function.name
                    args = tc.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                logger.info(f"[Tool] {name}({list(args.keys())})")
                result_json = await execute_tool(name, args, self._db)
                logger.debug(f"[Tool] → {result_json[:100]}")
                self._history.append({"role": "tool", "name": name, "content": result_json})

            # Second call: stream spoken response using tool data
            messages2 = [{"role": "system", "content": system}] + self._history[-22:]
            speech_parts = []
            async for phrase in self._stream_response(client, messages2, opts):
                speech_parts.append(phrase)
            speech = " ".join(speech_parts)

            speech, used_fallback = await self._ensure_quality(speech, user_text, policy_state, system)
            for phrase in self._emit_phrases(speech):
                yield phrase
            self._history.append({"role": "assistant", "content": speech})
            if len(self._history) > 40:
                self._history = self._history[-40:]
            self._turn_count += 1
            yield {"speech": speech, "action": self._infer_action(speech),
                   "used_fallback": used_fallback}
            return

        # ── Direct text response — quality-check then emit ───────────────────
        content = self._extract_speech(msg_body.get("content", ""))
        content, used_fallback = await self._ensure_quality(content, user_text, policy_state, system)

        speech_parts = []
        for phrase in self._emit_phrases(content):
            speech_parts.append(phrase)
            yield phrase
        speech = " ".join(speech_parts) or content
        self._history.append({"role": "assistant", "content": speech})
        if len(self._history) > 40:
            self._history = self._history[-40:]
        self._turn_count += 1
        yield {"speech": speech, "action": self._infer_action(speech),
               "used_fallback": used_fallback}

    # ── Streaming helpers ─────────────────────────────────────────────────────

    async def _stream_response(self, client, messages: list, opts: dict):
        """Async generator: yields speech phrases (split on '...')."""
        try:
            stream = await client.chat(
                model=settings.ollama_model,
                messages=messages,
                options=opts,
                stream=True,
            )
            buffer = ""
            full_text = ""
            async for chunk in stream:
                c = chunk if isinstance(chunk, dict) else chunk.__dict__
                delta = c.get("message", c).get("content", "") or ""
                if not delta:
                    continue
                full_text += delta
                buffer    += delta

                while "..." in buffer:
                    idx = buffer.index("...") + 3
                    phrase = buffer[:idx].strip()
                    buffer = buffer[idx:].strip()
                    if phrase:
                        yield phrase

            # Remainder
            remainder = self._extract_speech(buffer or full_text).strip()
            if remainder:
                yield remainder

        except Exception as e:
            logger.error(f"[AgentV2] stream error: {e}")
            yield self._fallback_speech()

    def _emit_phrases(self, text: str):
        """Sync generator: yields speech phrases split on '...'."""
        text = text.strip()
        parts = [p.strip() for p in text.split("...") if p.strip()]
        if not parts:
            if text:
                yield text
            return
        for i, part in enumerate(parts):
            phrase = part if i == len(parts) - 1 else part + " ..."
            if phrase.strip():
                yield phrase

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _complete_simple(self, prompt: str) -> str:
        """Non-streaming single-turn completion."""
        from ollama import AsyncClient
        client = AsyncClient(host=settings.ollama_host)
        system = _build_system(self.language)
        try:
            resp = await client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": settings.ollama_temperature, "num_predict": 128},
            )
            r = resp if isinstance(resp, dict) else resp.__dict__
            content = r.get("message", r).get("content", "")
            return self._extract_speech(content)
        except Exception as e:
            logger.error(f"[AgentV2] _complete_simple: {e}")
            return ""

    def _extract_speech(self, text: str) -> str:
        """Strip Qwen3 thinking tags, JSON wrappers, and markdown."""
        text = text.strip()
        # Remove Qwen3 <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Try JSON speech field
        if text.startswith("{"):
            try:
                data = json.loads(text)
                return str(data.get("speech") or data.get("response") or text)
            except Exception:
                pass
        m = re.search(r'"speech"\s*:\s*"(.*?)"', text, re.DOTALL)
        if m:
            try:
                return json.loads(f'"{m.group(1)}"')
            except Exception:
                return m.group(1)
        return text

    async def _ensure_quality(
        self, response: str, user_text: str, policy_state: str, system: str
    ) -> tuple[str, bool]:
        """
        Tiered quality recovery — cheapest path first.

        Tier 1 (free):   response is good → use it
        Tier 2 (free):   canned-only failure (echo/loop/too_long/repetition)
                         → canned response, skip Claude entirely
        Tier 3 (free):   model-failed (empty/wrong_script/hallucination)
                         → retry Qwen3 once with a terse correction prompt
        Tier 4 (paid):   Qwen3 retry also failed → Claude (if available + limit ok)
        Tier 5 (free):   Claude unavailable/capped → canned response

        Returns (final_text, used_claude)
        """
        score, reason = self._quality.check(response, user_text)

        # Tier 1: response is fine
        if score >= 0.5:
            return response, False

        logger.warning(
            f"[Quality] score={score:.1f} reason={reason} turn={self._turn_count} "
            f"| '{response[:60]}'"
        )

        # Tier 2: canned-only failure — no Claude needed
        if not self._quality.needs_claude(reason):
            logger.info(f"[Quality] Using canned (reason={reason}, Claude not needed)")
            return self._canned(policy_state), False

        # Tier 3: retry Qwen3 once with an explicit correction prompt
        retry_response = await self._retry_qwen(user_text, system)
        if retry_response:
            retry_score, retry_reason = self._quality.check(retry_response, user_text)
            if retry_score >= 0.5:
                logger.info(f"[Quality] Qwen3 retry succeeded (was {reason})")
                return retry_response, False

        # Tier 4: Claude fallback (only if key set + session limit not reached + cooldown ok)
        if self._fallback.available:
            claude_text, used = await self._fallback.generate(
                history=self._history,
                system_prompt=system,
                user_text=user_text,
                language=self.language,
            )
            if used and claude_text.strip():
                c_score, c_reason = self._quality.check(claude_text, user_text)
                if c_score >= 0.5:
                    return claude_text, True
                logger.warning(f"[Quality] Claude response also failed ({c_reason})")

        # Tier 5: canned — nothing else worked or Claude unavailable
        return self._canned(policy_state), False

    async def _retry_qwen(self, user_text: str, system: str) -> str:
        """
        One-shot Qwen3 retry with a terse correction prompt.
        Asks for a short direct response instead of the original failed one.
        Only used when the model produced empty/wrong_script/hallucination output.
        """
        from ollama import AsyncClient
        client = AsyncClient(host=settings.ollama_host)
        correction = {
            "ur":  "پچھلا جواب غلط تھا۔ مختصر اردو میں جواب دیں۔ 8 الفاظ سے کم۔",
            "ro":  "Pichla jawab galat tha. Mukhtasar Roman Urdu mein jawab dein. 8 alfaaz se kam.",
            "en":  "Previous response was incorrect. Give a short direct answer under 10 words.",
        }.get(self.language, "Short answer only.")
        try:
            resp = await client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system + f"\n\nNOTE: {correction}"},
                    {"role": "user",   "content": user_text},
                ],
                options={"temperature": 0.1, "num_predict": 80},
            )
            r = resp if isinstance(resp, dict) else resp.__dict__
            return self._extract_speech(r.get("message", r).get("content", ""))
        except Exception as e:
            logger.warning(f"[Quality] Qwen3 retry failed: {e}")
            return ""

    def _is_garbage(self, text: str) -> bool:
        """
        Detect LLM hallucinations before they reach TTS.

        Signs of garbage in Urdu mode:
          • Response is too long (model ignored the ≤12 word rule)
          • Repetitive trigrams (classic hallucination loop)
          • Unexpected English words mid-Urdu (mixed-script hallucination)
          • Contains no Urdu script at all when language is "ur"
        """
        text = text.strip()
        if not text:
            return True

        words = text.split()

        # Too long — model is rambling
        if len(words) > 22:
            logger.warning(f"[Guard] Response too long ({len(words)} words): '{text[:60]}'")
            return True

        # Repetitive trigrams (loop hallucination)
        if len(words) >= 6:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
            if len(trigrams) != len(set(trigrams)):
                logger.warning(f"[Guard] Repetitive trigram detected: '{text[:60]}'")
                return True

        # Urdu mode: catch English words mixed into the response unexpectedly
        if self.language == "ur":
            urdu_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
            if urdu_chars < 4:
                logger.warning(f"[Guard] No Urdu script in UR-mode response: '{text[:60]}'")
                return True
            long_english = re.findall(r"\b[a-zA-Z]{5,}\b", text)
            _ok_words = {"doctor", "booking", "number", "phone"}
            unexpected = [w for w in long_english if w.lower() not in _ok_words]
            if len(unexpected) >= 2:
                logger.warning(f"[Guard] Mixed-script hallucination: {unexpected} in '{text[:60]}'")
                return True

        return False

    def _canned(self, policy_state: str = "") -> str:
        """Return the safe canned response for the current language and dialogue state."""
        lang_map = _CANNED.get(self.language, _CANNED["en"])
        state_key = policy_state.upper() if policy_state else "default"
        return lang_map.get(state_key) or lang_map["default"]

    def _infer_action(self, speech: str) -> str:
        """Guess action from speech content (booking done vs still collecting)."""
        confirm_ur = ["بکنگ نمبر", "کنفرم ہو گئی", "appointment confirmed", "booking id"]
        if any(w in speech.lower() for w in confirm_ur):
            return "save"
        bye_ur = ["اللہ حافظ", "allah hafiz", "goodbye", "شکریہ"]
        if any(w in speech.lower() for w in bye_ur) and len(speech.split()) < 8:
            return "end"
        return "ask"

    def _fallback_speech(self) -> str:
        return {
            "ur": "سوری ... ابھی تکنیکی مسئلہ ہے ... دوبارہ کوشش کریں",
            "ro": "Sori ... abhi technical masla hai ... dobara koshish karein",
        }.get(self.language, "I'm experiencing a technical issue. Please try again.")

    def get_transcript(self) -> str:
        lines = []
        labels = {
            "ur":  {"user": "مریض",   "assistant": "سارہ"},
            "ro":  {"user": "Mareez", "assistant": "Sara"},
        }.get(self.language, {"user": "Patient", "assistant": "Sara"})
        for msg in self._history:
            role = msg.get("role")
            if role in ("user", "assistant") and msg.get("content"):
                lines.append(f"{labels[role]}: {msg['content']}")
        return "\n".join(lines)
