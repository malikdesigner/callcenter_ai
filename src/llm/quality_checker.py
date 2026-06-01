"""
Response quality checker + Claude fallback.

Tiered recovery strategy (cheapest-first):
  1. Qwen3 response arrives
  2. QualityChecker.check() evaluates it
  3a. reason is CANNED-ONLY (echo/loop/too_long/repetition)
      → use canned response immediately. 0 Claude calls.
  3b. reason is NEEDS-MODEL (empty/wrong_script/hallucination)
      → retry Qwen3 once with a correction prompt
      → if still bad AND Claude limit not reached → call Claude
      → if Claude unavailable or limit reached → canned response

Target: < 3 Claude calls per session under normal operation.
Claude is intentionally off by default — only activates if ANTHROPIC_API_KEY is set.
"""

import re
import time
from datetime import datetime
from loguru import logger
from config.settings import settings

# Claude Haiku pricing (per token)
_HAIKU_COST_IN  = 0.80  / 1_000_000   # $0.80 per million input tokens
_HAIKU_COST_OUT = 4.00  / 1_000_000   # $4.00 per million output tokens

def _calc_cost(tokens_in: int, tokens_out: int) -> float:
    return round(tokens_in * _HAIKU_COST_IN + tokens_out * _HAIKU_COST_OUT, 6)


# ── Known hallucination markers ───────────────────────────────────────────────
_HALLUCINATION_WORDS = {
    "velvet", "segments", "algorithm", "database", "computer",
    "variable", "parameter", "function", "سیگنٹز", "کمپیوٹر",
}

# Failure reasons that a canned response handles perfectly well.
# These do NOT justify a Claude API call — the issue is model drift,
# not a genuinely hard question that requires Claude's reasoning.
_CANNED_ONLY = {"echo", "loop", "too_long", "repetition"}

# Failure reasons where the model fundamentally produced unusable output.
# These MAY justify a Claude call after a Qwen3 retry fails.
_MODEL_FAILED = {"empty", "wrong_script", "hallucination"}


class QualityChecker:
    """Stateful per-session quality checker."""

    def __init__(self, language: str = "ur"):
        self.language = language
        self._last_response: str = ""

    def check(self, response: str, user_input: str) -> tuple[float, str]:
        """
        Returns (score: 0.0–1.0, reason: str).

        score = 1.0  → response is fine, use it
        reason in _CANNED_ONLY  → use canned, skip Claude
        reason in _MODEL_FAILED → retry Qwen3, then Claude if needed
        """
        text = response.strip()

        # 1. Empty
        if not text or len(text) < 3:
            return 0.0, "empty"

        words = text.split()

        # 2. Too long — model rambling (canned handles it)
        if len(words) > 22:
            logger.warning(f"[Quality] too_long ({len(words)}w): '{text[:60]}'")
            return 0.1, "too_long"

        # 3. Echo — response overlaps heavily with user's own words
        user_words = set(user_input.strip().split())
        sara_words = set(words)
        if sara_words:
            echo_ratio = len(user_words & sara_words) / len(sara_words)
            if echo_ratio >= 0.5:
                logger.warning(f"[Quality] echo ({echo_ratio:.0%}): '{text[:60]}'")
                return 0.1, "echo"

        # 4. Loop — identical to previous turn
        if text == self._last_response:
            logger.warning(f"[Quality] loop: '{text[:60]}'")
            return 0.2, "loop"

        # 5. Wrong script — Urdu mode but model replied in English/Latin
        if self.language == "ur":
            non_space = [c for c in text if not c.isspace()]
            ur_chars = sum(1 for c in non_space if "؀" <= c <= "ۿ")
            if non_space and ur_chars / len(non_space) < 0.25:
                logger.warning(f"[Quality] wrong_script: '{text[:60]}'")
                return 0.1, "wrong_script"

        # 6. Hallucination markers
        text_lower = text.lower()
        if any(w in text_lower for w in _HALLUCINATION_WORDS):
            logger.warning(f"[Quality] hallucination: '{text[:60]}'")
            return 0.0, "hallucination"

        # 7. Repetitive trigrams inside response
        if len(words) >= 6:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
            if len(trigrams) != len(set(trigrams)):
                logger.warning(f"[Quality] repetition: '{text[:60]}'")
                return 0.1, "repetition"

        self._last_response = text
        return 1.0, "ok"

    def needs_claude(self, reason: str) -> bool:
        """True only if Claude is warranted (not just a canned response)."""
        return reason in _MODEL_FAILED


class ClaudeFallback:
    """
    Claude API fallback — last resort only.

    Budget enforcement (hard stops, not informational):
      1. Monthly spend cap  — reads/writes DB table claude_usage.
                              Blocks calls when spend >= claude_monthly_budget ($5 default).
      2. Session call cap   — max claude_session_limit calls per session (default 3).
      3. 60-second cooldown — prevents back-to-back Claude calls.
      4. API key required   — disabled completely when ANTHROPIC_API_KEY is not set.

    All four checks run BEFORE each Claude call.
    Actual cost is recorded AFTER each call so the budget stays accurate
    even if the process restarts mid-month.
    """

    def __init__(self):
        self._session_calls: int  = 0
        self._session_tokens: int = 0
        self._last_call_at: float = 0.0

    # ── Synchronous guard (fast checks only) ──────────────────────────────────

    @property
    def _fast_available(self) -> bool:
        """Cheap synchronous checks that don't need the DB."""
        if not settings.anthropic_api_key:
            return False
        if self._session_calls >= settings.claude_session_limit:
            logger.info(
                f"[Claude] Session cap {self._session_calls}/"
                f"{settings.claude_session_limit} — using canned"
            )
            return False
        since_last = time.monotonic() - self._last_call_at
        if self._last_call_at > 0 and since_last < 60:
            logger.info(f"[Claude] Cooldown {60 - since_last:.0f}s remaining — using canned")
            return False
        return True

    # ── Async monthly budget check (DB) ───────────────────────────────────────

    async def _monthly_cost(self) -> float:
        """Return total Claude spend this calendar month from DB."""
        month = datetime.now().strftime("%Y-%m")
        try:
            from src.db.database import AsyncSessionLocal
            from src.db.models import ClaudeUsage
            from sqlalchemy import select
            async with AsyncSessionLocal() as session:
                row = (await session.execute(
                    select(ClaudeUsage).where(ClaudeUsage.month == month)
                )).scalars().first()
                return row.total_cost_usd if row else 0.0
        except Exception as e:
            logger.warning(f"[Claude] Budget check failed: {e}")
            return 0.0

    async def _record_spend(self, tokens_in: int, tokens_out: int, cost: float):
        """Persist this call's cost to DB so budget survives restarts."""
        month = datetime.now().strftime("%Y-%m")
        try:
            from src.db.database import AsyncSessionLocal
            from src.db.models import ClaudeUsage
            from sqlalchemy import select
            async with AsyncSessionLocal() as session:
                row = (await session.execute(
                    select(ClaudeUsage).where(ClaudeUsage.month == month)
                )).scalars().first()
                if row:
                    row.total_calls    += 1
                    row.total_tokens   += tokens_in + tokens_out
                    row.total_cost_usd += cost
                    row.updated_at      = datetime.now()
                else:
                    from src.db.models import ClaudeUsage as CU
                    session.add(CU(
                        month=month,
                        total_calls=1,
                        total_tokens=tokens_in + tokens_out,
                        total_cost_usd=cost,
                    ))
                await session.commit()
        except Exception as e:
            logger.warning(f"[Claude] Spend recording failed: {e}")

    # ── Main entry point ───────────────────────────────────────────────────────

    async def generate(
        self,
        history: list[dict],
        system_prompt: str,
        user_text: str,
        language: str,
    ) -> tuple[str, bool]:
        """
        Generate via Claude. Returns (text, used_claude).

        Checks all 4 guards before making the API call.
        Records actual token cost to DB after a successful call.
        """
        # Guard 1–3: fast synchronous checks
        if not self._fast_available:
            return "", False

        # Guard 4: monthly budget (DB lookup)
        current_spend = await self._monthly_cost()
        if current_spend >= settings.claude_monthly_budget:
            logger.warning(
                f"[Claude] Monthly budget exhausted "
                f"(${current_spend:.4f} >= ${settings.claude_monthly_budget}) — using canned"
            )
            return "", False

        # Show how much budget remains before this call
        remaining = settings.claude_monthly_budget - current_spend
        logger.info(f"[Claude] Budget remaining: ${remaining:.4f}")

        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in history
                if m.get("role") in ("user", "assistant") and m.get("content")
            ][-6:]                          # last 6 turns only — fewer tokens
            messages.append({"role": "user", "content": user_text})

            t0 = time.monotonic()
            resp = await client.messages.create(
                model=settings.claude_fallback_model,
                max_tokens=150,             # short = cheap
                system=system_prompt,
                messages=messages,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)

            tokens_in  = resp.usage.input_tokens
            tokens_out = resp.usage.output_tokens
            cost       = _calc_cost(tokens_in, tokens_out)

            text = resp.content[0].text.strip() if resp.content else ""
            self._session_calls  += 1
            self._session_tokens += tokens_in + tokens_out
            self._last_call_at    = time.monotonic()

            logger.info(
                f"[Claude] #{self._session_calls} | "
                f"{tokens_in}+{tokens_out} tok | "
                f"${cost:.5f} | {latency_ms}ms | "
                f"monthly=${current_spend + cost:.4f}/${settings.claude_monthly_budget} | "
                f"'{text[:60]}'"
            )

            await self._record_spend(tokens_in, tokens_out, cost)
            return text, True

        except Exception as e:
            logger.error(f"[Claude] API error: {e}")
            return "", False

    @property
    def session_stats(self) -> dict:
        return {
            "claude_calls":  self._session_calls,
            "claude_tokens": self._session_tokens,
            "session_cost":  _calc_cost(
                self._session_tokens // 2,   # rough split
                self._session_tokens // 2,
            ),
        }
