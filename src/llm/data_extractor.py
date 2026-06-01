"""
Light-weight heuristic data extractor.

Pulls patient_name and patient_phone from a transcript without any LLM call.
Used to keep DialoguePolicy.collected_data in sync so the state machine
correctly advances past ASK_NAME and ASK_PHONE.

Why this exists: HospitalAgentV2 uses tool-calling, not JSON slot-filling,
so it never writes to agent._collected. Without this extractor the policy
would always see an empty dict and loop on ASK_NAME forever.
"""

import re
from loguru import logger


def extract_patient_data(text: str, language: str = "ur") -> dict:
    """
    Extract {patient_name, patient_phone} from a single transcript.
    Returns only the fields that were confidently found.
    """
    result: dict[str, str] = {}

    # ── Phone number ─────────────────────────────────────────────────────────
    # Strip separators and look for a digit sequence of 9–13 chars
    digits_only = re.sub(r"[^\d]", "", text)
    if 9 <= len(digits_only) <= 13:
        result["patient_phone"] = digits_only
    else:
        # Might have spaces between groups: "920 420 520"
        m = re.search(r"\b[\d][\d\s\-]{7,15}[\d]\b", text)
        if m:
            d = re.sub(r"[^\d]", "", m.group())
            if 9 <= len(d) <= 13:
                result["patient_phone"] = d

    # ── Name — Urdu script ────────────────────────────────────────────────────
    if language == "ur":
        patterns = [
            r"میرا نام\s+(.{2,25}?)(?:\s+ہے|\s+میں|\s+کا|[،,]|$)",
            r"نام\s+(.{2,25}?)(?:\s+ہے|[،,]|$)",
            r"میں\s+(.{2,20}?)\s+ہوں",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                candidate = m.group(1).strip()
                words = candidate.split()
                # 1–3 words, no digits, not a common Urdu function word
                _SKIP = {"جی", "ہاں", "نئیں", "نہیں", "اچھا", "ٹھیک"}
                if (1 <= len(words) <= 3
                        and not any(c.isdigit() for c in candidate)
                        and candidate not in _SKIP):
                    result["patient_name"] = candidate
                    break

    # ── Name — Roman Urdu ─────────────────────────────────────────────────────
    elif language == "ro":
        patterns = [
            r"mera naam\s+(\w[\w\s]{1,20}?)(?:\s+hai|[,.]|$)",
            r"naam\s+(\w[\w\s]{1,20}?)(?:\s+hai|[,.]|$)",
            r"main\s+(\w[\w\s]{1,15}?)\s+hoon",
            r"i am\s+(\w[\w\s]{1,20}?)(?:[,.]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                candidate = m.group(1).strip().title()
                if 1 <= len(candidate.split()) <= 3:
                    result["patient_name"] = candidate
                    break

    # ── Name — English ────────────────────────────────────────────────────────
    elif language == "en":
        patterns = [
            r"my name is\s+(\w[\w\s]{1,20}?)(?:[,.]|\s+and|$)",
            r"i'm\s+(\w[\w\s]{1,20}?)(?:[,.]|$)",
            r"i am\s+(\w[\w\s]{1,20}?)(?:[,.]|$)",
            r"this is\s+(\w[\w\s]{1,20}?)(?:[,.]|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                candidate = m.group(1).strip().title()
                if 1 <= len(candidate.split()) <= 3:
                    result["patient_name"] = candidate
                    break

    if result:
        logger.debug(f"[Extractor] {result} ← '{text[:60]}'")

    return result
