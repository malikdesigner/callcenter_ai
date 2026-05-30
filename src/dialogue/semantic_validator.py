"""
Semantic Validator Layer

Analyzes the semantic validity of the transcript before state transition.
Ensures that the input makes sense given the current context and is not 
out-of-domain (e.g., talking about cars when asked for symptoms).
"""

import re
from loguru import logger

# Keywords that indicate the transcript is likely out-of-domain for a hospital context
OUT_OF_DOMAIN_KEYWORDS = {
    "ur": ["گاڑی", "موٹر", "سیاست", "کرکٹ", "فلم", "گانا", "خراب ہے", "میکینک"],
    "ro": ["gaari", "motor", "siyasat", "cricket", "film", "gana", "kharab hai", "mechanic"],
    "en": ["car", "motor", "politics", "cricket", "movie", "song", "broken car", "mechanic"]
}

# Heuristics for expected semantics per state
def is_semantically_valid(text: str, current_state: str, language: str = "ur") -> tuple[bool, str]:
    """
    Checks if the text is semantically valid for the given state.
    
    Returns:
        (is_valid, reason)
    """
    text = text.lower().strip()
    if not text:
        return False, "empty_text"

    # 1. Broad Out-Of-Domain Check (especially during symptom collection)
    if current_state == "ASK_SYMPTOM":
        ood_words = OUT_OF_DOMAIN_KEYWORDS.get(language, OUT_OF_DOMAIN_KEYWORDS["en"])
        for ood in ood_words:
            if ood in text:
                logger.warning(f"[Semantic Validator] Rejected '{text}' - Out of domain word detected: '{ood}'")
                return False, "out_of_domain"

    # 2. State-Specific Checks
    if current_state == "ASK_DATE":
        # Look for temporal semantics if possible
        temporal_hints_ur = ["آج", "کل", "پرسوں", "سوموار", "پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار", "تاریخ", "دن"]
        temporal_hints_ro = ["aaj", "kal", "parson", "somwar", "peer", "mangal", "budh", "jumerat", "jumma", "hafta", "itwar", "date", "din"]
        temporal_hints_en = ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "date", "day"]
        
        hints = temporal_hints_ur if language == "ur" else (temporal_hints_ro if language == "ro" else temporal_hints_en)
        # Also allow digits
        has_digit = any(char.isdigit() for char in text)
        has_temporal = any(hint in text for hint in hints)
        
        # If it's a very short response without any temporal hint or digit, it might be invalid
        if not has_temporal and not has_digit and len(text.split()) < 3:
            # We don't reject outright because user might say "next week" which isn't in hints,
            # but we can flag very short nonsense.
            pass

    elif current_state == "ASK_PHONE":
        # A phone response should typically contain digits or number words
        has_digit = any(char.isdigit() for char in text)
        number_words = ["صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو", "دس"]
        has_num_word = any(nw in text for nw in number_words)
        
        if not has_digit and not has_num_word and len(text) > 20:
            logger.warning(f"[Semantic Validator] Rejected '{text}' - Expected phone number but got long string without digits")
            return False, "invalid_phone_semantics"

    return True, "valid"
