"""
STT Filter Layer

Gatekeeper for Speech-to-Text output. Ensures that the pipeline doesn't process
low-confidence audio, background noise, or short gibberish.
"""

import re
from loguru import logger

def is_valid_input(text: str, confidence: dict, language: str = "ur") -> tuple[bool, str]:
    """
    Evaluates whether the transcribed text should be processed further.
    
    Returns:
        (is_valid, reason)
    """
    text = text.strip()
    avg_logprob = confidence.get("avg_logprob", 0.0)
    no_speech_prob = confidence.get("no_speech_prob", 0.0)
    
    # 1. Length check: Reject single letters or very short meaningless utterances
    if len(text) < 3:
        logger.warning(f"[STT Filter] Rejected '{text}' - Too short")
        return False, "too_short"
        
    # 2. Confidence thresholds
    # Whisper logprob is the natural log of token probability: prob = exp(logprob).
    # Phone numbers naturally score lower because digit sequences have less linguistic
    # context — we must not reject them with the same threshold as speech.
    import math
    prob = math.exp(avg_logprob)

    digit_count = sum(1 for c in text if c.isdigit())
    is_phone_context = digit_count >= 7  # likely a phone/ID number

    # Lenient threshold for phone numbers (0.35), strict for everything else (0.60)
    confidence_threshold = 0.35 if is_phone_context else 0.60
    if prob < confidence_threshold:
        logger.warning(
            f"[STT Filter] Rejected '{text}' - "
            f"Low confidence ({prob:.2f} < {confidence_threshold}) "
            f"[phone_context={is_phone_context}]"
        )
        return False, "low_confidence"

    if no_speech_prob > 0.4:
        logger.warning(f"[STT Filter] Rejected '{text}' - High no_speech probability ({no_speech_prob:.2f} > 0.4)")
        return False, "no_speech"
        
    # 3. Repeated noise/gibberish (e.g. "گفار ہوا ہے ۔ ۔ ۔ ۔")
    # Detect long strings of repeating characters or words
    if re.search(r'(.)\1{4,}', text):
        logger.warning(f"[STT Filter] Rejected '{text}' - Repeated characters")
        return False, "gibberish_repeated_chars"
        
    words = text.split()
    if len(words) > 3 and len(set(words)) == 1:
        logger.warning(f"[STT Filter] Rejected '{text}' - Repeated words")
        return False, "gibberish_repeated_words"
        
    return True, "valid"
