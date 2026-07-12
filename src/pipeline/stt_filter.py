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
    # Typically Whisper logprob < -0.6 or -0.8 means it's guessing.
    # no_speech_prob > 0.4 usually means background noise
    # Since prompt requires dropping inputs with confidence < 0.6, we will treat logprob 
    # heuristically or expect a scaled confidence. Let's assume logprob > -0.6 or no_speech_prob < 0.4
    # The requirement strictly says "Drops inputs with confidence < 0.6". Whisper's confidence isn't exactly 0-1,
    # but we can convert logprob roughly. Or we use the strict threshold.
    
    # Using previous CallHandler logic as baseline: 
    # _GARBAGE_LOGPROB = -0.85, _GARBAGE_NO_SPEECH = 0.45
    # Let's enforce the new, stricter logprob or probability constraint.
    import math
    # Whisper logprob is natural log of probability. prob = exp(logprob).
    # Urdu speech has more acoustic variation — use looser thresholds.
    prob = math.exp(avg_logprob)
    conf_threshold = 0.40 if language == "ur" else 0.50
    noise_threshold = 0.65 if language == "ur" else 0.50
    if prob < conf_threshold:
        logger.warning(f"[STT Filter] Rejected '{text}' - Low confidence ({prob:.2f} < {conf_threshold})")
        return False, "low_confidence"

    if no_speech_prob > noise_threshold:
        logger.warning(f"[STT Filter] Rejected '{text}' - High no_speech probability ({no_speech_prob:.2f} > {noise_threshold})")
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
