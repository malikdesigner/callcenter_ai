"""
Entry point for the AI Hospital Receptionist.

Usage:
  1. Make sure Ollama is running:  ollama serve
  2. Place your voice sample at:   voices/receptionist.wav
  3. Run:  python main.py
  4. Open: http://localhost:8000
"""

import os
import sys

import uvicorn
from loguru import logger


def check_prerequisites():
    errors = []

    # Voice sample (WAV or MP3 accepted — auto-converted on first run)
    voice_path = os.getenv("VOICE_SAMPLE_PATH", "voices/receptionist.wav")
    if not os.path.exists(voice_path):
        errors.append(
            f"  ✗  Voice sample not found at '{voice_path}'\n"
            f"     → Update VOICE_SAMPLE_PATH in .env to point to your voice file.\n"
            f"     → WAV or MP3 accepted (min 6 sec, clear speech, no background noise)."
        )

    # Ollama reachability (non-blocking warning)
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        models = [m["name"] for m in r.json().get("models", [])]
        # Accept partial matches (e.g. "llama3.2:3b" matches "llama3.2:3b-instruct-q4_K_M")
        if not any(model.split(":")[0] in m for m in models):
            errors.append(
                f"  ✗  Ollama model '{model}' not found.\n"
                f"     → Run:  ollama pull {model}"
            )
    except Exception:
        errors.append(
            "  ✗  Ollama is not running or not reachable at http://localhost:11434\n"
            "     → Start it with:  ollama serve"
        )

    if errors:
        print("\n" + "="*60)
        print("  Prerequisites not met:")
        print("="*60)
        for e in errors:
            print(e)
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Hospital Receptionist")
    print("="*60)

    check_prerequisites()

    logger.info("Starting server at http://localhost:8000")
    logger.info("Open your browser and click 'Start Call' to test.")

    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
