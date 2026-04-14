"""
Entry point for the AI Hospital Receptionist.

Usage:
  1. Place your voice sample at:   voices/receptionist.wav
  2. Run:  python main.py
  3. Open: http://localhost:8000  (English)
           http://localhost:8001  (Urdu)
"""

import asyncio
import os
import sys

import uvicorn
from loguru import logger


def check_prerequisites():
    errors = []
    warnings = []

    # Voice sample (required)
    voice_path = os.getenv("VOICE_SAMPLE_PATH", "voices/receptionist.wav")
    if not os.path.exists(voice_path):
        errors.append(
            f"  ✗  Voice sample not found at '{voice_path}'\n"
            f"     → Update VOICE_SAMPLE_PATH in .env to point to your voice file.\n"
            f"     → WAV or MP3 accepted (min 6 sec, clear speech, no background noise)."
        )

    # Cloud LLM check — read via settings so .env values are picked up
    from config.settings import settings
    has_cloud_llm = bool(
        settings.hugging_face_token
        or settings.gemini_api_key
        or settings.groq_api_key
    )
    if not has_cloud_llm:
        warnings.append(
            "  ⚠  No cloud LLM configured (HUGGING_FACE_TOKEN / GEMINI_API_KEY / GROQ_API_KEY).\n"
            "     → Falling back to local Ollama — make sure it is running."
        )

    # Ollama reachability (optional — only used as fallback)
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(model.split(":")[0] in m for m in models):
            warnings.append(
                f"  ⚠  Ollama model '{model}' not found (fallback only).\n"
                f"     → Run:  ollama pull {model}"
            )
    except Exception:
        warnings.append(
            "  ⚠  Ollama not running (fallback only) — cloud LLM will be used instead."
        )

    if warnings:
        print("\n" + "="*60)
        print("  Warnings (non-fatal):")
        print("="*60)
        for w in warnings:
            print(w)
        print("="*60 + "\n")

    if errors:
        print("\n" + "="*60)
        print("  Prerequisites not met:")
        print("="*60)
        for e in errors:
            print(e)
        print("="*60 + "\n")
        sys.exit(1)


def _free_port(port: int):
    """Kill any process holding the given port (Windows only)."""
    import subprocess, platform
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.check_output(
            f'netstat -ano | findstr ":{port} "', shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in result.splitlines():
            parts = line.split()
            if "LISTENING" in parts:
                pid = parts[-1]
                subprocess.call(f"taskkill /PID {pid} /F", shell=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info(f"Released port {port} (killed PID {pid})")
    except Exception:
        pass


async def serve():
    """Run both language servers in a single process so models load only once."""
    from src.api.server import create_app
    _free_port(8000)
    _free_port(8001)

    logger.info("Initializing multi-language support (Single-Process Mode)...")

    en_app = create_app("en")
    ur_app = create_app("ur")

    config_en = uvicorn.Config(en_app, host="0.0.0.0", port=8000,
                               log_level="info", reload=False)
    config_ur = uvicorn.Config(ur_app, host="0.0.0.0", port=8001,
                               log_level="info", reload=False)

    server_en = uvicorn.Server(config_en)
    server_ur = uvicorn.Server(config_ur)

    logger.info("Starting English Version at http://localhost:8000")
    logger.info("Starting Urdu Version at http://localhost:8001")
    logger.info("Both servers running. Press Ctrl+C to stop.")

    await asyncio.gather(server_en.serve(), server_ur.serve())


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Hospital Receptionist (Sara AI)")
    print("="*60)

    check_prerequisites()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Servers stopped.")
