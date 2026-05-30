"""
Entry point for the AI Hospital Receptionist.

Usage:
  1. Place your voice sample at:   voices/receptionist.wav
  2. Run:  python main.py
  3. Open: http://localhost:8000  (English)
           http://localhost:8001  (Urdu)
           http://localhost:8002  (Roman Urdu)
"""

import asyncio
import os
import sys

# CRITICAL FIX for Windows cuDNN Error 127:
# We MUST import torch before ANY other ML library (like faster_whisper/ctranslate2)
# is imported. Otherwise, ctranslate2 loads a conflicting cuDNN DLL first!
import torch

import uvicorn
from loguru import logger


def check_prerequisites():
    errors = []
    warnings = []

    # Voice sample (optional — Edge-TTS is used and does NOT need a local voice file)
    voice_path = os.getenv("VOICE_SAMPLE_PATH", "voices/receptionist.wav")
    if not os.path.exists(voice_path):
        warnings.append(
            f"  ⚠  Voice sample not found at '{voice_path}' (optional).\n"
            f"     → Edge-TTS (cloud) is active and does not need a local voice file.\n"
            f"     → This warning is safe to ignore unless you implement local voice cloning."
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
        from config.settings import settings as _s
        r = httpx.get(f"{_s.ollama_host}/api/tags", timeout=2)
        model = _s.ollama_model
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
    from src.pipeline.call_handler import CallHandler

    _free_port(8000)
    _free_port(8001)
    _free_port(8002)
    _free_port(8003)

    # Load models once here — both server startup events will find them already loaded
    # and skip, preventing the concurrent-load OOM that happens when both fire simultaneously.
    logger.info("Loading models (once for all servers)...")
    CallHandler.load_models()

    logger.info("Initializing multi-language support (Single-Process Mode)...")

    en_app = create_app("en")
    ur_app = create_app("ur")
    ro_app = create_app("ro")
    bi_app = create_app("bi")

    config_en = uvicorn.Config(en_app, host="0.0.0.0", port=8000,
                               log_level="info", reload=False)
    config_ur = uvicorn.Config(ur_app, host="0.0.0.0", port=8001,
                               log_level="info", reload=False)
    config_ro = uvicorn.Config(ro_app, host="0.0.0.0", port=8002,
                               log_level="info", reload=False)
    config_bi = uvicorn.Config(bi_app, host="0.0.0.0", port=8003,
                               log_level="info", reload=False)

    server_en = uvicorn.Server(config_en)
    server_ur = uvicorn.Server(config_ur)
    server_ro = uvicorn.Server(config_ro)
    server_bi = uvicorn.Server(config_bi)

    logger.info("Starting English Version    at http://localhost:8000")
    logger.info("Starting Urdu Version       at http://localhost:8001")
    logger.info("Starting Roman Urdu Version at http://localhost:8002")
    logger.info("Starting Bilingual          at http://localhost:8003")
    logger.info("All servers running. Press Ctrl+C to stop.")

    await asyncio.gather(server_en.serve(), server_ur.serve(), server_ro.serve(), server_bi.serve())


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AI Hospital Receptionist (Sara AI)")
    print("="*60)

    check_prerequisites()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Servers stopped.")
