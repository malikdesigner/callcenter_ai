"""
AI Hospital Receptionist — Sara AI

Ports:
  8000  English
  8001  Urdu
  8002  Language selector — Sara asks "English or Urdu?" and redirects the call
"""

import asyncio
import ctypes
import glob
import os
import sys

# Pre-load NVIDIA DLLs before ctranslate2 / torch are imported.
# nvidia-cublas-cu12 / nvidia-cudnn-cu12 install DLLs to site-packages/nvidia/*/bin
# but ctranslate2 uses LoadLibrary by name and won't find them unless pre-loaded.
if os.name == 'nt':
    _site = os.path.normpath(os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages'))
    _nvidia_bins = [
        os.path.normpath(d)
        for d in glob.glob(os.path.join(_site, 'nvidia', '*', 'bin'))
        if os.path.isdir(d)
    ]
    for _d in _nvidia_bins:
        os.add_dll_directory(_d)
    if _nvidia_bins:
        os.environ['PATH'] = ';'.join(_nvidia_bins) + ';' + os.environ.get('PATH', '')
    for _bin_dir in _nvidia_bins:
        for _dll_name in ['cublas64_12.dll', 'cublasLt64_12.dll', 'cudnn64_9.dll']:
            _dll_path = os.path.join(_bin_dir, _dll_name)
            if os.path.exists(_dll_path):
                try:
                    ctypes.CDLL(_dll_path)
                except OSError:
                    pass

import torch  # noqa: F401 — must import after DLL registration, before faster-whisper

import uvicorn
from loguru import logger


def check_prerequisites():
    warnings = []

    voice_path = os.getenv("VOICE_SAMPLE_PATH", "voices/receptionist.wav")
    if not os.path.exists(voice_path):
        warnings.append(
            f"  Voice sample not found at '{voice_path}' (optional — Edge-TTS is used instead)."
        )

    from config.settings import settings
    if not any([settings.gemini_api_key, settings.groq_api_key, getattr(settings, 'hugging_face_token', None)]):
        warnings.append(
            "  No cloud LLM key configured — falling back to local Ollama."
        )

    try:
        import httpx
        from config.settings import settings as _s
        r = httpx.get(f"{_s.ollama_host}/api/tags", timeout=2)
        model = _s.ollama_model
        if not any(model.split(":")[0] in m for m in [x["name"] for x in r.json().get("models", [])]):
            warnings.append(f"  Ollama model '{model}' not found — run: ollama pull {model}")
    except Exception:
        warnings.append("  Ollama not running (fallback only) — cloud LLM will be used.")

    if warnings:
        print("\n" + "="*55)
        print("  Warnings (non-fatal):")
        for w in warnings:
            print(w)
        print("="*55 + "\n")


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
    from src.api.server import create_app
    from src.pipeline.call_handler import CallHandler

    for port in [8000, 8001, 8002]:
        _free_port(port)

    from src.llm.agent import _load_hospital_knowledge
    _load_hospital_knowledge()

    logger.info("Loading models...")
    CallHandler.load_models()

    en_app = create_app("en")   # English only
    ur_app = create_app("ur")   # Urdu only
    bi_app = create_app("bi")   # Sara asks caller "English or Urdu?" → redirects

    servers = [
        uvicorn.Server(uvicorn.Config(en_app, host="0.0.0.0", port=8000, log_level="warning", reload=False)),
        uvicorn.Server(uvicorn.Config(ur_app, host="0.0.0.0", port=8001, log_level="warning", reload=False)),
        uvicorn.Server(uvicorn.Config(bi_app, host="0.0.0.0", port=8002, log_level="warning", reload=False)),
    ]

    logger.info("Sara AI ready:")
    logger.info("  English                → http://localhost:8000")
    logger.info("  Urdu                   → http://localhost:8001")
    logger.info("  Language selector call → http://localhost:8002")
    logger.info("Press Ctrl+C to stop.")

    await asyncio.gather(*[s.serve() for s in servers])


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Sara AI — City Medical Hospital")
    print("="*55)

    check_prerequisites()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        logger.info("Servers stopped.")
