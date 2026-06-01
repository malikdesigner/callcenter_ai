from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Database ───────────────────────────────────────────────────────────────
    # PostgreSQL (production): postgresql://user:pass@localhost/hospital_ai
    # SQLite (default/dev):    leave empty — auto-creates data/hospital.db
    database_url: str = ""

    # ── LLM — Ollama (PRIMARY — local GPU) ────────────────────────────────────
    # Recommended: ollama pull qwen3:8b   (~5 GB, good Urdu)
    # Best:        ollama pull qwen3:14b  (~9 GB, excellent Urdu)
    ollama_model:   str = "qwen3:8b"
    ollama_host:    str = "http://localhost:11434"
    ollama_num_gpu: int = -1     # -1 = all GPU layers; 0 = CPU only
    ollama_temperature: float = 0.3   # 0.3 gives consistent Urdu without being robotic

    # ── LLM — Cloud fallbacks (optional) ──────────────────────────────────────
    gemini_api_key:  str = ""
    gemini_model:    str = "gemini-2.0-flash-lite"
    groq_api_key:    str = ""
    groq_model:      str = "llama-3.3-70b-versatile"
    openai_api_key:  str = ""
    openai_model:    str = "gpt-4o-mini"
    hugging_face_token: str = ""

    # ── STT — Faster-Whisper ───────────────────────────────────────────────────
    # large-v3 for best accuracy; small/medium for faster machines
    whisper_model:        str = "large-v3"
    whisper_device:       str = "cuda"         # cuda | cpu
    whisper_compute_type: str = "int8_float16" # int8_float16 (GPU) | int8 (CPU)

    # ── TTS — Kokoro (local, English/Roman Urdu) ───────────────────────────────
    # pip install kokoro soundfile
    # Windows: also install espeak-ng from https://github.com/espeak-ng/espeak-ng/releases
    # Voices: bf_alice | bf_emma | af_bella | af_heart  (female, closest to South Asian accent)
    kokoro_voice: str = "bf_alice"
    kokoro_speed: float = 1.0

    # ── LLM — Anthropic Claude (fallback when Qwen3 quality check fails) ───────
    # Add ANTHROPIC_API_KEY to .env to enable. Leave blank to skip Claude fallback.
    # Fallback model: claude-haiku-4-5 (~$0.001/call — very cheap)
    # Budget: 10 fallback calls per session max (prevents runaway costs)
    anthropic_api_key:      str = ""
    claude_fallback_model:  str = "claude-haiku-4-5-20251001"
    claude_session_limit:   int = 3    # max Claude calls per session (hard cap)
    claude_monthly_budget:  float = 5.0  # informational USD limit (not enforced in-process)

    # ── TTS — ElevenLabs (optional, Urdu) ─────────────────────────────────────
    elevenlabs_api_key:     str = ""
    elevenlabs_voice_id_ur: str = ""
    elevenlabs_voice_id_en: str = ""

    # ── Audio ──────────────────────────────────────────────────────────────────
    sample_rate:      int   = 16000
    silence_duration: float = 1.8   # seconds of silence to trigger STT
    vad_threshold:    float = 0.35

    # ── Hospital ───────────────────────────────────────────────────────────────
    hospital_name:      str = "City Medical Hospital"
    receptionist_name:  str = "Sara"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
