from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Voice / TTS
    voice_sample_path: str = "voices/receptionist.wav"

    # LLM — OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # LLM — Gemini (FREE)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # LLM — Groq (FREE)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # LLM — Ollama (local, FREE)
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    # STT
    whisper_model: str = "small"           # tiny / base / small / medium / large-v3
    whisper_device: str = "cpu"            # cuda or cpu
    whisper_compute_type: str = "float16"   # float16 (GPU) or int8 (CPU)

    # Audio
    sample_rate: int = 16000
    silence_duration: float = 1.0         # seconds of silence before processing speech

    # VAD
    vad_threshold: float = 0.4

    # Hospital config
    hospital_name: str = "City Medical Hospital"
    receptionist_name: str = "Sara"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
