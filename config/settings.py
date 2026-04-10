from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Voice / TTS
    voice_sample_path: str = "voices/receptionist.wav"

    # LLM
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"

    # STT
    whisper_model: str = "medium"          # tiny / base / small / medium / large-v3
    whisper_device: str = "cuda"           # cuda or cpu
    whisper_compute_type: str = "float16"  # float16 (GPU) or int8 (CPU)

    # Audio
    sample_rate: int = 16000
    silence_duration: float = 1.5         # seconds of silence before processing speech

    # VAD
    vad_threshold: float = 0.5

    # Hospital config
    hospital_name: str = "City Medical Hospital"
    receptionist_name: str = "Sara"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
