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
    gemini_model: str = "gemini-2.0-flash-lite"

    # LLM — Groq (FREE)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # LLM — HuggingFace Inference API (FREE)
    hugging_face_token: str = ""
    huggingface_model: str = "Qwen/Qwen2.5-72B-Instruct"

    # LLM — Ollama (local GPU — PRIMARY provider)
    ollama_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    # -1 = all layers on GPU (default). 0 = CPU only (slow, use only if GPU crashes).
    ollama_num_gpu: int = -1

    # STT
    whisper_model: str = "small"           # tiny / base / small / medium / large-v3
    whisper_device: str = "cpu"            # cuda or cpu
    whisper_compute_type: str = "int8_float16"  # int8_float16 (GPU, less VRAM) or int8 (CPU)

    # Audio
    sample_rate: int = 16000
    silence_duration: float = 1.8         # seconds of silence before processing speech

    # VAD
    vad_threshold: float = 0.35

    # Hospital config
    hospital_name: str = "City Medical Hospital"
    receptionist_name: str = "Sara"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
