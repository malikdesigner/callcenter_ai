"""
Voice Activity Detection using Silero VAD.
Detects whether an audio chunk contains human speech.
"""

import torch
import numpy as np
from loguru import logger
from config.settings import settings


class VADDetector:
    def __init__(self):
        logger.info("Loading Silero VAD...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils

        self.model.eval()
        self.threshold = settings.vad_threshold
        self.sample_rate = settings.sample_rate
        logger.info("Silero VAD loaded.")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Returns True if the chunk contains speech.
        audio_chunk: mono float32 numpy array.
        """
        if len(audio_chunk) == 0:
            return False

        tensor = torch.from_numpy(audio_chunk.astype(np.float32))

        # Silero expects chunks of exactly 512 (16 kHz) or 256 (8 kHz) samples
        chunk_size = 512
        speech_detected = False

        for i in range(0, len(tensor), chunk_size):
            piece = tensor[i : i + chunk_size]
            if len(piece) < chunk_size:
                piece = torch.nn.functional.pad(piece, (0, chunk_size - len(piece)))
            with torch.no_grad():
                prob = self.model(piece, self.sample_rate).item()
            if prob > self.threshold:
                speech_detected = True
                break

        return speech_detected

    def get_speech_segments(self, audio: np.ndarray):
        """Return a list of {start, end} timestamp dicts for speech regions."""
        tensor = torch.from_numpy(audio.astype(np.float32))
        return self.get_speech_timestamps(
            tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
        )

    def reset(self):
        """Reset internal state between calls."""
        self.model.reset_states()
