import logging
import numpy as np
import whisper
from .base import ASRBackend
from config import WHISPER_MODEL

log = logging.getLogger(__name__)


class WhisperLocalASR(ASRBackend):
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        log.info("Loading local Whisper model: %s", model_name)
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio_np: np.ndarray) -> str:
        log.debug("Transcribing %.2fs of audio (local)", len(audio_np) / 16000)
        result = self._model.transcribe(audio_np, language="ne")
        text = result["text"].strip()
        log.debug("Transcription: %r", text)
        return text
