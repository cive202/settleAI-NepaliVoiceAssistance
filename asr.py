from config import WHISPER_MODEL
import numpy as np
import whisper


class ASR:
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        self._model = self._load_model(model_name)

    def _load_model(self, model_name=WHISPER_MODEL):
        return whisper.load_model(model_name)

    def transcribe_nepali(self, audio_np: np.ndarray) -> str:
        """
        Transcribes a float32 numpy audio array to text.

        Args:
            audio_np: float32 numpy array at 16kHz
            language: ISO language code (default "ne" for Nepali)

        Returns:
            Transcribed string, or empty string if nothing detected.
        """
        print("transcribing nepali")
        result = self._model.transcribe(audio_np, language="ne")
        return result["text"].strip()
