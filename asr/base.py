from abc import ABC, abstractmethod
import numpy as np


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_np: np.ndarray) -> str:
        pass
