import logging

import numpy as np
import torch
from transformers import AutoModel

from .base import ASRBackend
from config import ASR_DECODING, ASR_MODEL

log = logging.getLogger(__name__)


class IndicConformerASR(ASRBackend):
    def __init__(self, model_name: str = ASR_MODEL, decoding: str = ASR_DECODING) -> None:
        log.info("Loading local IndicConformer model: %s (decoding=%s)", model_name, decoding)
        self.decoding = decoding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Gated repo — requires accepting the model's terms while logged in
        # on huggingface.co/ai4bharat/indic-conformer-600m-multilingual, and
        # a matching HF token available locally (huggingface-cli login or
        # HF_TOKEN env var).
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def transcribe(self, audio_np: np.ndarray) -> str:
        wav = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        text = self._model(wav, "ne", self.decoding)
        log.debug("Transcription: %r", text)
        return text.strip()
