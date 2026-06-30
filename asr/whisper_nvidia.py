import logging
import numpy as np
import riva.client
from .base import ASRBackend
from config import ASR_KEY, ASR_NVCF_URI, ASR_FUNCTION_ID, SAMPLE_RATE

log = logging.getLogger(__name__)


class WhisperNvidiaASR(ASRBackend):
    def __init__(self) -> None:
        auth = riva.client.Auth(
            use_ssl=True,
            uri=ASR_NVCF_URI,
            metadata_args=[
                ["function-id", ASR_FUNCTION_ID],
                ["authorization", f"Bearer {ASR_KEY}"],
            ],
        )
        self._service = riva.client.ASRService(auth)
        self._config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=1,
            language_code="ne",
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )

    def transcribe(self, audio_np: np.ndarray) -> str:
        log.debug("Transcribing %.2fs of audio (nvidia)", len(audio_np) / SAMPLE_RATE)
        audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
        response = self._service.offline_recognize(audio_bytes, self._config, future=False)
        text = "".join(r.alternatives[0].transcript for r in response.results).strip()
        log.debug("Transcription: %r", text)
        return text
