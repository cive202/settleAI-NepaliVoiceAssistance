import logging
import numpy as np
import riva.client
from riva.client.proto import riva_asr_pb2 as rasr
from .base import ASRBackend
from config import ASR_KEY, ASR_NVCF_URI, ASR_FUNCTION_ID, ASR_TIMEOUT_S, SAMPLE_RATE
from perf import timed

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
        # offline_recognize() doesn't expose a timeout, so call the gRPC stub
        # directly (as offline_recognize does internally) to bound worst-case
        # latency if the endpoint stalls.
        request = rasr.RecognizeRequest(config=self._config, audio=audio_bytes)
        with timed("asr.nvidia_recognize"):
            response = self._service.stub.Recognize(
                request,
                metadata=self._service.auth.get_auth_metadata(),
                timeout=ASR_TIMEOUT_S,
            )
        text = "".join(r.alternatives[0].transcript for r in response.results).strip()
        log.debug("Transcription: %r", text)
        return text
