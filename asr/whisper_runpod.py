"""Client for the self-hosted Whisper endpoint in asr_worker/.

Mirrors tts.py's RunPodIndicParlerTTS: same queue-based job API (/run then
poll /status), same reasoning for always submitting via /run rather than
trying /runsync first.
"""

import base64
import io
import logging
import time

import numpy as np
import requests
import soundfile as sf

from .base import ASRBackend
from config import (
    ASR_POLL_TIMEOUT_S,
    ASR_RUNPOD_BASE_URL,
    ASR_RUNPOD_KEY,
    ASR_TIMEOUT_S,
    SAMPLE_RATE,
)
from perf import timed

log = logging.getLogger(__name__)

_DONE_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
_POLL_INTERVAL_S = 0.5


class RunPodWhisperASR(ASRBackend):
    def __init__(self) -> None:
        if not ASR_RUNPOD_BASE_URL:
            raise RuntimeError(
                "ASR_BACKEND=runpod but ASR_BASE_URL is unset — point it at "
                "https://api.runpod.ai/v2/<endpoint-id> (no trailing /run)"
            )
        self._base_url = ASR_RUNPOD_BASE_URL
        # Held for the process lifetime (api.py builds one ASR at startup), so
        # a shared Session reuses one pooled HTTPS connection to RunPod instead
        # of paying a fresh TLS handshake on every utterance.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {ASR_RUNPOD_KEY}",
                "Content-Type": "application/json",
            }
        )

    def transcribe(self, audio_np: np.ndarray) -> str:
        log.debug("Transcribing %.2fs of audio (runpod)", len(audio_np) / SAMPLE_RATE)
        buf = io.BytesIO()
        sf.write(buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        payload = {"input": {"audio_base64": base64.b64encode(buf.getvalue()).decode()}}

        with timed("asr.runpod_transcribe"):
            resp = self._session.post(f"{self._base_url}/run", json=payload, timeout=ASR_TIMEOUT_S)
            resp.raise_for_status()
            body = self._await_completion(resp.json())
            if body["status"] == "FAILED":
                raise RuntimeError(f"RunPod ASR job failed: {body.get('error')}")

        text = (body["output"].get("text") or "").strip()
        log.debug("Transcription: %r", text)
        return text

    def _await_completion(self, body: dict) -> dict:
        deadline = time.monotonic() + ASR_POLL_TIMEOUT_S
        while body.get("status") not in _DONE_STATUSES:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"RunPod ASR job {body.get('id')} did not finish within "
                    f"{ASR_POLL_TIMEOUT_S}s (last status: {body.get('status')})"
                )
            time.sleep(_POLL_INTERVAL_S)
            try:
                resp = self._session.get(
                    f"{self._base_url}/status/{body['id']}", timeout=ASR_TIMEOUT_S
                )
                resp.raise_for_status()
                body = resp.json()
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                # A transient 5xx from RunPod's own status endpoint shouldn't
                # kill the request — the job is still running server-side, so
                # just retry the poll. A 4xx (e.g. job genuinely gone) is real.
                if isinstance(e, requests.exceptions.HTTPError) and (
                    e.response is None or e.response.status_code < 500
                ):
                    raise
        return body
