"""
tts.py — Text-to-Speech with a swap-ready interface.

Responsibility:
  - Define a TTSEngine base class (easy to swap engines)
  - GTTSEngine: gTTS, fast but generic-sounding
  - RunPodIndicParlerTTS: ai4bharat/indic-parler-tts on a RunPod Serverless
    endpoint (tts_worker/) — currently the active engine (see TTS.__init__),
    native Nepali voice at higher per-sentence latency
  - Audio playback via pygame
"""

import base64
import io
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from functools import lru_cache

import pygame
import requests
from gtts import gTTS

from config import (
    TTS_BASE_URL,
    TTS_MODEL,
    TTS_POLL_TIMEOUT_S,
    TTS_RUNPOD_API_KEY,
    TTS_TIMEOUT_S,
    TTS_VOICE_DESCRIPTION,
)
from perf import timed

_DONE_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
_POLL_INTERVAL_S = 0.5


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes without playing."""
        pass


# Repeated replies (greetings especially) shouldn't re-hit Google's TTS endpoint.
@lru_cache(maxsize=128)
def _synthesize_gtts(text: str, lang: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, timeout=TTS_TIMEOUT_S).write_to_fp(buf)
    return buf.getvalue()


# ── gTTS implementation (current) ────────────────────────────────────────────
class GTTSEngine(TTSEngine):
    def __init__(self, lang: str = "ne"):
        self.lang = lang

    def synthesize(self, text: str) -> bytes:
        with timed("tts.gtts_synthesize"):
            return _synthesize_gtts(text, self.lang)

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(self.synthesize(text))
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            pygame.mixer.music.unload()
            os.unlink(tmp_path)


# ── indic-parler-tts on a RunPod Serverless queue-based endpoint ─────────────
def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    # synthesize() promises MP3 bytes (api.py's response contract is
    # `"audio": "<b64 mp3>"`) but the worker returns WAV, so transcode.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


class RunPodIndicParlerTTS(TTSEngine):
    def __init__(self, model: str = TTS_MODEL, description: str = TTS_VOICE_DESCRIPTION):
        self._model = model
        self._description = description
        self._base_url = TTS_BASE_URL
        # TTS() wraps this engine once at api.py startup and lives for the
        # whole process — a shared Session reuses one pooled HTTPS
        # connection to RunPod instead of paying a fresh TLS handshake
        # every call.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {TTS_RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            }
        )

    def synthesize(self, text: str) -> bytes:
        with timed("tts.runpod_synthesize"):
            # Always submit via /run and poll, rather than trying /runsync
            # first: RunPod's own logs showed queue-dispatch wait (time
            # between a job being queued and a worker picking it up, not GPU
            # processing — inference itself is near-instant once started)
            # regularly exceeding 10s even against a warm, idle worker. A
            # /runsync timeout doesn't hand back the job id it already
            # created server-side, so the old fallback ended up submitting a
            # duplicate job on every slow-dispatch case — which then queues
            # up and makes dispatch even slower for the rest of the reply.
            # /run always returns a job id in ~1-3s regardless of dispatch
            # latency, so there's nothing to fall back from.
            payload = {"input": {"text": text, "description": self._description}}
            resp = self._session.post(f"{self._base_url}/run", json=payload, timeout=TTS_TIMEOUT_S)
            resp.raise_for_status()
            body = self._await_completion(resp.json())
            if body["status"] == "FAILED":
                raise RuntimeError(f"RunPod TTS job failed: {body.get('error')}")

            wav_bytes = base64.b64decode(body["output"]["audio_base64"])
            return _wav_to_mp3(wav_bytes)

    def _await_completion(self, body: dict) -> dict:
        deadline = time.monotonic() + TTS_POLL_TIMEOUT_S
        while body.get("status") not in _DONE_STATUSES:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"RunPod TTS job {body.get('id')} did not finish within "
                    f"{TTS_POLL_TIMEOUT_S}s (last status: {body.get('status')})"
                )
            time.sleep(_POLL_INTERVAL_S)
            try:
                resp = self._session.get(f"{self._base_url}/status/{body['id']}", timeout=TTS_TIMEOUT_S)
                resp.raise_for_status()
                body = resp.json()
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                # A transient 5xx from RunPod's own status endpoint (observed
                # in practice) shouldn't kill the whole request — the job
                # itself is still running server-side regardless; just retry
                # the poll. A 4xx (e.g. job genuinely gone) is a real error.
                if isinstance(e, requests.exceptions.HTTPError) and (
                    e.response is None or e.response.status_code < 500
                ):
                    raise
        return body

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(self.synthesize(text))
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            pygame.mixer.music.unload()
            os.unlink(tmp_path)


# ── TTS — what main.py imports ───────────────────────────────────────────────
# ~4s/sentence on RunPod, vs. gTTS's <2s — accepted for the native Nepali
# voice quality. Change back to GTTSEngine() if latency becomes a problem.
class TTS:
    def __init__(self):
        self._engine: TTSEngine = RunPodIndicParlerTTS()

    def speak(self, text: str) -> None:
        print("Speaking...")
        self._engine.speak(text)

    def synthesize(self, text: str) -> bytes:
        return self._engine.synthesize(text)

    def swap_engine(self, engine: TTSEngine) -> None:
        self._engine = engine
