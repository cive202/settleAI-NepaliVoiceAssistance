"""
api.py — FastAPI HTTP server wrapping the SettleAI voice pipeline.

Endpoints:
  POST /api/process  audio file → {user_text, assistant_text, tts_audio}
  POST /api/text     text body  → {user_text, assistant_text, tts_audio}
  POST /api/reset    clear conversation history
  GET  /api/health   readiness check
"""

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from asr import ASR
from config import API_KEY, LLM_MODEL, SAMPLE_RATE, SYSTEM_PROMPT, WHISPER_MODEL
from llm import LLM
from tts import GTTSEngine

_asr: ASR | None = None
_llm: LLM | None = None
_tts: GTTSEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr, _llm, _tts
    print("Loading models…")
    _asr = ASR(model_name=WHISPER_MODEL)
    _llm = LLM(api_key=API_KEY, model=LLM_MODEL, system_prompt=SYSTEM_PROMPT)
    _tts = GTTSEngine(lang="ne")
    print("All models ready.")
    yield


app = FastAPI(title="SettleAI Voice API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_audio(raw: bytes) -> np.ndarray:
    """Return float32 mono array at 16 kHz from any soundfile-compatible bytes."""
    audio, sr = sf.read(io.BytesIO(raw))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


class TextBody(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": _asr is not None}


@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...)):
    raw = await audio.read()
    try:
        audio_np = _load_audio(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")

    text = _asr.transcribe(audio_np)
    if not text:
        raise HTTPException(
            status_code=422, detail="Nothing transcribed — please try again"
        )

    reply = _llm.get_response(text)
    tts_bytes = _tts.synthesize(reply)

    return {
        "user_text": text,
        "assistant_text": reply,
        "tts_audio": base64.b64encode(tts_bytes).decode(),
    }


@app.post("/api/text")
async def process_text(body: TextBody):
    reply = _llm.get_response(body.text)
    tts_bytes = _tts.synthesize(reply)
    return {
        "user_text": body.text,
        "assistant_text": reply,
        "tts_audio": base64.b64encode(tts_bytes).decode(),
    }


@app.post("/api/reset")
def reset():
    _llm.reset_history()
    return {"status": "ok"}
